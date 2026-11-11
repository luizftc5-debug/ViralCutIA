#!/usr/bin/env python3
"""Measure speech-recognition quality (Word Error Rate) of the real pipeline on reference audio.

Input is a FLEURS-style manifest (https://huggingface.co/datasets/google/fleurs, CC-BY-4.0):
a TSV with ``id, file, raw transcription, ...`` columns plus the audio files, or a JSON list of
``{"audio": "path", "text": "reference", "lang": "pt"}`` objects.

    python scripts/benchmark_transcription.py --tsv test.tsv --audio-dir audio/ --lang pt \
        --limit 20 --model small --out report.json

Every utterance goes through exactly the code the application runs: ffmpeg decode to 16 kHz
mono, ``viralcut.pipeline.transcribe.transcribe``. Model, VAD, mode and the other knobs come
from the same ``VIRALCUT_*`` environment variables, so a configuration measured here is the
configuration that ships.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))


def load_manifest(args: argparse.Namespace) -> list[dict]:
    items: list[dict] = []
    if args.json:
        items = json.loads(Path(args.json).read_text(encoding="utf-8"))
    else:
        audio_dir = Path(args.audio_dir)
        index = {p.name: p for p in audio_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}}
        with open(args.tsv, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) > 2 and row[1] in index:
                    n = int(row[5]) if len(row) > 5 and row[5].isdigit() else 0  # FLEURS: samples in the recording
                    items.append({"audio": str(index[row[1]]), "text": row[2], "lang": args.lang, "n_samples": n})
    # a spread across the file rather than the first N consecutive rows (which share one sentence)
    step = max(1, len(items) // args.limit) if args.limit else 1
    return items[::step][: args.limit or None]


def edge_summary(edges: list[dict]) -> dict | None:
    """Start error is signed (negative: the caption appears before the sound). The end is reported
    one-sided: only a word that ends *after* the last audible instant would make a caption
    linger; an earlier end is not a defect (and the reference end includes noise tails)."""
    import numpy as np

    if not edges:
        return None
    st = np.array([x for e in edges for x in e["start_err"]])
    en = np.array([x for e in edges for x in e["end_err"]])
    return {
        "utterances": int(len(st)),
        "start_mean_abs_ms": round(float(np.mean(np.abs(st))) * 1000),
        "start_median_ms": round(float(np.median(st)) * 1000),
        "start_p90_abs_ms": round(float(np.percentile(np.abs(st), 90)) * 1000),
        "start_early_over_200ms": f"{int(np.sum(st < -0.2))}/{len(st)}",
        "end_late_over_200ms": f"{int(np.sum(en > 0.2))}/{len(en)}",
    }


def edge_errors(words, spans) -> dict | None:
    """Timing error of the first/last word of every utterance against the true audible edges."""
    starts, ends = [], []
    for sp in spans:
        inside = [w for w in words if sp["t0"] - 0.3 <= (w.start + w.end) / 2 <= sp["t1"] + 0.3]
        if inside:
            starts.append(inside[0].start - sp["v0"])  # negative: caption early
            ends.append(inside[-1].end - sp["v1"])  # positive: caption lingers
    if not starts:
        return None
    return {"start_err": [round(x, 3) for x in starts], "end_err": [round(x, 3) for x in ends]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tsv")
    src.add_argument("--json")
    ap.add_argument("--audio-dir", default=".")
    ap.add_argument("--lang", default="auto", help="language of the corpus (reference language)")
    ap.add_argument("--force-language", action="store_true", help="pass the language to the recogniser instead of auto-detecting")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--model", default=None, help="override VIRALCUT_WHISPER_MODEL")
    ap.add_argument("--mode", default=None, help="override VIRALCUT_TRANSCRIPTION_MODE")
    ap.add_argument("--align", action="store_true", help="also run forced alignment and report timing before/after")
    ap.add_argument("--only", default="", help="substring filter on the item name (JSON manifests)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--show-errors", type=int, default=12)
    args = ap.parse_args()

    if args.model:
        os.environ["VIRALCUT_WHISPER_MODEL"] = args.model
    if args.mode:
        os.environ["VIRALCUT_TRANSCRIPTION_MODE"] = args.mode
    os.environ.setdefault("VIRALCUT_DATA_DIR", str(Path("/tmp") / "viralcut-bench-data"))

    from viralcut.evaluation import top_errors, wer
    from viralcut.pipeline.align import align_words, refine_with_energy
    from viralcut.pipeline.audio import analyze_audio, decode_audio
    from viralcut.pipeline.transcribe import transcribe

    items = load_manifest(args)
    if args.only:
        items = [i for i in items if any(o in i.get("name", i["audio"]) for o in args.only.split(","))]
    print(f"{len(items)} utterances, model={os.environ.get('VIRALCUT_WHISPER_MODEL', 'default')}", flush=True)
    rows, results = [], []
    audio_s = decode_s = 0.0
    ref_all: list[str] = []
    hyp_all: list[str] = []
    lang_ok = 0
    outside_total = 0
    edge_all: list[dict] = []
    edge_aligned: list[dict] = []
    t_start = time.perf_counter()
    for i, it in enumerate(items, 1):
        t0 = time.perf_counter()
        samples = decode_audio(Path(it["audio"]), None, 0.0)
        dur = len(samples) / 16000
        if it.get("n_samples") and len(samples) < 0.97 * it["n_samples"]:
            print(f"[{i:>3}/{len(items)}] skipped {Path(it['audio']).name}: truncated download ({len(samples)} of {it['n_samples']} samples)", flush=True)
            continue
        lang = it.get("lang") if args.force_language and it.get("lang") not in (None, "auto") else None
        tr = transcribe(samples, lang)
        took = time.perf_counter() - t0
        hyp = " ".join(w.text for w in tr.words)
        r = wer(it["text"], hyp, lang=it.get("lang") or "")
        results.append(r)
        ref_all.append(it["text"])
        hyp_all.append(hyp)
        audio_s += dur
        decode_s += took
        want = (it.get("lang") or "")[:2]
        lang_ok += int(bool(want) and tr.language[:2] == want)
        outside = None
        if "spans" in it:  # ground-truth utterance spans: words outside them are invented or badly timed
            outside = [w.text for w in tr.words if not any(s["t0"] - 0.35 <= (w.start + w.end) / 2 <= s["t1"] + 0.35 for s in it["spans"])]
            outside_total += len(outside)
        edge = edge_errors(tr.words, it["spans"]) if it.get("spans") and "v0" in it["spans"][0] else None
        if edge:
            edge_all.append(edge)
        if args.align and edge:
            aw, _ = align_words(tr.words, samples, tr.language)
            aw, _ = refine_with_energy(aw, analyze_audio(samples))
            edge_aligned.append(edge_errors(aw, it["spans"]))
        rows.append({"audio": Path(it["audio"]).name, "edges": edge, "word_times": [[w.text, w.start, w.end] for w in tr.words], "kind": it.get("kind", ""), "dur": round(dur, 2), "wer": round(r.wer, 4), "words": len(tr.words), "outside_spans": outside, "lang": tr.language, "ref": it["text"], "hyp": hyp, "sec": round(took, 2)})
        extra = f"  words outside speech: {len(outside)}" if outside is not None else ""
        label = it.get("name", Path(it["audio"]).name)
        shown = f"WER {r.wer:5.1%}" if r.ref_words else f"words={len(tr.words):>3}"
        print(f"[{i:>3}/{len(items)}] {label:<22} {shown}  {dur:5.1f}s audio in {took:5.1f}s  lang={tr.language}{extra}", flush=True)

    total = wer(" ".join(ref_all), " ".join(hyp_all), lang=(items[0].get("lang") or "") if items else "")
    corpus = {
        "wer_corpus": round(total.wer, 4),
        "wer_mean_utterance": round(sum(r.wer for r in results) / max(1, len(results)), 4),
        "substitutions": sum(r.substitutions for r in results),
        "deletions": sum(r.deletions for r in results),
        "insertions": sum(r.insertions for r in results),
        "ref_words": sum(r.ref_words for r in results),
        "words_outside_speech": outside_total,
        "edge_timing": edge_summary(edge_all),
        "edge_timing_aligned": edge_summary(edge_aligned),
        "audio_seconds": round(audio_s, 1),
        "processing_seconds": round(decode_s, 1),
        "realtime_factor": round(decode_s / max(audio_s, 1e-9), 3),
        "language_detected_correctly": f"{lang_ok}/{len(items)}",
        "model": os.environ.get("VIRALCUT_WHISPER_MODEL", "default"),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
    }
    print(json.dumps(corpus, indent=2))
    print("most frequent errors (kind, reference, hypothesis):")
    for (kind, ref, hyp), n in top_errors(results, args.show_errors):
        print(f"  {n}x {kind:>3}  {ref!r} -> {hyp!r}")
    if args.out:
        args.out.write_text(json.dumps({"summary": corpus, "utterances": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
