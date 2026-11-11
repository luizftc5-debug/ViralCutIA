"""End to end on REAL human speech (FLEURS excerpts, CC-BY-4.0) in Portuguese and English.

Upload a video -> transcribe -> align -> trim -> burn captions -> verify against the audio.
Needs libass in ffmpeg and downloads the Whisper model on first use.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from syncutil import sync_metrics

from viralcut.config import get_settings
from viralcut.evaluation import wer
from viralcut.pipeline.orchestrator import Artifacts
from viralcut.pipeline.probe import MediaInfo
from viralcut.storage import get_storage

pytestmark = pytest.mark.integration

FIX = Path(__file__).parent / "fixtures" / "speech"
REFS = json.loads((FIX / "references.json").read_text(encoding="utf-8"))
# Largest word error rate each model may show on these clean read-speech fixtures. The bound
# tightens with the model; it is a regression alarm, not the quality claim (see docs/TRANSCRIPTION.md).
MAX_WER = {
    "tiny": 0.40,
    "base": 0.25,
    "small": 0.12,
    "medium": 0.10,
    "large-v3-turbo": 0.08,
    "large-v3": 0.08,
}
MODEL = os.environ.get("VIRALCUT_TEST_WHISPER_MODEL", "small")


@pytest.fixture(scope="module", autouse=True)
def _model():
    old = os.environ.get("VIRALCUT_WHISPER_MODEL")
    os.environ["VIRALCUT_WHISPER_MODEL"] = MODEL
    get_settings.cache_clear()
    yield
    if old is None:
        os.environ.pop("VIRALCUT_WHISPER_MODEL", None)
    else:
        os.environ["VIRALCUT_WHISPER_MODEL"] = old
    get_settings.cache_clear()


def _make_video(name: str, out: Path) -> Path:
    """1280x720 static picture + the real speech as AAC: the picture cannot hide caption pixels."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=0x24323f:s=1280x720:r=30:d={REFS[name]['duration'] + 0.5}",
         "-i", str(FIX / f"{name}.ogg"), "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)],
        check=True,
    )  # fmt: skip
    return out


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from viralcut.api.app import create_app
    from viralcut.worker.main import Worker

    with TestClient(create_app()) as client:
        yield client, Worker(), tmp_path_factory.mktemp("e2e")


def _run(env, name: str, captions: dict):
    client, worker, tmp = env
    video = _make_video(name, tmp / f"{name}.mp4")
    jid = client.post("/api/jobs", params={"filename": video.name}, content=video.read_bytes()).json()["id"]
    opts = {"captions": captions, "clip_count": 1, "duration": "15-30", "min_score": 0, "language": "auto"}
    assert client.post(f"/api/jobs/{jid}/start", json=opts).status_code in (200, 202)
    worker.run_once()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "completed", job["error"]
    return job


def _download(client, url: str, dest: Path) -> Path:
    dest.write_bytes(client.get(url).content)
    return dest


@pytest.fixture(scope="module", params=["pt_talk", "en_talk"])
def rendered(request, env):
    """Same clip twice: captions burned in (viral_dynamic) and captions off."""
    client, worker, tmp = env
    name = request.param
    job = _run(env, name, {"enabled": True, "style": "viral_dynamic"})
    clip = job["clips"][0]
    on = _download(client, clip["urls"]["video"], tmp / f"{name}_on.mp4")
    r = client.patch(
        f"/api/jobs/{job['id']}/clips/{clip['id']}",
        json={"captions": {"enabled": False, "style": "viral_dynamic"}},
    )
    assert r.status_code == 200
    worker.run_once()
    clip_off = client.get(f"/api/jobs/{job['id']}/clips/{clip['id']}").json()
    off = _download(client, clip_off["urls"]["video"], tmp / f"{name}_off.mp4")
    return name, job, clip, clip_off, on, off


def test_transcript_matches_what_was_said(rendered):
    name, job, *_ = rendered
    storage = get_storage()
    art = Artifacts.load(storage.analysis_dir(job["id"]), MediaInfo.from_dict(client_media(job)))
    ref = REFS[name]
    r = wer(ref["text"], art.transcript.text, lang=ref["language"])
    assert art.transcript.language == ref["language"], "language auto-detection"
    assert r.wer <= MAX_WER.get(MODEL, 0.15), f"WER {r.wer:.1%} with {MODEL}: {r.errors}"
    assert art.transcript.quality["dropped_segments"] == [] or all(
        d["reason"] for d in art.transcript.quality["dropped_segments"]
    )
    # accents and punctuation survive (the product promise), e.g. Portuguese "umidade", "mãos", "sensação"
    if name == "pt_talk":
        assert any(ch in art.transcript.text for ch in "ãçéí")
    assert any(w.text.endswith((".", ",", "?")) for w in art.transcript.words), "punctuation preserved"


def client_media(job):
    return {
        "duration": job["media"]["duration"],
        "width": job["media"]["width"],
        "height": job["media"]["height"],
        "fps": job["media"]["fps"],
        "video_codec": job["media"]["video_codec"],
        "has_audio": True,
    }


def test_words_were_forced_aligned_to_the_audio(rendered, env):
    client = env[0]
    name, job, clip, *_ = rendered
    alignment = client.get(f"/api/jobs/{job['id']}/clips/{clip['id']}").json()["alignment"]
    assert alignment["method"] == "ctc", f"forced alignment must have run: {alignment}"
    assert alignment["aligned"] >= 0.8 * alignment["words"] and alignment["chunks_rejected"] == 0
    assert alignment["median_shift_s"] != 0.0, "Whisper's own timestamps were corrected"
    assert MODEL in job["analysis"]["asr"]["model"] and job["analysis"]["transcription"].startswith(
        "faster-whisper:"
    )
    stages = {s["key"]: s["status"] for s in job["stages"]}
    assert stages["syncing"] == "done" and stages["subtitles"] == "done" and stages["rendering"] == "done"


def test_output_is_a_verified_vertical_mp4(rendered):
    *_, clip, clip_off, on, off = rendered
    for c, f in ((clip, on), (clip_off, off)):
        assert c["qa"]["passed"], c["qa"]["checks"]
        probe = json.loads(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(f)],
                capture_output=True,
                text=True,
            ).stdout
        )
        v = next(s for s in probe["streams"] if s["codec_type"] == "video")
        a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        assert (
            (v["width"], v["height"]) == (1080, 1920)
            and v["codec_name"] == "h264"
            and v["pix_fmt"] == "yuv420p"
        )
        assert a["codec_name"] == "aac" and int(a["sample_rate"]) == 48000
        assert abs(float(v["duration"]) - float(a["duration"])) < 0.1, "A/V in sync"
        assert -20 <= c["qa"]["integrated_lufs"] <= -9 and c["qa"]["true_peak_db"] <= -0.5


def test_subtitles_on_burns_captions_and_off_is_clean(rendered):
    *_, clip, clip_off, on, off = rendered
    assert clip["has_captions_burned"] is True and clip_off["has_captions_burned"] is False
    assert clip_off["has_srt"], "the SRT sidecar stays available when burn-in is off"
    from syncutil import gray_frames

    clean = gray_frames(off)
    # a static picture with no captions: every frame is the same frame
    assert float(abs(clean.astype("int16") - clean[0].astype("int16")).max()) <= 24, (
        "caption-off render must be clean"
    )
    burned = gray_frames(on)
    n = min(len(burned), len(clean))
    assert float(abs(burned[:n].astype("int16") - clean[:n].astype("int16")).max()) > 100, (
        "captions are visible when on"
    )


def test_captions_are_synchronised_with_the_audio(rendered):
    name, *_, on, off = rendered
    m = sync_metrics(on, off)
    assert m["caption_runs"] >= 3 and m["speech_runs"] >= 2
    assert m["speech_covered"] >= 0.90, f"speech without a caption on screen: {m}"
    assert m["stray_visible"] <= 0.06, f"captions on screen while nobody speaks: {m}"
    assert m["last_caption_off"] <= m["last_speech_end"] + 0.55, f"captions outlive the speech: {m}"


def test_transcript_review_before_rendering_applies_corrections_to_the_captions(env):
    """Pause after clip selection, correct a word everywhere, continue: nothing rendered before."""
    client, worker, tmp = env
    video = _make_video("pt_talk", tmp / "review.mp4")
    jid = client.post("/api/jobs", params={"filename": video.name}, content=video.read_bytes()).json()["id"]
    opts = {
        "captions": {"enabled": True, "style": "viral_dynamic"},
        "clip_count": 1,
        "duration": "15-30",
        "min_score": 0,
        "language": "pt",
        "review_transcript": True,
        "glossary": "Machu Picchu",
    }
    assert client.post(f"/api/jobs/{jid}/start", json=opts).status_code in (200, 202)
    worker.run_once()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "review", job["error"]
    assert all(c["status"] == "pending" and c["urls"] is None for c in job["clips"]), (
        "nothing is rendered before the review"
    )

    t = client.get(f"/api/jobs/{jid}/transcript").json()
    assert t["language"] == "pt" and len(t["words"]) > 20 and not t["edited"]
    target = next(w for w in t["words"] if w["text"].strip(".,").lower() == "concha")
    r = client.put(
        f"/api/jobs/{jid}/transcript", json={"replace": [{"find": "concha", "replace": "CONCHA-REVISADA"}]}
    )
    assert r.status_code == 200 and r.json()["changed"] == 1
    assert client.post(f"/api/jobs/{jid}/resume").json()["status"] == "queued"
    worker.run_once()

    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "completed", job["error"]
    clip = job["clips"][0]
    assert clip["status"] == "ready" and clip["qa"]["passed"] and clip["has_captions_burned"]
    srt = client.get(clip["urls"]["srt"]).text
    assert "CONCHA-REVISADA" in srt and "concha" not in srt.lower().replace("concha-revisada", "")
    # the corrected word keeps the time it was spoken at
    start = target["start"]
    t2 = client.get(f"/api/jobs/{jid}/transcript").json()
    fixed = next(w for w in t2["words"] if w["text"].startswith("CONCHA-REVISADA"))
    assert abs(fixed["start"] - start) < 0.05 and t2["edited"] is True
