"""Creator Studio: editors, presets, effects, assets and combined renders through the API."""

import io
import json
import subprocess

import pytest
from PIL import Image

from viralcut.options import JobOptions
from viralcut.pipeline import effects, growth
from viralcut.pipeline.editors import EDITORS, PRESETS, EditorsConfig, apply_preset
from viralcut.pipeline.probe import MediaInfo
from viralcut.pipeline.render import ImageCard, MusicBed, RenderExtras, Segment, build_filtergraph
from viralcut.pipeline.transcribe import Word


def _w(text, t0=0.0, step=0.4):
    out = []
    for tok in text.split():
        out.append(Word(tok, round(t0, 3), round(t0 + step * 0.8, 3)))
        t0 += step
    return out


def test_thirteen_editors_with_real_config_and_twelve_presets():
    keys = [e["key"] for e in EDITORS]
    assert len(keys) == 13 and len(set(keys)) == 13
    assert {
        "corte",
        "referencia",
        "lettering",
        "3d",
        "trilha",
        "tela-verde",
        "animacao",
        "tomada",
        "legenda",
        "prova",
        "ganchos",
        "texto-atras",
    } <= set(keys)
    cfg = EditorsConfig.model_validate({k: {} for k in keys})
    dumped = cfg.model_dump(by_alias=True)
    assert set(dumped) == set(keys)  # every editor has a configuration block
    assert len(PRESETS) == 12
    # presets differ in real parameters, not only in name
    sigs = {json.dumps([p["options"], p["editors"]], sort_keys=True) for p in PRESETS.values()}
    assert len(sigs) == 12
    for key in PRESETS:
        JobOptions.model_validate(apply_preset({}, key))  # every preset yields valid options


def test_preset_keeps_user_assets_and_count():
    opts = apply_preset({"clip_count": 3, "editors": {"trilha": {"asset": "a" * 32}}}, "documentary")
    assert opts["clip_count"] == 3 and opts["editors"]["trilha"]["asset"] == "a" * 32
    assert opts["editors"]["trilha"]["enabled"] and opts["style"] == "cinematic"


def test_retakes_and_fillers_are_cut_and_the_last_take_kept():
    words = _w("So the main reason is. So the main reason is practice every day. um and that works.")
    retakes = effects.retake_cuts(words, "en", 0.6)
    assert len(retakes) == 1 and retakes[0][2].startswith("So the main reason is.")
    fillers = effects.filler_cuts(words, "en")
    assert len(fillers) == 1
    keeps = effects.subtract([(0.0, 20.0)], [(a, b) for a, b, _ in retakes] + fillers)
    assert len(keeps) == 2 and keeps[0][0] > 1.5  # the first take (start of the list) is gone
    kept = effects.drop_words(words, [(a, b) for a, b, _ in retakes] + fillers)
    text = " ".join(w.text for w in kept)
    assert text.startswith("So the main reason is practice") and " um " not in f" {text} "


def test_overlay_ass_layers():
    t = effects.title_events(
        "A really long title that has to wrap over lines", 1080, 1920, animation="typewriter"
    )
    assert len(t) > 10 and all(e.startswith("Dialogue:") for e in t)  # one event per revealed character
    bar = effects.progress_bar_events(20.0, 1080, 1920, "#FF0000", 0.1)
    assert "\\clip(0," in bar[1] and "\\t(0,20000" in bar[1]
    behind = effects.behind_text_events("focus", 10.0, 1080, 1920, "#FFFFFF", y_frac=0.2)
    assert "FOCUS" in behind[0] and "\\pos(540,384)" in behind[0]
    ass = effects.merge_ass(None, t + bar, 1080, 1920)
    assert ass.startswith("[Script Info]") and ass.count("Dialogue:") == len(t) + 2


def test_depth_filter_cost_tiers():
    assert "perspective" not in effects.depth_filter("subtle", 10, 1080, 1920, 30)
    assert "perspective" in effects.depth_filter("strong", 10, 1080, 1920, 30)


def test_render_graph_with_cards_and_music():
    media = MediaInfo.from_dict({"duration": 30, "width": 1920, "height": 1080, "fps": "30/1", "has_audio": True,
                                 "video_codec": "h264", "audio_codec": "aac", "sar": "1:1"})  # fmt: skip
    from pathlib import Path

    ex = RenderExtras(
        video_post=["null"], cards=[ImageCard(Path("p.png"), 2.0, 5.0)], music=MusicBed(Path("m.mp3"))
    )
    ex.inputs(1, 30, 10.0)
    segs = [Segment(0, 300, {"type": "fit", "full": True})]
    g = build_filtergraph(segs, media, __import__("fractions").Fraction(30), 1920, 1080, None, True, None, ex)
    assert "[1:v]scale=" in g and "overlay=x=(W-w)/2" in g and "enable='between(t,2.000,5.000)'" in g
    assert "[2:a]atrim" in g and "sidechaincompress" in g and g.count("[aout]") == 1


def test_growth_toolkit_is_extractive():
    g = growth.build(
        text="Never build anything before you talk to ten customers. Today every product starts with ten conversations.",
        titles=["Talk to ten customers first"], caption="", hook_text="Never build anything before you talk to ten customers",
        language="en", global_keywords=["customers"], category="education", content_type="shorts", followups=["Why practice works"], variants=[],
    )  # fmt: skip
    assert len(g["titles"]) == 3 and "customers" in g["keywords"] and g["thumbnail_text"]
    assert "#customers" in g["hashtags"] and "guarantee" in g["note"]


def test_content_types_and_lengths():
    assert JobOptions(content_type="youtube").output_aspect == "16:9"
    assert JobOptions(content_type="compilation", aspect="16:9").output_aspect == "16:9"
    assert JobOptions(content_type="tiktok").output_aspect == "9:16"
    JobOptions(content_type="youtube", duration="custom", custom_min=120, custom_max=900)
    with pytest.raises(ValueError):
        JobOptions(content_type="shorts", duration="custom", custom_min=20, custom_max=600)


# --------------------------------------------------------------------------- through the API


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "#2a9d8f").save(buf, "PNG")
    return buf.getvalue()


def _mp3(tmp_path) -> bytes:
    p = tmp_path / "m.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=8", "-c:a", "libmp3lame", str(p)],
        check=True,
    )
    return p.read_bytes()


def test_assets_history_and_a_combined_render(client, worker, silent_video, tmp_path):
    jid = client.post("/api/jobs", params={"filename": "s.mp4"}, content=silent_video.read_bytes()).json()[
        "id"
    ]
    bad = client.post(
        f"/api/jobs/{jid}/assets", params={"kind": "image", "filename": "x.png"}, content=b"not an image"
    )
    assert bad.status_code == 422
    img = client.post(
        f"/api/jobs/{jid}/assets", params={"kind": "image", "filename": "proof.png"}, content=_png()
    ).json()
    mus = client.post(
        f"/api/jobs/{jid}/assets", params={"kind": "music", "filename": "m.mp3"}, content=_mp3(tmp_path)
    ).json()
    assert img["width"] == 400 and mus["duration"] > 7
    assert {a["id"] for a in client.get(f"/api/jobs/{jid}/assets").json()["assets"]} == {img["id"], mus["id"]}

    editors = {
        "lettering": {"enabled": True, "text": "Combined test", "animation": "pop"},
        "animacao": {"enabled": True},
        "prova": {"enabled": True, "items": [{"asset": img["id"], "at": 1.0, "duration": 2.0}]},
        "trilha": {"enabled": True, "asset": mus["id"], "beat_sync": False},
        "3d": {"enabled": True, "strength": "subtle"},
    }
    r = client.post(
        f"/api/jobs/{jid}/start",
        json={"clip_count": 1, "min_score": 0, "editors": editors, "performance": "eco"},
    )
    assert r.status_code == 200
    worker.run_once()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "completed", job["error"]
    clip = job["clips"][0]
    assert clip["qa"]["passed"]
    notes = " ".join(clip["framing_notes"])
    for marker in ("/lettering", "/animacao", "/prova", "/trilha", "/3d"):
        assert marker in notes, marker
    assert job["options"]["editors"]["3d"]["enabled"] is True  # stored with the editors' own names

    hist = client.get("/api/jobs").json()["jobs"]
    assert any(j["id"] == jid and j["clip_count"] == 1 and j["thumbnail"] for j in hist)

    # per-clip override: turn the music off for this clip only and re-render
    r = client.patch(f"/api/jobs/{jid}/clips/{clip['id']}", json={"editors": {"trilha": {"enabled": False}}})
    assert r.status_code == 200
    worker.run_once()
    clip2 = client.get(f"/api/jobs/{jid}/clips/{clip['id']}").json()
    assert (
        clip2["version"] == 2
        and clip2["qa"]["passed"]
        and "/trilha: music" not in " ".join(clip2["framing_notes"])
    )
