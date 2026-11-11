"""Video styles, variations, Caption Studio overrides, resource profiles and the content cache."""

import os
import time

import pytest
from helpers import STRONG, WEAK, make_context

from viralcut import cache
from viralcut.options import CaptionOptions, JobOptions
from viralcut.pipeline import captions as cap
from viralcut.pipeline import scoring
from viralcut.pipeline.candidates import speech_candidates
from viralcut.pipeline.render import Segment, add_punch_transitions, jump_cuts
from viralcut.pipeline.styles import STYLES, edit_params, recommend, resolve
from viralcut.pipeline.transcribe import Word
from viralcut.pipeline.variations import plan_variations
from viralcut.resources import Hardware, plan_for, recommend as recommend_profile, use_profile, active_plan

# --------------------------------------------------------------------------- styles


def test_every_style_sets_real_parameters_and_intensity_scales_them():
    assert len(STYLES) == 9  # plus "auto" = the ten styles offered
    for key, st in STYLES.items():
        assert st.caption in cap.STYLES, key
        assert st.transition in ("cut", "punch", "smooth")
        assert st.audio in ("natural", "voice", "punchy")
    sub, bal, high = (edit_params("viral", i, "general") for i in ("subtle", "balanced", "high"))
    assert sub.max_pause > bal.max_pause > high.max_pause  # higher energy = tighter pacing
    assert sub.zoom["amp"] < bal.zoom["amp"] < high.zoom["amp"] <= 0.12
    assert sub.transition == "cut" and bal.transition == "punch"  # subtle never decorates cuts
    assert edit_params("gaming", "high", "general").max_pause is None  # gameplay rhythm is never trimmed
    assert resolve("dynamic") == "viral" and resolve("editorial") == "minimal" and resolve("bogus") == "auto"


def test_auto_style_follows_detected_content_and_keeps_profile_params():
    p = edit_params("auto", "balanced", "podcast", ["podcast: two people share the frame"], (0.7, {"allowed": True, "amp": 0.05, "min_gap": 8.0, "ramp": 0.45, "max_share": 0.25}))
    assert p.style == "podcast" and p.chosen_by == "auto" and "podcast" in p.reason
    assert p.max_pause == 0.7 and p.zoom["amp"] == 0.05  # tuned profile values, not the preset's
    assert p.caption == STYLES["podcast"].caption


def test_recommendation_is_explained_from_measurements():
    r = recommend({"hook": 0.8, "narrative": 0.3, "value": 0.4}, "general", wpm=175, motion_pct=0.3, face_ratio=0.9)
    assert r["style"] == "viral" and r["reasons"]
    r = recommend({"hook": 0.3, "narrative": 0.3, "value": 0.3}, "general", wpm=140, motion_pct=0.9, face_ratio=0.0)
    assert r["style"] == "gaming" and any("motion" in x for x in r["reasons"])


def test_job_options_custom_length_and_style_framing():
    o = JobOptions(duration="custom", custom_min=12, custom_max=25, style="gaming")
    assert o.duration_range == (12, 25) and o.default_framing == "blur"
    with pytest.raises(ValueError):
        JobOptions(duration="custom", custom_min=20, custom_max=22)
    assert JobOptions(style="dynamic").video_style == "viral"


# --------------------------------------------------------------------------- transitions


def test_punch_transitions_only_on_jump_cuts_in_tracked_crops():
    crop = {"type": "crop", "w": 608, "h": 1080, "y": 0, "keys": [(0.0, 600)]}
    segs = [Segment(0, 90, crop), Segment(120, 240, crop), Segment(240, 360, {"type": "blur"}), Segment(400, 520, crop), Segment(560, 700, crop)]
    t = 0.0
    for s in segs:
        s.out_start = t
        t += (s.f1 - s.f0) / 30
    assert jump_cuts(segs) == [1, 3, 4]  # 90->120, 360->400, 520->560; 240->240 is a real shot change
    n = add_punch_transitions(segs, 30, amp=0.07)
    assert n == 2  # every other eligible jump cut: a punch on every cut becomes a tic
    assert [bool(s.zooms) for s in segs] == [False, True, False, False, True]
    t0, t1, amp, ramp = segs[1].zooms[0]
    assert amp == 0.07 and t0 == pytest.approx(segs[1].out_start - ramp)  # already zoomed at the cut


# --------------------------------------------------------------------------- variations


def test_variations_reuse_the_moment_and_change_the_edit():
    ctx = make_context(STRONG + " | " + WEAK)
    cands = speech_candidates(ctx.sentences, ctx.words, ctx.media.duration, 8, 40)
    for c in cands:
        scoring.score_speech(c, ctx)
    c = max(cands, key=lambda c: c.score)
    vs = plan_variations(c, ctx, 3, "viral", "balanced", 8)
    assert len(vs) == 2 and [v.index for v in vs] == [2, 3]
    assert all(c.start <= v.start < v.end <= c.end for v in vs)
    kinds = {v.kind for v in vs}
    assert kinds <= {"hook_first", "style", "pacing"}
    style_v = [v for v in vs if v.kind == "style"]
    if style_v:
        assert style_v[0].style != "viral" and style_v[0].caption_style == STYLES[style_v[0].style].caption
    assert plan_variations(c, ctx, 1, "viral", "balanced", 8) == []


# --------------------------------------------------------------------------- Caption Studio


def _words(n=8):
    return [Word(f"word{i}", 0.2 + i * 0.35, 0.5 + i * 0.35) for i in range(n)]


def test_caption_overrides_reach_the_ass_file():
    base = CaptionOptions(style="high_retention")
    custom = base.model_copy(
        update={"words_per_page": 5, "max_lines": 1, "case": "normal", "text_color": "#112233", "background": "box", "animation_intensity": "strong", "outline": 2.0}
    )
    spec = cap.effective_spec(custom)
    assert spec.max_words == 5 and spec.max_lines == 1 and not spec.upper and spec.background and spec.outline == 2.0
    pages = cap.paginate(_words(), custom, 5.0)
    ass = cap.build_ass(pages, custom, "universal", 1080, 1920)
    assert "\\1c&H332211&" in ass  # text colour (ASS is BGR)
    assert "\\p1" in ass  # background panel drawing
    assert "word1" in ass and "WORD1" not in ass  # standard capitalisation
    default_pages = cap.paginate(_words(), base, 5.0)
    assert max(len(p.words) for p in pages) > max(len(p.words) for p in default_pages)


def test_offset_stays_inside_the_safe_area():
    lo = cap.make_layout(CaptionOptions(position="top", offset=-0.2), "tiktok", 1080, 1920)
    hi = cap.make_layout(CaptionOptions(position="bottom", offset=0.2), "tiktok", 1080, 1920)
    assert lo.center_y > 1920 * 0.10 and hi.center_y < 1920 * (1 - 0.22)


def test_animation_off_writes_no_transforms():
    o = CaptionOptions(style="energy", animation_type="none")
    ass = cap.build_ass(cap.paginate(_words(), o, 5.0), o, "universal", 1080, 1920)
    assert "\\t(" not in ass and "\\fad(" not in ass


# --------------------------------------------------------------------------- resources


def _hw(n, ram, gpu=None):
    return Hardware(n, n, ram, ram, gpu or {})


def test_profiles_leave_headroom_and_scale_with_the_machine():
    hw = _hw(8, 16)
    eco, bal, turbo = (plan_for(p, hw) for p in ("eco", "balanced", "turbo"))
    assert eco.threads == 4 and eco.nice > bal.nice > turbo.nice == 0 and not eco.keep_asr_model
    assert bal.threads == 8 and bal.nice > 0  # all cores, but yields to the user's own applications
    assert turbo.render_workers == 2 and turbo.threads == 4  # two renders share the machine, no oversubscription
    assert plan_for("turbo", _hw(2, 4)).render_workers == 1
    assert recommend_profile(_hw(2, 4))[0] == "eco"
    assert recommend_profile(_hw(4, 16))[0] == "balanced"
    assert recommend_profile(_hw(16, 64))[0] == "turbo"


def test_job_profile_override_is_scoped():
    before = active_plan().profile
    with use_profile("eco") as plan:
        assert plan.profile == "eco" and active_plan().profile == "eco"
    assert active_plan().profile == before


# --------------------------------------------------------------------------- cache


def test_content_cache_round_trip_and_purge(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 1000)
    d1 = cache.file_digest(f)
    assert cache.file_digest(f) == d1 and (tmp_path / "a.bin.digest").exists()
    f.write_bytes(b"y" * 1000)
    os.utime(f, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    assert cache.file_digest(f) != d1  # changed content -> new key, never a stale hit
    key = cache.make_key("t", d1)
    cache.put_json("transcripts", key, {"words": [1, 2]})
    assert cache.get_json("transcripts", key) == {"words": [1, 2]}
    assert cache.purge(max_age_hours=0) >= 1
    assert cache.get_json("transcripts", key) is None


# --------------------------------------------------------------------------- through the API


def test_preview_endpoint_and_style_edits(client, worker, silent_video):
    r = client.post("/api/caption-preview", json={"captions": {"style": "cinematic", "background": "box"}})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"

    jid = client.post("/api/jobs", params={"filename": "s.mp4"}, content=silent_video.read_bytes()).json()["id"]
    opts = {"clip_count": 1, "min_score": 0, "variations": 2, "style": "cinematic", "intensity": "subtle", "performance": "eco"}
    assert client.post(f"/api/jobs/{jid}/start", json=opts).status_code == 200
    worker.run_once()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "completed", job["error"]
    assert len(job["clips"]) == 2, "one clip and one variation of it"
    main, var = job["clips"]
    assert main["variant"] is None and var["variant"]["index"] == 2 and var["variant"]["of"] == main["id"]
    assert main["applied_style"]["style"] == "cinematic" and main["qa"]["passed"] and var["qa"]["passed"]
    assert main["recommendation"]["style"] and main["recommendation"]["reasons"]

    r = client.patch(f"/api/jobs/{jid}/clips/{main['id']}", json={"style": "viral", "intensity": "high"})
    assert r.status_code == 200 and r.json()["style"] == "viral"
    worker.run_once()
    clip = client.get(f"/api/jobs/{jid}/clips/{main['id']}").json()
    assert clip["status"] == "ready" and clip["version"] == 2 and clip["applied_style"]["style"] == "viral"


def test_database_from_an_older_version_gets_the_new_column(tmp_path):
    from sqlalchemy import create_engine, inspect, text

    from viralcut.db import _add_missing_columns

    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE clips (id VARCHAR(32) PRIMARY KEY, framing JSON)"))
    _add_missing_columns(eng)
    _add_missing_columns(eng)  # idempotent
    assert "edit" in {col["name"] for col in inspect(eng).get_columns("clips")}
