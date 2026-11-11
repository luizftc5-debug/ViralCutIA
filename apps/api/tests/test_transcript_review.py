"""Manual transcript correction: text changes, timings and the original are preserved."""

import json

import pytest

from viralcut.db import Clip, Job, session_scope
from viralcut.pipeline.transcribe import Word
from viralcut.pipeline.transcript_edit import Replacement, apply_replacements, apply_word_edits
from viralcut.storage import get_storage


def _w(text, start, end=None):
    return Word(text, start, end if end is not None else start + 0.3)


def test_replacement_keeps_punctuation_and_timing():
    words = [_w("Hello,", 0.0), _w("welcome", 0.4), _w("to", 0.8), _w("viral", 1.2), _w("cut.", 1.6)]
    out, changed = apply_replacements(words, [Replacement("viral cut", "ViralCut")])
    assert [w.text for w in out] == ["Hello,", "welcome", "to", "ViralCut."]
    assert changed == [3]
    assert (out[3].start, out[3].end) == (1.2, 1.9), (
        "the new word spans exactly the time of the words it replaced"
    )


def test_replacement_that_adds_words_shares_the_time_span():
    words = [_w("the", 0.0), _w("gpt4", 0.4, 1.2), _w("model", 1.4)]
    out, changed = apply_replacements(words, [Replacement("gpt4", "GPT four")])
    assert [w.text for w in out] == ["the", "GPT", "four", "model"] and changed == [1, 2]
    assert out[1].start == 0.4 and out[2].end <= 1.2 + 1e-6 and out[1].end <= out[2].start


def test_replacement_is_whole_word_and_case_insensitive_by_default():
    words = [_w("cat", 0), _w("category", 1), _w("Cat", 2)]
    out, changed = apply_replacements(words, [Replacement("cat", "dog")])
    assert [w.text for w in out] == ["dog", "category", "dog"] and changed == [0, 2]
    out, _ = apply_replacements(words, [Replacement("cat", "dog", case_sensitive=True)])
    assert [w.text for w in out] == ["dog", "category", "Cat"]


def test_word_edits_by_index_change_text_only():
    words = [_w("a", 0), _w("b", 1)]
    out, changed = apply_word_edits(words, {1: "  Bee ", 5: "x", 0: "a"})
    assert [w.text for w in out] == ["a", "Bee"] and changed == [1]
    assert (out[1].start, out[1].end) == (1, 1.3)


@pytest.fixture()
def seeded(client):
    """A finished job with two clips and a transcript, written exactly as the pipeline stores them."""
    words = [
        {"text": t, "start": i * 0.5, "end": i * 0.5 + 0.4, "prob": 0.9}
        for i, t in enumerate("we use viral cut to make clips and viral cut is fast".split())
    ]
    with session_scope() as s:
        job = Job(status="completed", options={}, original_filename="v.mp4")
        s.add(job)
        s.flush()
        jid = job.id
        for rank, (a, b) in enumerate([(0.0, 3.0), (6.0, 9.0)], 1):
            s.add(Clip(job_id=jid, rank=rank, status="ready", start=a, end=b, version=1, qa={"passed": True}))
    d = get_storage().analysis_dir(jid)
    (d / "transcript.json").write_text(
        json.dumps({"language": "en", "engine": "test", "words": words, "quality": {}})
    )
    yield jid
    # the database and the worker are shared by the whole session: a seeded job that is still
    # "queued" would be picked up by another test's worker
    with session_scope() as s:
        job = s.get(Job, jid)
        if job is not None:
            s.delete(job)
    get_storage().delete_job(jid)


def test_global_correction_requeues_only_clips_that_contain_it(client, seeded):
    r = client.put(
        f"/api/jobs/{seeded}/transcript", json={"replace": [{"find": "viral cut", "replace": "ViralCut"}]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["changed"] == 2 and body["words"] == 10  # two occurrences, two words fewer
    t = client.get(f"/api/jobs/{seeded}/transcript").json()
    assert [w["text"] for w in t["words"]].count("ViralCut") == 2 and t["edited"] is True
    assert t["words"][2]["text"] == "ViralCut" and t["words"][2]["start"] == 1.0, (
        "timing of the corrected word is unchanged"
    )
    with session_scope() as s:
        clips = {c.rank: c.status for c in s.get(Job, seeded).clips}
    assert clips[1] == "queued", "the first clip contains the corrected words"
    assert body["affected_clips"] and len(body["affected_clips"]) >= 1
    original = json.loads((get_storage().analysis_dir(seeded) / "transcript.original.json").read_text())
    assert [w["text"] for w in original["words"]][2:4] == ["viral", "cut"], "the recogniser's output is kept"


def test_no_change_is_a_noop_and_wrong_state_is_refused(client, seeded):
    assert (
        client.put(
            f"/api/jobs/{seeded}/transcript", json={"replace": [{"find": "nothing", "replace": "x"}]}
        ).json()["changed"]
        == 0
    )
    with session_scope() as s:
        s.get(Job, seeded).status = "processing"
    assert client.get(f"/api/jobs/{seeded}/transcript").status_code == 409
    assert client.put(f"/api/jobs/{seeded}/transcript", json={}).status_code == 409


def test_review_state_can_be_corrected_without_queueing_renders_then_resumed(client, seeded):
    with session_scope() as s:
        job = s.get(Job, seeded)
        job.status = "review"
        for c in job.clips:
            c.status = "pending"
    r = client.put(f"/api/jobs/{seeded}/transcript", json={"words": [{"index": 2, "text": "Viral"}]})
    assert r.json()["changed"] == 1
    with session_scope() as s:
        assert {c.status for c in s.get(Job, seeded).clips} == {"pending"}, (
            "nothing renders before the user continues"
        )
    resumed = client.post(f"/api/jobs/{seeded}/resume").json()
    assert resumed["status"] == "queued"
    assert client.post(f"/api/jobs/{seeded}/resume").status_code == 409
    with session_scope() as s:
        assert s.get(Job, seeded).analysis["review"] == "resume"


def test_cancel_works_while_waiting_for_review(client, seeded):
    with session_scope() as s:
        s.get(Job, seeded).status = "review"
    assert client.post(f"/api/jobs/{seeded}/cancel").json()["status"] == "canceled"


def test_engine_decisions_are_explained_in_plain_language():
    from viralcut.api.serializers import decision_notes

    clip = Clip(
        job_id="j",
        start=0,
        end=10,
        framing={
            "notes": ["captions placed in the blurred band so they do not cover the picture"],
            "align": {"method": "ctc", "aligned": 40, "words": 42, "median_shift_s": 0.12},
            "edit": {
                "profile": {
                    "name": "education",
                    "evidence": ["education: 9 vocabulary hits (because, example)"],
                },
                "notes": ["removed 1.4s of silence (kept pauses up to 0.6 s)"],
                "zooms": [
                    {
                        "t0": 6.9,
                        "t1": 8.5,
                        "amp": 0.072,
                        "reason": "vocal emphasis",
                        "evidence": {"word": "never", "lift_db": 9.2},
                    }
                ],
                "audio": {"applied": ["high-pass 70 Hz"]},
            },
        },
    )
    text = " | ".join(decision_notes(clip))
    assert "Content type: education" in text and "Punch-in 6.9-8.5s, 7%" in text and "“never”" in text
    assert "40 of 42 words" in text and "+120 ms" in text and "Audio: high-pass 70 Hz." in text
    assert decision_notes(Clip(job_id="j", start=0, end=1, framing={})) == []


def test_browser_preflight_allows_every_method_the_studio_uses(client):
    """The studio calls the API from another origin: a method missing from CORS fails only in a browser."""
    for method in ("PUT", "PATCH", "POST", "DELETE"):
        r = client.options(
            "/api/jobs/x/transcript",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert r.status_code == 200, f"preflight for {method} was refused"
        assert method in r.headers["access-control-allow-methods"]
