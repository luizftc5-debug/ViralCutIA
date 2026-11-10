"""Full pipeline with real speech recognition. Needs libflite + a Whisper model download."""

import re

import pytest

pytestmark = pytest.mark.integration


def _run(client, worker, path, options):
    jid = client.post("/api/jobs", params={"filename": path.name}, content=path.read_bytes()).json()["id"]
    client.post(f"/api/jobs/{jid}/start", json=options)
    worker.run_once()
    return client.get(f"/api/jobs/{jid}").json()


def test_speech_video_end_to_end_with_captions(client, worker, speech_video):
    job = _run(
        client,
        worker,
        speech_video,
        {"captions": {"enabled": True, "style": "high_retention"}, "clip_count": 3},
    )
    assert job["status"] == "completed", job["error"]
    a = job["analysis"]
    assert a["language"] == "en" and a["words"] > 100 and a["engine"] == "heuristic"
    assert a["transcription"].startswith("faster-whisper:")
    assert {s["key"]: s["status"] for s in job["stages"]}["subtitles"] == "done"
    assert len(job["clips"]) >= 2
    scores = [c["score"] for c in job["clips"]]
    assert scores == sorted(scores, reverse=True)

    top = job["clips"][0]
    assert top["qa"]["passed"] and top["has_captions_burned"] and top["has_srt"]
    assert top["hook"]["types"], "the selected opening has an identified hook"
    assert top["titles"] and top["copy_source"] == "extractive"
    texts = " ".join(c["hook"]["text"].lower() for c in job["clips"])
    assert "why do most people" in texts or "three years ago" in texts
    assert not any(c["hook"]["text"].lower().startswith(("okay", "so anyway")) for c in job["clips"]), (
        "filler is not selected"
    )

    detail = client.get(f"/api/jobs/{job['id']}/clips/{top['id']}").json()
    assert detail["pages"] and detail["transcript"]
    srt = client.get(top["urls"]["srt"]).text
    assert re.search(r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}", srt)

    # ---- correct a caption and restyle; the edit must appear in the new SRT
    first = detail["pages"][0]
    r = client.patch(
        f"/api/jobs/{job['id']}/clips/{top['id']}",
        json={
            "page_edits": [{"index": 0, "text": "EDITED " + first["text"]}],
            "captions": {"enabled": True, "style": "clean_editorial", "position": "top"},
        },
    )
    assert r.status_code == 200
    worker.run_once()
    after = client.get(f"/api/jobs/{job['id']}/clips/{top['id']}").json()
    assert after["status"] == "ready" and after["version"] == 2 and after["qa"]["passed"]
    assert after["captions"]["style"] == "clean_editorial"
    assert "EDITED" in client.get(after["urls"]["srt"]).text


def test_captions_disabled_means_no_burn_in(client, worker, speech_video):
    job = _run(
        client, worker, speech_video, {"captions": {"enabled": False}, "clip_count": 1, "duration": "15-30"}
    )
    assert job["status"] == "completed", job["error"]
    clip = job["clips"][0]
    assert clip["has_captions_burned"] is False
    assert clip["has_srt"] is True, "the SRT sidecar is still available"
    assert 14 <= clip["end"] - clip["start"] <= 31
