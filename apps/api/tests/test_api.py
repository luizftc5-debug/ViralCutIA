"""HTTP API and worker, on real (generated) media."""

import io
import zipfile

from viralcut.db import Job, session_scope
from viralcut.storage import get_storage

FAST = {"captions": {"enabled": False}, "duration": "auto", "clip_count": 2, "min_score": 0}


def _upload(client, path, name=None):
    return client.post("/api/jobs", params={"filename": name or path.name}, content=path.read_bytes())


def test_health_and_config(client):
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["limits"]["formats"] == ["MP4", "MOV", "MKV", "WEBM"]
    assert len(cfg["caption_styles"]) == 4
    assert cfg["engine"]["llm_enabled"] is False and cfg["engine"]["narrative_analysis"] == "heuristic"


def test_upload_validation(client, silent_video, tmp_path, monkeypatch):
    data = silent_video.read_bytes()
    assert client.post("/api/jobs", params={"filename": "x.exe"}, content=data).status_code == 422
    assert (
        client.post("/api/jobs", params={"filename": "x.mkv"}, content=data).status_code == 422
    )  # content/extension mismatch
    assert (
        client.post(
            "/api/jobs", params={"filename": "x.mp4"}, content=b"#EXTM3U\nfile:///etc/passwd\n" * 10
        ).status_code
        == 422
    )
    assert client.post("/api/jobs", params={"filename": "x.mp4"}, content=b"").status_code == 422
    fake = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 4096
    r = client.post("/api/jobs", params={"filename": "x.mp4"}, content=fake)
    assert r.status_code == 422 and "readable video" in r.json()["detail"]

    from viralcut.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    assert _upload(client, silent_video).status_code == 413
    monkeypatch.setattr(get_settings(), "max_upload_mb", 100)
    monkeypatch.setattr(get_settings(), "max_duration_s", 5)
    r = _upload(client, silent_video)
    assert r.status_code == 422 and "longer than" in r.json()["detail"]
    # rejected uploads leave nothing behind
    assert not any(get_storage().root.iterdir())


def test_unknown_and_malformed_ids(client):
    assert client.get("/api/jobs/" + "0" * 32).status_code == 404
    assert client.get("/api/jobs/not-an-id").status_code == 404
    assert client.get("/api/jobs/..%2F..%2Fetc").status_code == 404
    assert client.get(f"/api/jobs/{'0' * 32}/clips/{'1' * 32}/video").status_code == 404


def test_full_flow_without_speech(client, worker, silent_video):
    r = _upload(client, silent_video, "../../My Pattern Test.mp4")
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "uploaded" and job["filename"] == "My Pattern Test.mp4"
    assert job["media"]["width"] == 640 and job["expires_at"]
    jid = job["id"]

    bad = client.post(f"/api/jobs/{jid}/start", json={"clip_count": 99})
    assert bad.status_code == 422
    assert client.post(f"/api/jobs/{jid}/start", json=FAST).json()["status"] == "queued"
    assert client.post(f"/api/jobs/{jid}/start", json=FAST).status_code == 409

    assert worker.run_once() is True
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "completed", job["error"]
    assert job["progress"] == 1.0
    stages = {s["key"]: s["status"] for s in job["stages"]}
    assert stages["analyzing"] == "done" and stages["rendering"] == "done" and stages["ready"] == "done"
    assert stages["subtitles"] == "skipped", "a tone has no speech to caption"
    assert job["analysis"]["engine"] == "signals-only"
    assert any("No usable speech" in w for w in job["analysis"]["warnings"])
    assert 1 <= len(job["clips"]) <= 2

    clip = job["clips"][0]
    assert clip["status"] == "ready" and clip["qa"]["passed"] is True
    assert (clip["qa"]["width"], clip["qa"]["height"]) == (720, 1280)
    assert clip["has_captions_burned"] is False and clip["has_srt"] is False
    assert clip["layouts"] == ["blur"]
    assert 0 <= clip["score"] <= 100 and len(clip["dimensions"]) == 6
    assert "no speech" in clip["explanation"].lower()

    video = client.get(clip["urls"]["video"])
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    assert video.content[4:8] == b"ftyp"
    ranged = client.get(clip["urls"]["video"], headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206 and len(ranged.content) == 100
    dl = client.get(clip["urls"]["download"])
    assert 'filename="My-Pattern-Test-clip-01.mp4"' in dl.headers["content-disposition"]
    assert client.get(clip["urls"]["thumbnail"]).headers["content-type"] == "image/jpeg"
    assert client.get(clip["urls"]["srt"]).status_code == 404

    z = client.get(f"/api/jobs/{jid}/export.zip")
    assert z.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert (
        "My-Pattern-Test-clip-01.mp4" in names and "clips.json" in names and "publishing-notes.txt" in names
    )

    logs = client.get(f"/api/jobs/{jid}/logs").json()["logs"]
    assert any("rendered and verified" in entry["message"] for entry in logs)

    # ---- studio edit: change boundaries + framing, then re-render
    cid = clip["id"]
    assert client.patch(f"/api/jobs/{jid}/clips/{cid}", json={"start": 1.0, "end": 2.0}).status_code == 422
    edited = client.patch(
        f"/api/jobs/{jid}/clips/{cid}", json={"start": 1.0, "end": 7.0, "framing_mode": "center"}
    )
    assert edited.status_code == 200 and edited.json()["status"] == "queued"
    assert client.patch(f"/api/jobs/{jid}/clips/{cid}", json={"framing_mode": "blur"}).status_code == 409
    assert worker.run_once() is True
    after = client.get(f"/api/jobs/{jid}/clips/{cid}").json()
    assert after["status"] == "ready" and after["version"] == 2 and after["score_stale"] is True
    assert after["layouts"] == ["crop"] and abs(after["duration"] - 6.0) < 0.2
    assert after["urls"]["video"].endswith("?v=2")
    assert client.get(after["urls"]["video"]).status_code == 200
    assert not get_storage().clip_file(jid, cid, 1, "video").exists(), "old version is cleaned up"

    # ---- performance feedback
    m = client.post(
        f"/api/jobs/{jid}/clips/{cid}/metrics",
        json={"platform": "shorts", "views": 1200, "avg_percent_watched": 71.5},
    )
    assert m.status_code == 201
    listed = client.get(f"/api/jobs/{jid}/clips/{cid}/metrics").json()["metrics"]
    assert listed[0]["views"] == 1200 and listed[0]["platform"] == "shorts"

    # ---- delete removes rows and files
    assert client.delete(f"/api/jobs/{jid}").status_code == 204
    assert client.get(f"/api/jobs/{jid}").status_code == 404
    assert not (get_storage().root / jid).exists()


def test_vertical_and_audioless_sources(client, worker, vertical_video, no_audio_video):
    for path, layout in ((vertical_video, "fit"), (no_audio_video, "blur")):
        jid = _upload(client, path).json()["id"]
        client.post(f"/api/jobs/{jid}/start", json={**FAST, "clip_count": 1})
        worker.run_once()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed", job["error"]
        clip = job["clips"][0]
        assert clip["qa"]["passed"] and clip["layouts"] == [layout]
        if path is no_audio_video:
            assert any("no audio track" in w for w in job["analysis"]["warnings"])


def test_cancel_queued_job_and_failed_render_is_reported(client, worker, silent_video):
    jid = _upload(client, silent_video).json()["id"]
    client.post(f"/api/jobs/{jid}/start", json=FAST)
    assert client.post(f"/api/jobs/{jid}/cancel").json()["status"] == "canceled"
    assert worker.run_once() is False, "a canceled job is never picked up"

    # source file vanishes before processing -> clean, user-facing failure
    client.post(f"/api/jobs/{jid}/start", json=FAST)
    get_storage().source_path(jid, ".mp4").unlink()
    worker.run_once()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "failed" and "upload it again" in job["error"]


def test_cancel_during_processing(client, worker, silent_video, monkeypatch):
    from viralcut.pipeline import orchestrator

    jid = _upload(client, silent_video).json()["id"]
    client.post(f"/api/jobs/{jid}/start", json=FAST)
    real = orchestrator.analyze_visual

    def cancel_midway(*args, **kwargs):
        with session_scope() as s:
            s.get(Job, jid).cancel_requested = True
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "analyze_visual", cancel_midway)
    worker.run_once()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "canceled" and job["clips"] == []


def test_unexpected_error_is_retried_then_failed(client, worker, silent_video, monkeypatch):
    from viralcut.pipeline import orchestrator

    jid = _upload(client, silent_video).json()["id"]
    client.post(f"/api/jobs/{jid}/start", json=FAST)

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(orchestrator, "analyze_visual", boom)
    worker.run_once()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "queued" and job["attempts"] == 1
    worker.run_once()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "failed" and job["attempts"] == 2
    assert "disk on fire" not in job["error"], "internal details are not exposed"
    logs = client.get(f"/api/jobs/{jid}/logs").json()["logs"]
    assert any("Retrying" in entry["message"] for entry in logs)


def test_retention_purges_expired_jobs(client, silent_video):
    import datetime as dt

    from viralcut.db import utcnow
    from viralcut.jobs import purge_expired

    jid = _upload(client, silent_video).json()["id"]
    assert purge_expired() == 0
    with session_scope() as s:
        s.get(Job, jid).expires_at = utcnow() - dt.timedelta(minutes=1)
    assert purge_expired() == 1
    assert client.get(f"/api/jobs/{jid}").status_code == 404
    assert not (get_storage().root / jid).exists()
