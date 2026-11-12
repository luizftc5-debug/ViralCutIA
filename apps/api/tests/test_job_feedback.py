"""What the job page is told while a video is processed and after it failed."""

from viralcut.db import Job, session_scope
from viralcut.jobs import JobReporter
from viralcut.pipeline.ffmpeg import FFmpegError
from viralcut.pipeline.orchestrator import failure_message, followup_ideas


def test_the_stage_says_what_is_happening(client):
    with session_scope() as s:
        job = Job(original_filename="x.mp4", status="processing")
        s.add(job)
        s.flush()
        job_id = job.id
    try:
        rep = JobReporter(job_id)
        rep.start("transcribing")
        rep.detail("Downloading the speech model (large-v3-turbo), first time only: 640 MB of 1.6 GB")

        def stage():
            return next(
                x for x in client.get(f"/api/jobs/{job_id}").json()["stages"] if x["key"] == "transcribing"
            )

        assert "640 MB of 1.6 GB" in stage()["detail"]
        assert stage()["started_at"].endswith("Z"), "the page runs a clock from the start of the stage"
        rep.progress(0.4, force=True)
        assert stage()["detail"] and stage()["progress"] == 0.4, "progress keeps the line"
        assert stage()["started_at"]
        rep.finish()
        assert stage()["detail"] == "" and stage()["status"] == "done"
    finally:
        with session_scope() as s:
            s.delete(s.get(Job, job_id))


def test_a_failure_says_where_it_stopped_and_what_kind_of_problem_it_was():
    msg = failure_message("reframing", FFmpegError("ffmpeg failed: Could not open encoder before EOF"))
    assert "“Smart reframing” (step 6 of 9)" in msg and "ffmpeg" in msg
    assert "Could not open encoder" not in msg, "the raw error belongs to the technical log"
    assert "out of memory" in failure_message(
        "transcribing", RuntimeError("CUDA failed with error out of memory")
    )
    assert "disk is full" in failure_message("rendering", OSError(28, "No space left on device"))
    assert failure_message("", ValueError("x")).startswith("Processing stopped. Something unexpected")


def test_follow_up_ideas_are_different_moments_not_one_moment_three_times():
    from types import SimpleNamespace as C

    def cand(start, end, score, text):
        return C(kind="speech", start=start, end=end, score=score, text=text)

    clip = cand(0, 20, 90, "The clip itself. More.")
    same = "Three years ago I lost everything. Then it changed."
    cands = [
        clip,
        cand(40, 60, 80, same),
        cand(41, 63, 79, same),  # the same moment with other in/out points
        cand(42, 58, 78, same),
        cand(10, 30, 77, "Overlaps the clip itself. No."),
        cand(70, 90, 60, "Stop setting goals. Build systems."),
        C(kind="signal", start=100, end=120, score=99, text=""),
    ]
    assert followup_ideas(clip, cands, [clip]) == ["Three years ago I lost everything", "Stop setting goals"]
