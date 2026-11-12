"""A job left "processing" by a process that no longer exists is freed when a worker starts."""

import socket

from viralcut.db import Job, session_scope
from viralcut.jobs import recover_orphans


def _job(s, worker_id: str, cancel: bool = False) -> str:
    job = Job(original_filename="x.mp4", status="processing", worker_id=worker_id, cancel_requested=cancel)
    s.add(job)
    s.flush()
    return job.id


def test_orphans_on_this_host_are_recovered(client):
    host = socket.gethostname()
    with session_scope() as s:
        dead = _job(s, f"{host}:999999999:1")
        dead_canceled = _job(s, f"{host}:999999998:1", cancel=True)
        other_host = _job(s, "another-machine:999999999:1")
    assert recover_orphans() >= 2
    with session_scope() as s:
        assert s.get(Job, dead).status == "queued"
        assert s.get(Job, dead_canceled).status == "canceled"
        assert s.get(Job, other_host).status == "processing"
        # leave nothing claimable behind for the other tests' workers
        for jid in (dead, dead_canceled, other_host):
            s.delete(s.get(Job, jid))
