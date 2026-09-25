"""Idempotency-Key support for POST /jobs (service + HTTP levels).

Covers: first submission vs replay, no side effects on replay, 409 on
parameter mismatch, failed submissions not consuming the key, dependency
normalisation in the fingerprint, per-tenant key scoping, current-state
replays after lifecycle changes, in-process concurrency, cross-process
(separate locks) races on a shared SQLite file, persistence across a
container-style restart, and the header-less legacy interface.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

from app import db, scheduler
from app.config import get_settings

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Service level
# ---------------------------------------------------------------------------

def submit(body_key=None, **kw):
    """Submit a job; with body_key use the idempotent service entrypoint."""
    kw.setdefault("tenant_id", "t")
    kw.setdefault("gpus", 1)
    kw.setdefault("duration_min", 10)
    kw.setdefault("weight", 1.0)
    if body_key is None:
        return scheduler.submit_job(
            kw["tenant_id"], kw["gpus"], kw["duration_min"], kw["weight"],
            depends_on=kw.get("depends_on"),
        ), True
    return scheduler.submit_job_idempotent(
        kw["tenant_id"], kw["gpus"], kw["duration_min"], kw["weight"],
        depends_on=kw.get("depends_on") or [],
        idempotency_key=body_key,
    )


def test_first_call_creates_replay_returns_same_job(conn):
    job, created = submit(body_key="k1")
    assert created is True and job["status"] == "QUEUED"

    replay, created2 = submit(body_key="k1")
    assert created2 is False
    assert replay["id"] == job["id"]
    # Exactly one job exists for the key.
    assert scheduler.list_jobs(tenant_id="t") == [replay]


def test_replay_does_not_update_tenant_weight(conn):
    job, _ = submit(body_key="k1", weight=3.0)
    assert scheduler.get_tenant("t")["weight"] == 3.0

    # Change the tenant weight out of band.
    scheduler.upsert_tenant("t", 7.0)
    before = scheduler.get_tenant("t")
    assert before["weight"] == 7.0

    # Replaying the original submission (weight 3.0) must not reset it.
    replay, created = submit(body_key="k1", weight=3.0)
    assert created is False and replay["id"] == job["id"]
    after = scheduler.get_tenant("t")
    assert after["weight"] == 7.0
    assert after["aging_settled_at"] == before["aging_settled_at"]


def test_same_key_different_parameters_conflicts(conn):
    job, _ = submit(body_key="k1", gpus=1, duration_min=10)
    with pytest.raises(scheduler.Conflict) as exc:
        submit(body_key="k1", gpus=2, duration_min=10)
    assert exc.value.status_code == 409

    with pytest.raises(scheduler.Conflict):
        submit(body_key="k1", gpus=1, duration_min=11)
    with pytest.raises(scheduler.Conflict):
        submit(body_key="k1", gpus=1, duration_min=10, weight=2.5)

    # The conflicting attempts created nothing.
    jobs = scheduler.list_jobs(tenant_id="t")
    assert [j["id"] for j in jobs] == [job["id"]]


def test_conflicting_depends_on_detected(conn):
    prereq, _ = submit(tenant_id="base", body_key=None)
    job, _ = submit(body_key="k1", depends_on=[prereq["id"]])
    with pytest.raises(scheduler.Conflict):
        submit(body_key="k1", depends_on=[])
    assert scheduler.list_jobs(tenant_id="t")[0]["id"] == job["id"]


def test_failed_submission_does_not_consume_key(conn):
    # Capacity overflow on a first-time keyed submission (rejected before
    # any write; the key is not consumed).
    with pytest.raises(scheduler.SchedulerError):
        submit(body_key="k1", gpus=99, duration_min=10)

    # In-transaction failure: unknown prerequisite.
    with pytest.raises(scheduler.SchedulerError):
        submit(body_key="k1", depends_on=["J999999"])

    # In-transaction failure: dependency cycle.
    submit(tenant_id="a", gpus=1, duration_min=5)  # J000001
    with pytest.raises(scheduler.SchedulerError):
        submit(body_key="k1", depends_on=["J000002"])

    # The key is still free: a corrected request creates exactly one job.
    job, created = submit(body_key="k1", gpus=1, duration_min=10)
    assert created is True
    rows = db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM idempotent_jobs"
    ).fetchone()
    assert rows["n"] == 1
    assert [j["id"] for j in scheduler.list_jobs(tenant_id="t")] == [job["id"]]


def test_depends_on_fingerprint_is_deduped_and_sorted(conn):
    p1, _ = submit(tenant_id="base")                       # J000001
    p2_id = scheduler.submit_job("base", 1, 5)["id"]      # J000002
    # Duplicates and out-of-order input normalise to the sorted unique list.
    job, created = submit(
        body_key="k1", depends_on=[p2_id, p1["id"], p2_id]
    )
    assert created is True
    assert job["depends_on"] == [p1["id"], p2_id]

    row = db.get_conn().execute(
        "SELECT * FROM idempotent_jobs WHERE idempotency_key = 'k1'"
    ).fetchone()
    assert json.loads(row["depends_on"]) == [p1["id"], p2_id]

    # The same submission expressed with the canonical list is a replay.
    replay, created2 = submit(body_key="k1", depends_on=[p1["id"], p2_id])
    assert created2 is False and replay["id"] == job["id"]


def test_key_scoped_per_tenant(conn):
    job_a, _ = submit(tenant_id="team-a", body_key="shared")
    job_b, _ = submit(tenant_id="team-b", body_key="shared")
    assert job_a["id"] != job_b["id"]

    # Each tenant replays independently.
    replay_a, created = submit(tenant_id="team-a", body_key="shared")
    assert created is False and replay_a["id"] == job_a["id"]
    replay_b, created = submit(tenant_id="team-b", body_key="shared")
    assert created is False and replay_b["id"] == job_b["id"]


def test_replay_returns_current_job_state(conn):
    job, _ = submit(body_key="k1")
    scheduler.run_batch()  # QUEUED -> RUNNING
    replay, created = submit(body_key="k1")
    assert created is False
    assert replay["status"] == "RUNNING"
    assert replay["scheduled_at"] is not None

    scheduler.complete_job(job["id"])
    replay, _ = submit(body_key="k1")
    assert replay["status"] == "COMPLETED"


def _lower_cluster_capacity(monkeypatch, gpus: int) -> None:
    """Simulate a restart with a smaller CLUSTER_GPUS."""
    monkeypatch.setenv("CLUSTER_GPUS", str(gpus))
    get_settings.cache_clear()


def test_replay_after_capacity_reduction_returns_current_job(conn, monkeypatch):
    # Original request exactly fills the capacity-8 cluster.
    job, created = submit(body_key="k1", gpus=8, duration_min=10)
    assert created is True
    scheduler.run_batch()  # QUEUED -> RUNNING at capacity 8

    # Restart with CLUSTER_GPUS lowered to 4: the original request now
    # exceeds the current capacity, but the stored fingerprint decides first.
    _lower_cluster_capacity(monkeypatch, 4)
    tenant_before = db.get_conn().execute(
        "SELECT weight, updated_at FROM tenants WHERE id = 't'"
    ).fetchone()
    ledger_before = db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM ledger"
    ).fetchone()["n"]

    replay, created = submit(body_key="k1", gpus=8, duration_min=10)
    assert created is False
    assert replay["id"] == job["id"]
    assert replay["status"] == "RUNNING"  # the job's CURRENT state

    # The replay rewrote nothing: tenant row, ledger and job count unchanged.
    tenant_after = db.get_conn().execute(
        "SELECT weight, updated_at FROM tenants WHERE id = 't'"
    ).fetchone()
    assert dict(tenant_after) == dict(tenant_before)
    assert db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM ledger"
    ).fetchone()["n"] == ledger_before
    assert len(scheduler.list_jobs(tenant_id="t")) == 1


def test_conflict_after_capacity_reduction_even_over_capacity(conn, monkeypatch):
    job, _ = submit(body_key="k1", gpus=2, duration_min=10)
    _lower_cluster_capacity(monkeypatch, 4)

    # Different parameters that THEMSELVES exceed the current capacity:
    # the fingerprint mismatch (409) is judged before capacity (400).
    with pytest.raises(scheduler.Conflict) as exc:
        submit(body_key="k1", gpus=6, duration_min=10)
    assert exc.value.status_code == 409

    # Different parameters within the current capacity conflict as well.
    with pytest.raises(scheduler.Conflict):
        submit(body_key="k1", gpus=3, duration_min=10)

    assert [j["id"] for j in scheduler.list_jobs(tenant_id="t")] == [job["id"]]


def test_first_time_keyed_submission_uses_current_capacity(conn, monkeypatch):
    _lower_cluster_capacity(monkeypatch, 4)
    # Key miss: the first-time submission is validated against the CURRENT
    # capacity and fails without consuming the key.
    with pytest.raises(scheduler.SchedulerError) as exc:
        submit(body_key="k1", gpus=5, duration_min=10)
    assert exc.value.status_code == 400
    assert db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM idempotent_jobs"
    ).fetchone()["n"] == 0

    # A corrected retry with the same key still creates the job.
    job, created = submit(body_key="k1", gpus=4, duration_min=10)
    assert created is True
    # ...and a request within the current capacity but WITHOUT a key is
    # independent of it; an over-capacity key-less request is still rejected.
    with pytest.raises(scheduler.SchedulerError) as exc:
        submit(gpus=5, duration_min=10)
    assert exc.value.status_code == 400
    assert [j["id"] for j in scheduler.list_jobs(tenant_id="t")] == [job["id"]]


def test_concurrent_first_submissions_create_one_job(conn):
    import threading

    results: list[tuple[str, bool]] = []
    errors: list[Exception] = []

    def fire():
        try:
            job, created = submit(body_key="race")
            results.append((job["id"], created))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=fire) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 8
    ids = {jid for jid, _ in results}
    assert ids == {results[0][0]}
    assert sum(1 for _, created in results if created) == 1
    assert len(scheduler.list_jobs(tenant_id="t")) == 1


def test_replay_survives_restart_with_file_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "restart.db")
    monkeypatch.setenv("DB_PATH", db_path)
    get_settings.cache_clear()

    db.reset_for_tests()
    db.connect(db_path)
    job, created = scheduler.submit_job_idempotent(
        "t", 2, 15, 1.0, [], idempotency_key="persist"
    )
    assert created is True
    db.get_conn().close()
    db.reset_for_tests()

    # Simulate a container restart: fresh module-level connection to the same
    # SQLite file. The key record survived, so the retry is a replay.
    db.connect(db_path)
    try:
        replay, created2 = scheduler.submit_job_idempotent(
            "t", 2, 15, 1.0, [], idempotency_key="persist"
        )
        assert created2 is False
        assert replay["id"] == job["id"]
        assert len(scheduler.list_jobs(tenant_id="t")) == 1
    finally:
        db.get_conn().close()
        db.reset_for_tests()
        get_settings.cache_clear()


def test_replay_survives_restart_with_lowered_capacity(tmp_path, monkeypatch):
    db_path = str(tmp_path / "restart-lowered.db")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("CLUSTER_GPUS", "8")
    get_settings.cache_clear()

    db.reset_for_tests()
    db.connect(db_path)
    job, created = scheduler.submit_job_idempotent(
        "t", 8, 15, 1.0, [], idempotency_key="persist"
    )
    assert created is True
    db.get_conn().close()
    db.reset_for_tests()

    # Restart with CLUSTER_GPUS lowered to 4: the original 8-GPU request now
    # exceeds the current capacity. The stored fingerprint still decides
    # first: identical parameters replay the job, different ones conflict.
    monkeypatch.setenv("CLUSTER_GPUS", "4")
    get_settings.cache_clear()
    db.connect(db_path)
    try:
        replay, created2 = scheduler.submit_job_idempotent(
            "t", 8, 15, 1.0, [], idempotency_key="persist"
        )
        assert created2 is False
        assert replay["id"] == job["id"]
        with pytest.raises(scheduler.Conflict):
            scheduler.submit_job_idempotent(
                "t", 6, 15, 1.0, [], idempotency_key="persist"
            )
        assert len(scheduler.list_jobs(tenant_id="t")) == 1
    finally:
        db.get_conn().close()
        db.reset_for_tests()
        get_settings.cache_clear()


def _multiprocess_worker(db_path: str, key: str, out: str) -> None:
    """Fresh-process submitter: exercises the UNIQUE-key/IntegrityError path
    because each process has its own module-level lock and connection."""
    sys.path.insert(0, str(ROOT))
    os.environ["DB_PATH"] = db_path
    os.environ["CLUSTER_GPUS"] = "8"
    from app import db as db_mod
    from app import scheduler as sched
    from app.config import get_settings as _get_settings

    _get_settings.cache_clear()
    db_mod.connect(db_path)
    try:
        job, created = sched.submit_job_idempotent(
            "mp", 1, 10, 1.0, [], idempotency_key=key
        )
        Path(out).write_text(json.dumps({"id": job["id"], "created": created}))
    except Exception as exc:  # pragma: no cover - failure path
        Path(out).write_text(json.dumps({"error": repr(exc)}))


def test_concurrent_processes_create_one_job(tmp_path, monkeypatch):
    db_path = str(tmp_path / "mp.db")
    monkeypatch.setenv("DB_PATH", db_path)
    get_settings.cache_clear()
    # Create the schema on disk first; workers then open their own handles.
    db.reset_for_tests()
    db.connect(db_path)
    db.get_conn().close()
    db.reset_for_tests()

    ctx = multiprocessing.get_context("spawn")
    procs = []
    out_files = []
    for i in range(4):
        out = str(tmp_path / f"out{i}.json")
        out_files.append(out)
        p = ctx.Process(
            target=_multiprocess_worker, args=(db_path, "mp-key", out)
        )
        procs.append(p)
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    payloads = [json.loads(Path(f).read_text()) for f in out_files]
    assert all("error" not in p for p in payloads), payloads
    ids = {p["id"] for p in payloads}
    assert len(ids) == 1
    assert sum(1 for p in payloads if p["created"]) == 1

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

HEADERS = {"Idempotency-Key": "abc-123"}


def test_http_first_201_replay_200_same_body(client):
    r = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 30, "weight": 1},
        headers=HEADERS)
    assert r.status_code == 201, r.text
    first = r.json()

    r2 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 2, "duration_min": 30, "weight": 1},
        headers=HEADERS)
    assert r2.status_code == 200, r2.text
    assert r2.json()["id"] == first["id"]
    assert r2.json()["queued_at"] == first["queued_at"]

    # Only one job exists.
    jobs = client.get("/jobs").json()
    assert [j["id"] for j in jobs] == [first["id"]]


def test_http_same_key_different_body_409(client):
    body = {"tenant_id": "team-a", "gpus": 2, "duration_min": 30}
    assert client.post("/jobs", json=body, headers=HEADERS).status_code == 201
    r = client.post(
        "/jobs", json={**body, "duration_min": 45}, headers=HEADERS)
    assert r.status_code == 409
    assert "Idempotency-Key" in r.json()["error"]
    # No second job created.
    assert len(client.get("/jobs").json()) == 1


def test_http_failed_validation_leaves_key_free(client):
    # Unknown prerequisite -> 400, key not consumed.
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5,
        "depends_on": ["J000099"]}, headers=HEADERS)
    assert r.status_code == 400
    # The same key now succeeds.
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}, headers=HEADERS)
    assert r.status_code == 201, r.text
    # Immediate replay is still a 200 replay of that job.
    r = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}, headers=HEADERS)
    assert r.status_code == 200


def test_http_blank_header_uses_legacy_interface(client):
    body = {"tenant_id": "team-a", "gpus": 1, "duration_min": 5}
    r1 = client.post("/jobs", json=body, headers={"Idempotency-Key": ""})
    r2 = client.post("/jobs", json=body, headers={"Idempotency-Key": "   "})
    r3 = client.post("/jobs", json=body)
    assert r1.status_code == r2.status_code == r3.status_code == 201
    assert len({r1.json()["id"], r2.json()["id"], r3.json()["id"]}) == 3


def test_http_key_scoped_by_tenant(client):
    r1 = client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 1, "duration_min": 5},
        headers=HEADERS)
    r2 = client.post("/jobs", json={
        "tenant_id": "team-b", "gpus": 1, "duration_min": 5},
        headers=HEADERS)
    assert r1.status_code == r2.status_code == 201
    assert r1.json()["id"] != r2.json()["id"]


def test_http_fingerprint_decides_before_capacity(client, monkeypatch):
    # First submission fills the capacity-8 cluster exactly.
    body = {"tenant_id": "team-a", "gpus": 8, "duration_min": 30, "weight": 1}
    r = client.post("/jobs", json=body, headers=HEADERS)
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]

    # Restart with CLUSTER_GPUS lowered to 4: the original request now
    # exceeds the current capacity.
    monkeypatch.setenv("CLUSTER_GPUS", "4")
    get_settings.cache_clear()

    # Identical retry -> 200 with the job's current object, not a 400.
    r = client.post("/jobs", json=body, headers=HEADERS)
    assert r.status_code == 200, r.text
    assert r.json()["id"] == job_id

    # Same key with different parameters -> 409, even though the new
    # parameters themselves exceed the current capacity.
    r = client.post("/jobs", json={**body, "gpus": 6}, headers=HEADERS)
    assert r.status_code == 409
    assert "Idempotency-Key" in r.json()["error"]

    # A NEW key (first-time submission) is validated against the current
    # capacity -> 400, and the failure does not consume that key.
    other = {"Idempotency-Key": "other-key"}
    r = client.post("/jobs", json={**body, "gpus": 6}, headers=other)
    assert r.status_code == 400
    r = client.post("/jobs", json={**body, "gpus": 4}, headers=other)
    assert r.status_code == 201, r.text

    # Key-less submissions keep the original rule: over current capacity
    # -> 400; every key-less request creates a new job.
    r = client.post("/jobs", json={**body, "gpus": 6})
    assert r.status_code == 400
    r = client.post("/jobs", json={**body, "gpus": 4})
    assert r.status_code == 201

    # Exactly three jobs exist: the replay/conflict created nothing.
    assert len(client.get("/jobs").json()) == 3
