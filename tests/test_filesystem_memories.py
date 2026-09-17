from __future__ import annotations

import importlib.util
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from tmux_pilot.fs_store import RecordStore

import project_memory as pm


IMPORTER_PATH = Path(__file__).parents[1] / "scripts" / "import_sqlite_memories.py"


def _state(project_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        _agent_zoo_context={
            "project_name": "demo",
            "project_dir": str(project_dir),
            "session_id": "session-1",
        }
    )


def _wait_for(store: pm.ProjectMemoryStore, memory_id: str, *statuses: str) -> dict:
    wanted = set(statuses)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        receipt = store.receipt(memory_id)
        if receipt is not None and (not wanted or receipt.get("status") in wanted):
            return receipt
        time.sleep(0.01)
    return store.receipt(memory_id) or {}


def _load_importer():
    spec = importlib.util.spec_from_file_location("import_sqlite_memories", IMPORTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filesystem_crud_publishes_revisioned_tombstone(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    project_dir = tmp_path / "projects" / "demo"
    state = _state(project_dir)
    store = pm.ProjectMemoryStore()
    try:
        memory = store.create(state, "Use the filesystem store.", r"filesystem")
        assert memory["id"].startswith("mem_")
        assert store.list_all(state) == [memory]
        assert _wait_for(store, memory["id"], "shared")["shared_saved"] is True

        shared_root = project_dir / "plugin-data" / "project-memory"
        shared = RecordStore(shared_root, create=False)
        published = shared.get("project_memory", memory["id"])
        assert published["revision"] == 1
        assert published["deleted"] is False

        updated = store.update(state, memory["id"], content="Use RecordStore.")
        assert updated["content"] == "Use RecordStore."
        assert _wait_for(store, memory["id"], "shared")["revision"] == 2
        assert shared.get("project_memory", memory["id"])["revision"] == 2

        assert store.delete(state, memory["id"]) is True
        assert store.list_all(state) == []
        assert _wait_for(store, memory["id"], "shared")["revision"] == 3
        tombstone = shared.get("project_memory", memory["id"])
        assert tombstone["deleted"] is True
        assert tombstone["revision"] == 3
    finally:
        store.close()


def test_last_good_snapshot_survives_shared_outage_and_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    project_dir = tmp_path / "projects" / "demo"
    state = _state(project_dir)
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "Persist the last-good cache.", r"last[- ]good")
    assert _wait_for(store, memory["id"], "shared")["status"] == "shared"

    local_repositories = list((tmp_path / "local" / "project-memory").glob("*") )
    local_root = next(path for path in local_repositories if path.is_dir())
    local_store = RecordStore(local_root, create=False)
    snapshot = local_store.get(pm.ProjectMemoryStore._SNAPSHOT_NAMESPACE, memory["id"])
    assert snapshot["id"] == memory["id"]
    assert local_store.get(
        pm.ProjectMemoryStore._SNAPSHOT_NAMESPACE,
        pm.ProjectMemoryStore._SNAPSHOT_META_KEY,
    )["kind"] == "last_good_snapshot"

    # list_all must not synchronously consult the shared root after the cache is ready.
    original_refresh = store._refresh_shared
    store._refresh_shared = lambda _generation: (_ for _ in ()).throw(AssertionError("shared read on cache path"))
    try:
        assert store.list_all(state) == [memory]
    finally:
        store._refresh_shared = original_refresh
        store.close()

    shared_root = project_dir / "plugin-data" / "project-memory"
    moved_root = tmp_path / "shared-away"
    shared_root.rename(moved_root)

    restarted = pm.ProjectMemoryStore()
    try:
        assert restarted.list_all(state) == [memory]
        status = restarted.snapshot_status()
        assert status["ready"] is True
        assert status["last_good"] is True
        assert status["status"] in {"cached", "unavailable"}
        # Give the daemon a chance to observe the unavailable shared root; the
        # cached result must remain available and must not become an empty result.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and restarted.snapshot_status()["status"] != "unavailable":
            time.sleep(0.01)
        assert restarted.list_all(state) == [memory]
        assert restarted.snapshot_status()["last_good"] is True
    finally:
        restarted.close()


def test_revision_conflict_is_visible_and_normal_update_resolves_it(tmp_path, monkeypatch):
    project_dir = tmp_path / "projects" / "demo"
    local_one = tmp_path / "local-one"
    local_two = tmp_path / "local-two"
    state = _state(project_dir)
    first = pm.ProjectMemoryStore()
    second = pm.ProjectMemoryStore()
    try:
        monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(local_one))
        memory = first.create(state, "Original", r"original")
        assert _wait_for(first, memory["id"], "shared")["status"] == "shared"

        monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(local_two))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and second.list_all(state) != [memory]:
            time.sleep(0.01)
        assert second.list_all(state) == [memory]

        monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(local_one))
        first.update(state, memory["id"], content="First writer")
        assert _wait_for(first, memory["id"], "shared")["status"] == "shared"

        monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(local_two))
        second.update(state, memory["id"], content="Stale writer")
        conflict = _wait_for(second, memory["id"], "conflict")
        assert conflict["shared_saved"] is False
        assert "conflict" in second.receipt_text(memory["id"])
        assert second.snapshot_status()["conflicts"] == [memory["id"]]
        assert second.list_all(state)[0]["content"] == "Stale writer"

        # A normal update explicitly resolves the conflict against the observed
        # shared revision; no metadata repair or hidden overwrite is needed.
        second.update(state, memory["id"], content="Resolved writer")
        resolved = _wait_for(second, memory["id"], "shared")
        assert resolved["shared_saved"] is True
        assert second.snapshot_status()["conflicts"] == []
        shared = RecordStore(project_dir / "plugin-data" / "project-memory", create=False)
        assert shared.get("project_memory", memory["id"])["content"] == "Resolved writer"
    finally:
        first.close()
        second.close()


def test_shared_unavailable_is_not_reported_as_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    project_dir = tmp_path / "projects" / "demo"
    shared_root = project_dir / "plugin-data" / "project-memory"
    shared_root.parent.mkdir(parents=True)
    shared_root.write_text("not a record store", encoding="utf-8")
    store = pm.ProjectMemoryStore()
    try:
        state = _state(project_dir)
        assert store.list_all(state) == []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and store.snapshot_status()["status"] == "not_ready":
            time.sleep(0.01)
        status = store.snapshot_status()
        assert status["status"] == "unavailable"
        assert status["shared_available"] is False
        assert status["last_good"] is False
    finally:
        store.close()


def test_sqlite_import_is_read_only_idempotent_and_conflict_preserving(tmp_path):
    importer = _load_importer()
    source = tmp_path / "legacy.sqlite"
    destination = tmp_path / "filesystem"
    connection = sqlite3.connect(source)
    connection.execute(
        "CREATE TABLE azo_project_memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, trigger TEXT NOT NULL, "
        "created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    connection.execute(
        "INSERT INTO azo_project_memories VALUES (?, ?, ?, ?, ?)",
        ("mem_fixed", " Preserve spaces ", r"\\bspaces\\b", 100.0, 200.0),
    )
    connection.commit()
    connection.close()
    source_bytes = source.read_bytes()

    first = importer.import_sqlite_memories(source, destination)
    assert first["imported"] == 1
    assert source.read_bytes() == source_bytes
    before_dry_run = {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    dry_run = importer.import_sqlite_memories(source, destination, dry_run=True)
    assert dry_run["existing"] == 1
    after_dry_run = {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert after_dry_run == before_dry_run
    second = importer.import_sqlite_memories(source, destination)
    assert second["existing"] == 1

    store = RecordStore(destination, create=False)
    imported = store.get("project_memory", "mem_fixed")
    assert imported["id"] == "mem_fixed"
    assert imported["content"] == " Preserve spaces "
    assert imported["trigger"] == r"\\bspaces\\b"
    assert imported["created_at"] == 100.0
    assert imported["updated_at"] == 200.0

    writable = RecordStore(destination)
    writable.update(
        "project_memory",
        "mem_fixed",
        lambda current: {
            **current,
            "content": "Different authoritative content",
            "revision": 9,
            "operation_id": "external",
        },
    )
    conflict = importer.import_sqlite_memories(source, destination)
    assert conflict["conflicts"] == ["mem_fixed"]
    assert writable.get("project_memory", "mem_fixed")["content"] == "Different authoritative content"


def test_snapshot_merge_preserves_higher_revision_and_known_tombstone(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    project_dir = tmp_path / "projects" / "demo"
    state = _state(project_dir)
    store = pm.ProjectMemoryStore()
    try:
        store.list_all(state)
        local_store = store._local_store
        assert local_store is not None
        high = {
            "schema_version": 1,
            "id": "mem_high",
            "content": "new",
            "trigger": "new",
            "created_at": 1.0,
            "updated_at": 3.0,
            "revision": 3,
            "deleted": False,
            "operation_id": "op-high",
        }
        tombstone = {
            "schema_version": 1,
            "id": "mem_deleted",
            "content": "old",
            "trigger": "old",
            "created_at": 1.0,
            "updated_at": 3.0,
            "revision": 3,
            "deleted": True,
            "operation_id": "op-delete",
        }
        local_store.put(pm.ProjectMemoryStore._SNAPSHOT_NAMESPACE, "mem_high", high)
        local_store.put(pm.ProjectMemoryStore._SNAPSHOT_NAMESPACE, "mem_deleted", tombstone)
        lower = {**high, "content": "stale", "revision": 2, "operation_id": "op-stale"}
        store._persist_snapshot({"mem_high": lower})
        assert local_store.get(pm.ProjectMemoryStore._SNAPSHOT_NAMESPACE, "mem_high")["content"] == "new"
        assert local_store.get(pm.ProjectMemoryStore._SNAPSHOT_NAMESPACE, "mem_deleted")["deleted"] is True
    finally:
        store.close()


def test_conflicted_local_tombstone_can_be_resolved_by_normal_delete(tmp_path, monkeypatch):
    project_dir = tmp_path / "projects" / "demo"
    local_one = tmp_path / "local-one"
    local_two = tmp_path / "local-two"
    state = _state(project_dir)
    first = pm.ProjectMemoryStore()
    second = pm.ProjectMemoryStore()
    try:
        monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(local_one))
        memory = first.create(state, "Original", r"original")
        assert _wait_for(first, memory["id"], "shared")["status"] == "shared"
        monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(local_two))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and second.list_all(state) != [memory]:
            time.sleep(0.01)
        assert second.list_all(state) == [memory]

        monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(local_one))
        first.update(state, memory["id"], content="First writer")
        assert _wait_for(first, memory["id"], "shared")["status"] == "shared"
        monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(local_two))
        assert second.delete(state, memory["id"]) is True
        assert _wait_for(second, memory["id"], "conflict")["status"] == "conflict"
        assert second.list_all(state) == []

        # The local tombstone is itself a conflicted overlay.  A normal delete
        # must resolve against the observed shared live record, not return False.
        assert second.delete(state, memory["id"]) is True
        assert _wait_for(second, memory["id"], "shared")["status"] == "shared"
        shared = RecordStore(project_dir / "plugin-data" / "project-memory", create=False)
        assert shared.get("project_memory", memory["id"])["deleted"] is True
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("failure", ["snapshot", "overlay", "pending"])
@pytest.mark.parametrize("deleted", [False, True])
@pytest.mark.parametrize("successor", [False, True])
def test_publication_cleanup_recovers_after_restart_and_remote_edit(tmp_path, monkeypatch, failure, deleted, successor):
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    # Drive the real filesystem operations deliberately, without worker races.
    monkeypatch.setattr(pm.ProjectMemoryStore, "_start_worker_locked", lambda *args: None)
    state = _state(tmp_path / "shared")
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "Original", "original")
    memory_id = memory["id"]
    operation_id = store.receipt(memory_id)["operation_id"]
    local = store._local_store
    method, namespace = {
        "snapshot": ("update", store._SNAPSHOT_NAMESPACE),
        "overlay": ("update", store._OVERLAY_NAMESPACE),
        "pending": ("delete", store._PENDING_NAMESPACE),
    }[failure]
    original = getattr(local, method)

    def fail_cleanup(*args, **kwargs):
        if args[0] == namespace:
            raise OSError("injected cleanup failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(local, method, fail_cleanup)
        store._publish_operation(operation_id, store._generation)
    assert store.receipt(memory_id)["shared_saved"]
    assert local.get(store._PENDING_NAMESPACE, operation_id)["status"] == "published"
    assert operation_id in store._pending
    if failure == "pending":
        assert local.get(store._OVERLAY_NAMESPACE, memory_id) is None
        assert memory_id not in store._overlay
    remaining = set()
    if successor:
        store.update(state, memory_id, content="Newer local memory")
        remaining.add(store.receipt(memory_id)["operation_id"])
    expected_overlay = local.get(store._OVERLAY_NAMESPACE, memory_id) if successor else None
    store.close()

    shared = RecordStore(state._agent_zoo_context["project_dir"] + "/plugin-data/project-memory", create=True)
    newer = {**shared.get(store._RECORD_NAMESPACE, memory_id), "revision": 2,
             "operation_id": "remote-edit", "content": "Newer remote memory", "deleted": deleted}
    shared.put(store._RECORD_NAMESPACE, memory_id, newer)
    restarted = pm.ProjectMemoryStore()
    try:
        restarted.list_all(state)
        assert restarted._receipts[operation_id]["shared_saved"]
        assert restarted._next_operation_locked() == operation_id
        restarted._refresh_shared(restarted._generation)
        restarted._publish_operation(operation_id, restarted._generation)
        assert set(restarted._pending) == remaining
        local = restarted._local_store
        assert {key for key, _ in local.items(store._PENDING_NAMESPACE)} == remaining
        assert local.get(store._OVERLAY_NAMESPACE, memory_id) == expected_overlay
        assert restarted._overlay.get(memory_id) == (expected_overlay["record"] if successor else None)
        assert restarted._shared_snapshot[memory_id] == newer
        assert local.get(store._SNAPSHOT_NAMESPACE, memory_id) == newer
        assert shared.get(store._RECORD_NAMESPACE, memory_id) == newer
        recalled = restarted.get(state, memory_id)
        if successor:
            assert recalled["content"] == "Newer local memory"
        else:
            assert recalled is None if deleted else recalled["content"] == newer["content"]
        assert restarted.snapshot_status()["conflicts"] == []
    finally:
        restarted.close()
