#!/usr/bin/env python3
"""Opt-in, provider-free canonical history/memory workflow; never run on live state.

Requires an existing compatible AZ/AU/TP environment and this checkout's src on
PYTHONPATH. --root must not exist. Retains a tiny private home/spool and prints
JSON receipts. Calls real plugin components, persistence and filesystem APIs;
not a full backend/TUI or multi-host acceptance run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def emit(phase, **values):
    print(json.dumps({"phase": phase, **values}, default=str), flush=True)


def wait_for(read, ready, detail, timeout=10.0):
    deadline = time.monotonic() + timeout
    while True:
        value = read()
        if ready(value):
            return value
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{detail}: {value!r}")
        time.sleep(0.025)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="New private directory on a local filesystem; retained")
    args = parser.parse_args()
    root = args.root.expanduser().absolute()
    root.mkdir(mode=0o700, exist_ok=False)
    home, local = root / "home", root / "local"
    for key in ("AGENT_ZOO_HOME", "AGENT_ZOO_STATE_ROOT"):
        os.environ[key] = str(home)
    os.environ["AGENT_ZOO_LOCAL_STATE_ROOT"] = str(local)
    os.environ["AZO_PROJECT_MEMORY_CONFIG"] = str(root / "no-installed-config.yaml")

    # Import after isolation. No managed plugin, provider or launcher is loaded.
    import project_memory as pm
    from agent_utils import Entry, Session
    from agent_utils.builder import PipelineBuilder, core, buffer_editing
    from agent_utils.checkpoint_revisions import publish_revision
    from agent_utils.files.buffer_manager import BufferManager
    from agent_utils.session import flatten
    from agent_zoo import projects
    from agent_zoo.runtime_support import SESSION_PERSISTENCE_EXCLUDED_ATTRS
    from agent_zoo.session_load import PendingSessionLoad, load_session
    from agent_zoo.session_store import SessionPersistence

    emit("sources", root=root, plugin=pm.__file__,
         plugin_sha256=hashlib.sha256(Path(pm.__file__).read_bytes()).hexdigest())
    project = "journal-history-probe"
    projects.configure_state_root(home)
    projects.ensure_project(project)
    project_dir = home / "projects" / project
    stores, writers = [], []

    def turn(session, marker):
        for role, text in (("user", marker), ("assistant", "answer " + marker)):
            session.state.entries.append(Entry(
                messages=[{"role": role, "content": text}],
                index=len(session.state.entries), step=session.state.step))

    def plugin_session(regex=True, history=True, enabled=True, document=None):
        session = Session(system_prompt="Private diagnostic; no provider calls.")
        session.state.buffer_manager = BufferManager()
        builder = PipelineBuilder(session).add(core()).add(buffer_editing())
        pm.register_features(builder, session=session, config={"project_memory": {
            "enabled": enabled, "regex_memories": {"enabled": regex},
            "project_sessions": {"enabled": history}, "include_bundled_memories": False}})
        session.pipeline = builder.build()
        if document is not None:
            session.load_document(document)
        session._init_state()
        session.state._session_id = getattr(session.state, "_session_id", str(uuid.uuid4()))
        session.state._agent_zoo_context = {
            "project_name": project, "project_dir": str(project_dir),
            "local_state_root": str(local)}
        components = {type(item): item for item in flatten(session.pipeline)}
        if pm.ProjectMemoryStore in components:
            stores.append(components[pm.ProjectMemoryStore])
        return session, components

    def attach(session, components):
        for cls in (pm.ProjectMemoryStore, pm.ProjectMemoryBuffer, pm.ProjectSessionBuffers):
            if cls in components:
                components[cls](session.state)

    def read_buffer(session, name):
        # Exact buffers cache per turn; advance the observer step for retry.
        session.state.step += 1
        return session.state.buffer_manager.resolve_for_read(session.state, name).text

    def projection(session, sid, marker, commit):
        index = wait_for(lambda: read_buffer(session, "project_sessions"),
                         lambda text: sid in text, "catalog discovery failed")
        buffer_id = pm._session_buffer_ids([{"session_id": sid}])[sid]
        require(buffer_id in index, "catalog did not expose the existing ses_ ID")
        text = wait_for(lambda: read_buffer(session, buffer_id),
                        lambda value: f"Source Commit: {commit}" in value,
                        "exact commit projection failed")
        require(marker in text, "newest distinctive turn missing")
        require("Source Format: journal" in text, "selected historical snapshot, not journal")
        require("Source Instance:" in text and "Session source:" in text, "missing source identity")
        emit("projection", session_id=sid, buffer_id=buffer_id, commit_id=commit,
             text=text)
        return buffer_id, text

    def save(session, persistence):
        receipt = persistence.save(session, exclude_attrs=SESSION_PERSISTENCE_EXCLUDED_ATTRS)
        require(receipt.local_saved, f"local save failed: {receipt}")
        shared = persistence.wait_for_shared(receipt.revision_id, timeout_s=10)
        require(shared is not None and shared.shared_saved, f"shared save failed: {shared}")
        emit("save", **shared.to_dict())
        return receipt.revision_id

    def close_store(store):
        store.close(timeout_s=2)
        require(not store.worker_alive, "private memory worker did not stop")

    def published(store):
        wait_for(store.snapshot_status,
                 lambda value: value["ready"] and not value["pending_operations"]
                 and not value["conflicts"], "memory publication failed")

    try:
        for regex, history, enabled in ((True, True, True), (True, False, True),
                                        (False, True, True), (False, False, True),
                                        (True, True, False)):
            session, components = plugin_session(regex, history, enabled)
            require((pm.RecordMemory in components) == (regex and enabled), "regex control coupled")
            require((pm.ProjectSessionBuffers in components) == (history and enabled), "history control coupled")
            attach(session, components)
            manager = session.state.buffer_manager
            require(manager.is_special("project_memory") == (regex and enabled), "memory buffer control coupled")
            require(manager.is_special("project_sessions") == (history and enabled), "history buffer control coupled")
            require(manager.is_special("ses_12345678") == (history and enabled), "history namespace control coupled")
            emit("controls", regex=regex, history=history, enabled=enabled)
            if pm.ProjectMemoryStore in components:
                close_store(components[pm.ProjectMemoryStore])

        viewer, components = plugin_session()
        attach(viewer, components)
        for mixed in (False, True):
            sid = str(uuid.uuid4())
            source = Session(system_prompt="Private journal history source.")
            source.state._session_id = sid
            locator = project_dir / "sessions" / sid / "session.json"
            parent = None
            if mixed:
                turn(source, "legacy-only-turn")
                parent = publish_revision(source, locator.parent, session_id=sid,
                                          execution_id="historical-probe").revision.revision_id
            instance = uuid.uuid4().hex
            persistence = SessionPersistence(locator, session_id=sid, instance_id=instance,
                                             parent_commit_id=parent, local_state_root=local)
            writers.append(persistence)
            marker = ("mixed" if mixed else "journal-only") + "-newest-" + sid
            turn(source, marker)
            first = save(source, persistence)
            projects.register_session(project, sid, title=marker)
            # Cold local root forces a real shared mixed-repository read.
            observer, observed_components = plugin_session(regex=False)
            observer.state._agent_zoo_context["local_state_root"] = str(root / "cold-local")
            attach(observer, observed_components)
            buffer_id, pinned = projection(observer, sid, marker, first)
            turn(source, "after-pin-" + sid)
            second = save(source, persistence)
            require(first != second, "new turn did not produce a new commit")
            require(read_buffer(observer, buffer_id) == pinned, "existing buffer silently retargeted")
            restarted, restarted_components = plugin_session(regex=False)
            attach(restarted, restarted_components)
            projection(restarted, sid, "after-pin-" + sid, second)
            loaded = load_session(locator, session_id=sid, commit_id=first,
                                  local_state_root=local, timeout_s=0)
            if isinstance(loaded, PendingSessionLoad):
                loaded = loaded.wait(timeout_s=10)
            require(not isinstance(loaded, PendingSessionLoad), "exact reload stayed pending")
            require(loaded.state.commit_id == first, "exact reload selected another commit")
            restored = Session()
            restored.load_document(loaded.state.document)
            restored._init_state()
            content = str([entry.messages for entry in restored.state.entries])
            require(marker in content and "after-pin-" + sid not in content, "exact restore changed history")
            require(not locator.exists(), "journal save created a loose snapshot")
            emit("history_restore", mixed=mixed, first=first, second=second, stable_pin=True)

        store = components[pm.ProjectMemoryStore]
        result = components[pm.RecordMemory].execute(viewer.state,
                    content="private durable memory", trigger="probe-recall-token")
        memory_id = store.list_all(viewer.state)[0]["id"]
        published(store)
        emit("record_memory", result=result, receipt=store.receipt(memory_id))
        turn(viewer, "probe-recall-token")
        components[pm.ProjectMemoryRecall](viewer.state)
        require(any(item["id"] == memory_id for item in getattr(viewer.state, pm.PENDING_RECALLS_ATTR)),
                "regex memory did not recall")
        components[pm.SuppressMemory].execute(viewer.state, id=memory_id, turns=20)
        require(memory_id in getattr(viewer.state, pm.SUPPRESSIONS_ATTR), "suppression missing")
        require(not getattr(viewer.state, pm.PENDING_RECALLS_ATTR), "suppressed recall still pending")
        close_store(store)

        shared_root = project_dir / "plugin-data" / "project-memory"
        parked = shared_root.with_name("project-memory-offline")
        shared_root.rename(parked)
        shared_root.write_text("Private diagnostic outage: not a directory\n")
        offline, offline_components = plugin_session()
        attach(offline, offline_components)
        offline_store = offline_components[pm.ProjectMemoryStore]
        require(offline_store.get(offline.state, memory_id)["content"] == "private durable memory",
                "last-good memory disappeared during outage")
        result = offline_components[pm.UpdateMemory].execute(
            offline.state, id=memory_id, content="private queued update")
        require(offline_store.receipt(memory_id)["local_saved"], "update not locally durable")
        require(not offline_store.receipt(memory_id)["shared_saved"], "outage claimed shared update")
        offline_components[pm.SuppressMemory].execute(offline.state, id=memory_id, turns=20)
        close_store(offline_store)
        sid = offline.state._session_id
        locator = project_dir / "sessions" / sid / "session.json"
        persistence = SessionPersistence(locator, session_id=sid, instance_id=uuid.uuid4().hex,
                                         local_state_root=local)
        writers.append(persistence)
        commit = save(offline, persistence)
        restored, restored_components = plugin_session(document=persistence.load(commit_id=commit).document)
        attach(restored, restored_components)
        restored_store = restored_components[pm.ProjectMemoryStore]
        require(restored_store.get(restored.state, memory_id)["content"] == "private queued update",
                "queued update lost across canonical restore")
        require(restored_store.snapshot_status()["last_good"], "last-good cache lost across restore")
        require(memory_id in getattr(restored.state, pm.SUPPRESSIONS_ATTR),
                "durable session suppression lost across restore")
        require("private queued update" in read_buffer(restored, "project_memory"),
                "restored memory buffer did not expose queued update")
        emit("offline_restore", result=result, status=restored_store.snapshot_status(), commit_id=commit)
        close_store(restored_store)
        shared_root.unlink()
        parked.rename(shared_root)
        recovered, recovered_components = plugin_session()
        attach(recovered, recovered_components)
        recovered_store = recovered_components[pm.ProjectMemoryStore]
        published(recovered_store)
        require(recovered_store.get(recovered.state, memory_id)["content"] == "private queued update",
                "recovered update missing")
        result = recovered_components[pm.DeleteMemory].execute(recovered.state, id=memory_id)
        published(recovered_store)
        close_store(recovered_store)
        final, final_components = plugin_session()
        attach(final, final_components)
        final_store = final_components[pm.ProjectMemoryStore]
        published(final_store)
        require(final_store.get(final.state, memory_id) is None, "deletion lost on restart")
        from tmux_pilot.fs_store import RecordStore
        tombstone = RecordStore(shared_root, create=False).get("project_memory", memory_id)
        require(tombstone and tombstone["deleted"], "shared tombstone missing")
        emit("delete_memory", result=result, tombstone=tombstone)
        emit("complete", status="PASS", root=root,
             limits="component/filesystem workflow only; no backend/TUI/provider or NFS qualification")
    finally:
        for store in stores:
            store.close(timeout_s=2)
        for persistence in writers:
            persistence.close(timeout_s=2)


if __name__ == "__main__":
    main()
