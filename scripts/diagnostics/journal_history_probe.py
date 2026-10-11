#!/usr/bin/env python3
"""Opt-in, provider-free canonical session-history workflow; never run on live state.

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
    writers = []

    def turn(session, marker):
        for role, text in (("user", marker), ("assistant", "answer " + marker)):
            session.state.entries.append(Entry(
                messages=[{"role": role, "content": text}],
                index=len(session.state.entries), step=session.state.step))

    def plugin_session(history=True, live=True, enabled=True):
        session = Session(system_prompt="Private diagnostic; no provider calls.")
        session.state.buffer_manager = BufferManager()
        builder = PipelineBuilder(session).add(core()).add(buffer_editing())
        pm.register_features(builder, session=session, config={"project_memory": {
            "enabled": enabled,
            "project_sessions": {"enabled": history},
            "live_sessions": {"enabled": live}}})
        session.pipeline = builder.build()
        session._init_state()
        session.state._session_id = getattr(session.state, "_session_id", str(uuid.uuid4()))
        session.state._agent_zoo_context = {
            "project_name": project, "project_dir": str(project_dir),
            "local_state_root": str(local)}
        components = {type(item): item for item in flatten(session.pipeline)}
        return session, components

    def attach(session, components):
        for cls in (pm.ProjectSessionBuffers, pm.ProjectLiveSessionBuffer):
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

    def restore(locator, sid, commit):
        loaded = load_session(locator, session_id=sid, commit_id=commit,
                              local_state_root=local, timeout_s=0)
        if isinstance(loaded, PendingSessionLoad):
            loaded = loaded.wait(timeout_s=10)
        require(not isinstance(loaded, PendingSessionLoad), "exact reload stayed pending")
        require(loaded.state.commit_id == commit, "exact reload selected another commit")
        restored = Session()
        restored.load_document(loaded.state.document)
        restored._init_state()
        return str([entry.messages for entry in restored.state.entries])

    try:
        for history, live, enabled in ((True, True, True), (True, False, True),
                                       (False, True, True), (False, False, True),
                                       (True, True, False)):
            session, components = plugin_session(history, live, enabled)
            require((pm.ProjectSessionBuffers in components) == (history and enabled),
                    "history control coupled")
            require((pm.ProjectLiveSessionBuffer in components) == (live and enabled),
                    "live control coupled")
            attach(session, components)
            manager = session.state.buffer_manager
            require(manager.is_special("project_sessions") == (history and enabled),
                    "history buffer control coupled")
            require(manager.is_special("ses_12345678") == (history and enabled),
                    "history namespace control coupled")
            require(manager.is_special("live_sessions") == (live and enabled),
                    "live buffer control coupled")
            emit("controls", history=history, live=live, enabled=enabled)

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
            observer, observed_components = plugin_session(live=False)
            observer.state._agent_zoo_context["local_state_root"] = str(root / "cold-local")
            attach(observer, observed_components)
            buffer_id, pinned = projection(observer, sid, marker, first)
            turn(source, "after-pin-" + sid)
            second = save(source, persistence)
            require(first != second, "new turn did not produce a new commit")
            require(read_buffer(observer, buffer_id) == pinned, "existing buffer silently retargeted")
            restarted, restarted_components = plugin_session(live=False)
            attach(restarted, restarted_components)
            projection(restarted, sid, "after-pin-" + sid, second)
            content = restore(locator, sid, first)
            require(marker in content and "after-pin-" + sid not in content, "exact restore changed history")
            require(not locator.exists(), "journal save created a loose snapshot")
            emit("history_restore", mixed=mixed, first=first, second=second, stable_pin=True)

            # Block shared history after a successful local restore. An issued
            # alias stays pinned, and canonical restore can use the local journal.
            shared_root = locator.parent
            parked = shared_root.with_name(sid + "-offline")
            shared_root.rename(parked)
            try:
                shared_root.write_text("Private diagnostic outage: not a directory\n")
                require(read_buffer(observer, buffer_id) == pinned,
                        "pinned history changed during shared outage")
                require(restore(locator, sid, first) == content,
                        "local exact restore changed during shared outage")
                emit("history_outage", mixed=mixed, commit_id=first,
                     stable_pin=True, local_restore=True)
            finally:
                shared_root.unlink(missing_ok=True)
                parked.rename(shared_root)

        emit("complete", status="PASS", root=root,
             limits="component/filesystem workflow only; no backend/TUI/provider or NFS qualification")
    finally:
        for persistence in writers:
            persistence.close(timeout_s=2)


if __name__ == "__main__":
    main()
