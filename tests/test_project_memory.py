from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import pytest
from agent_utils import Entry, Feature, Session, State
from agent_utils.types import ToolCall
from agent_zoo import pipelines
from agent_zoo.modes import default_modes, rlm_modes
from agent_utils.files.buffer_manager import BufferManager
from tmux_pilot.database import AgentDB, RunDB

import project_memory as pm


def entry(index: int, role: str, content: str, *, step: int | None = None, **message):
    return Entry(
        messages=[{"role": role, "content": content, **message}],
        index=index,
        step=index if step is None else step,
    )


@pytest.fixture
def project_db(tmp_path):
    db = AgentDB(tmp_path / "project.sqlite")
    db.init()
    yield db
    db.close()


def state_for(db: AgentDB, *, run_id: str = "run-1", step: int = 1) -> State:
    state = State(token_budget=10_000)
    state.run_db = RunDB(db, run_id, session_id=run_id)
    state.buffer_manager = BufferManager()
    state.step = step
    state.entries = [entry(0, "system", "system prompt", step=0)]
    state.project_memory_suppressions = {}
    state.project_memory_seen_events = []
    state.project_memory_pending_recalls = []
    state.project_memory_scan_bootstrapped = False
    state.project_memory_last_entry_count = 0
    return state


def add_message(state: State, role: str, content: str, **message) -> Entry:
    message_step = message.pop("step", state.step)
    item = entry(len(state.entries), role, content, step=message_step, **message)
    state.entries.append(item)
    return item


def saved_entry(
    index: int,
    role: str,
    content: str,
    *,
    system_generated: bool = False,
    tool_calls: list[dict] | None = None,
) -> dict:
    item = {
        "messages": [{"role": role, "content": content}],
        "index": index,
        "step": index,
        "tokens": 0,
        "pinned": False,
        "compressed": False,
        "forgotten": False,
        "system_generated": system_generated,
        "mode_at_submission": None,
        "mode_user_message_addendum": "",
        "mode_user_message_addendum_role": "system",
        "mode_submission_recorded": False,
    }
    if tool_calls:
        item["tool_calls"] = tool_calls
    return item


def write_saved_session(project_dir, session_id: str, entries: list[dict]):
    session_dir = project_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    source = session_dir / "session.json"
    source.write_text(
        json.dumps(
            {
                "format_version": 1,
                "meta": {"session_id": session_id},
                "entries": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def project_session_state(project_dir, current_session_id: str) -> State:
    state = State(token_budget=10_000)
    state.step = 1
    state.buffer_manager = BufferManager()
    state._session_id = current_session_id
    state._agent_zoo_context = {
        "project_name": "demo",
        "project_dir": str(project_dir),
        "session_id": current_session_id,
        "project_session_index": str(project_dir / "sessions" / "index.json"),
    }
    return state


def call_tool(tool, state: State, **kwargs) -> str:
    """Dispatch one tool call against the in-memory transcript."""
    tool_call = ToolCall(
        id=f"call-{len(state.entries)}",
        name=tool.name,
        raw_args=json.dumps(kwargs),
        parsed_args=kwargs,
    )
    assistant = add_message(
        state,
        "assistant",
        "",
        tool_calls=[
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool.name,
                    "arguments": tool_call.raw_args,
                },
            }
        ],
    )
    assistant.tool_calls = [tool_call]
    state.pending_tool_calls = [tool_call]
    tool.transform(state)
    state.pending_tool_calls = []
    assert tool_call.result is not None
    assert tool_call.error is False
    return str(tool_call.result)


def test_store_crud_and_short_ids(project_db):
    state = state_for(project_db)
    store = pm.ProjectMemoryStore()

    memory = store.create(state, "Use pixi run for tests.", r"\bpixi\b")
    assert memory["id"].startswith("mem_")
    assert len(memory["id"]) == 10
    assert store.get(state, memory["id"])["content"] == "Use pixi run for tests."

    updated = store.update(state, memory["id"], trigger=r"pytest|pixi")
    assert updated["content"] == memory["content"]
    assert updated["trigger"] == r"pytest|pixi"
    assert store.list_all(state) == [updated]

    assert store.delete(state, memory["id"]) is True
    assert store.delete(state, memory["id"]) is False
    assert store.list_all(state) == []


def test_store_rejects_invalid_values(project_db):
    state = state_for(project_db)
    store = pm.ProjectMemoryStore()

    with pytest.raises(ValueError, match="content"):
        store.create(state, "  ", "valid")
    with pytest.raises(ValueError, match="invalid memory trigger regex"):
        store.create(state, "content", "[")


def test_store_startup_and_reads_do_not_create_schema(project_db):
    state = state_for(project_db)
    store = pm.ProjectMemoryStore()

    store(state)
    assert store.list_all(state) == []
    assert store.get(state, "mem_missing") is None
    assert project_db.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (pm.TABLE_NAME,),
    ).fetchone() is None


def test_project_memory_buffer_is_live_readonly_and_shared(project_db):
    first = state_for(project_db, run_id="one")
    second = state_for(project_db, run_id="two")
    store = pm.ProjectMemoryStore()
    component = pm.ProjectMemoryBuffer(store)
    component(first)
    component(second)

    initial = first.buffer_manager.resolve_for_read(first, "project_memory")
    assert initial.readonly is True
    assert initial.id == "project_memory"
    assert initial.path == "memory://project"
    assert "Count: 0" in initial.text

    memory = store.create(second, "Shared across project sessions.", r"cross[- ]session")
    component(first)
    refreshed = first.buffer_manager.resolve_for_read(first, "project_memory")
    assert memory["id"] in refreshed.text
    assert 'Trigger regex: "cross[- ]session"' in refreshed.text
    assert "Shared across project sessions." in refreshed.text


def test_project_memory_buffer_provider_is_pickle_safe(project_db):
    state = state_for(project_db)
    store = pm.ProjectMemoryStore()
    component = pm.ProjectMemoryBuffer(store)
    manager = state.buffer_manager
    component(state)
    assert state.buffer_manager is manager
    provider = state.buffer_manager._special_buffers["project_memory"]
    assert type(provider).__module__ == "operator"
    assert type(provider).__name__ == "methodcaller"
    memory = store.create(state, "Survive RLM state serialization.", r"serialization")

    restored_manager = pickle.loads(pickle.dumps(state.buffer_manager))
    state.buffer_manager = restored_manager
    state.__dict__.pop("_project_memory_render", None)
    component(state)

    rendered = state.buffer_manager.resolve_for_read(state, "project_memory")
    assert memory["id"] in rendered.text
    assert "Survive RLM state serialization." in rendered.text




def test_project_memory_buffer_supports_legacy_exact_registration_api():
    class LegacyBufferManager:
        def __init__(self):
            self._special_buffers = {}
            self._special_cache = {"project_memory": (1, object())}

        def is_special(self, buffer_id):
            return buffer_id in self._special_buffers

        def register_special_buffer(self, buffer_id, provider):
            if buffer_id in self._special_buffers:
                raise ValueError("already registered")
            self._special_buffers[buffer_id] = provider

    state = SimpleNamespace(buffer_manager=LegacyBufferManager())
    component = pm.ProjectMemoryBuffer(pm.ProjectMemoryStore())

    component(state)
    component(state)

    provider = state.buffer_manager._special_buffers["project_memory"]
    assert type(provider).__name__ == "methodcaller"
    assert "project_memory" not in state.buffer_manager._special_cache


def test_local_crud_invalidates_same_step_buffer_cache(project_db):
    state = state_for(project_db)
    store = pm.ProjectMemoryStore()
    pm.ProjectMemoryBuffer(store)(state)
    assert "Count: 0" in state.buffer_manager.resolve_for_read(state, "project_memory").text

    memory = store.create(state, "Current immediately.", "immediately")
    current = state.buffer_manager.resolve_for_read(state, "project_memory")
    assert memory["id"] in current.text


def test_project_session_index_and_compact_buffers_are_lazy_and_filtered(tmp_path):
    project_dir = tmp_path / "demo"
    sessions_dir = project_dir / "sessions"
    sessions_dir.mkdir(parents=True)
    current_id = "aaaaaaaa-0000-0000-0000-000000000000"
    recent_id = "344655c1-1111-1111-1111-111111111111"
    older_id = "97568885-2222-2222-2222-222222222222"
    rlm_id = "bbbbbbbb-3333-3333-3333-333333333333"

    write_saved_session(project_dir, current_id, [saved_entry(0, "user", "current")])
    recent_source = write_saved_session(
        project_dir,
        recent_id,
        [
            saved_entry(0, "system", "system"),
            saved_entry(1, "user", "Run the experiment."),
            saved_entry(
                2,
                "assistant",
                "I will inspect the files.",
                tool_calls=[{"id": "call-1"}],
            ),
            saved_entry(3, "tool", "large tool output"),
            saved_entry(4, "assistant", "The experiment finished successfully."),
            saved_entry(5, "user", "terminal completion", system_generated=True),
            saved_entry(6, "user", "Fine tune all checkpoints."),
            saved_entry(7, "assistant", "I will fine tune all eight checkpoints."),
        ],
    )
    write_saved_session(project_dir, older_id, [saved_entry(0, "user", "older")])
    write_saved_session(project_dir, rlm_id, [saved_entry(0, "user", "rlm")])
    index = [
        {
            "session_id": older_id,
            "title": "older",
            "description": "old description",
            "kind": "default",
            "created_at": 100.0,
            "updated_at": 200.0,
        },
        {
            "session_id": current_id,
            "title": "current",
            "description": "active",
            "kind": "default",
            "created_at": 300.0,
            "updated_at": 900.0,
        },
        {
            "session_id": rlm_id,
            "title": "child",
            "description": "hidden",
            "kind": "rlm",
            "created_at": 400.0,
            "updated_at": 800.0,
        },
        {
            "session_id": recent_id,
            "title": "recent experiment",
            "description": "two-line\ndescription",
            "kind": "default",
            "created_at": 500.0,
            "updated_at": 700.0,
        },
    ]
    (sessions_dir / "index.json").write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )
    state = project_session_state(project_dir, current_id)
    component = pm.ProjectSessionBuffers(clock=lambda: 1_700_000_000.0)

    component(state)
    cache_root = project_dir / "plugin-data" / "project-memory" / "compact"
    assert not cache_root.exists()

    rendered_index = state.buffer_manager.resolve_for_read(state, "project_sessions")
    assert rendered_index.readonly is True
    assert "Project Sessions" in rendered_index.text
    assert "Buffer Generated (UTC): 2023-11-14 22:13:20" in rendered_index.text
    assert "Sessions: 2" in rendered_index.text
    assert current_id not in rendered_index.text
    assert rlm_id not in rendered_index.text
    assert rendered_index.text.index(recent_id) < rendered_index.text.index(older_id)
    assert "Buffer: ses_344655c1" in rendered_index.text
    assert "Title: recent experiment | Description: two-line description" in rendered_index.text
    assert not cache_root.exists()

    compact = state.buffer_manager.resolve_for_read(state, "ses_344655c1")
    assert compact.readonly is True
    assert compact.path == f"project-memory://sessions/{recent_id}"
    assert f"Session ID: {recent_id}" in compact.text
    assert "Title: recent experiment" in compact.text
    assert "Description: two-line description" in compact.text
    assert "Created (UTC): 1970-01-01 00:08:20" in compact.text
    assert "Updated (UTC): 1970-01-01 00:11:40" in compact.text
    assert "Buffer Generated (UTC): 2023-11-14 22:13:20" in compact.text
    assert f"Source Transcript: {recent_source.resolve()}" in compact.text
    assert "USER [entry 1, lines " in compact.text
    assert "ASSISTANT [entry 4, lines " in compact.text
    assert "USER [entry 6, lines " in compact.text
    assert "ASSISTANT [entry 7, lines " in compact.text
    assert compact.text.count("\nUSER [") == 2
    assert compact.text.count("\nASSISTANT [") == 2
    assert "I will inspect the files." not in compact.text
    assert "large tool output" not in compact.text
    assert "terminal completion" not in compact.text
    assert "The experiment finished successfully." in compact.text
    assert cache_root.is_dir()

    with pytest.raises(KeyError, match="No buffer 'ses_bbbbbbbb'"):
        state.buffer_manager.resolve_for_read(state, "ses_bbbbbbbb")
    with pytest.raises(KeyError, match="No buffer 'ses_aaaaaaaa'"):
        state.buffer_manager.resolve_for_read(state, "ses_aaaaaaaa")


def test_compact_session_cache_reuses_and_invalidates_source(tmp_path, monkeypatch):
    project_dir = tmp_path / "demo"
    session_id = "344655c1-1111-1111-1111-111111111111"
    source = write_saved_session(
        project_dir,
        session_id,
        [saved_entry(0, "user", "first"), saved_entry(1, "assistant", "original")],
    )
    (project_dir / "sessions" / "index.json").write_text(
        json.dumps(
            [
                {
                    "session_id": session_id,
                    "title": "cache",
                    "description": "",
                    "kind": "default",
                    "created_at": 1.0,
                    "updated_at": 2.0,
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    state = project_session_state(project_dir, "ffffffff-0000-0000-0000-000000000000")
    component = pm.ProjectSessionBuffers(clock=lambda: 3.0)
    component(state)
    original = pm._build_compact_transcript
    calls = []

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pm, "_build_compact_transcript", tracked)
    first = state.buffer_manager.resolve_for_read(state, "ses_344655c1")
    second = state.buffer_manager.resolve_for_read(state, "ses_344655c1")
    assert first.text == second.text
    assert len(calls) == 1

    write_saved_session(
        project_dir,
        session_id,
        [
            saved_entry(0, "user", "first"),
            saved_entry(1, "assistant", "changed response with more text"),
        ],
    )
    assert source.exists()
    changed = state.buffer_manager.resolve_for_read(state, "ses_344655c1")
    assert "changed response with more text" in changed.text
    assert len(calls) == 2


def test_session_buffer_ids_extend_colliding_prefixes():
    sessions = [
        {"session_id": "12345678-a000-0000-0000-000000000000"},
        {"session_id": "12345678-b000-0000-0000-000000000000"},
        {"session_id": "abcdef12-c000-0000-0000-000000000000"},
    ]

    assert pm._session_buffer_ids(sessions) == {
        "12345678-a000-0000-0000-000000000000": "ses_12345678a",
        "12345678-b000-0000-0000-000000000000": "ses_12345678b",
        "abcdef12-c000-0000-0000-000000000000": "ses_abcdef12",
    }


def test_project_session_namespace_provider_survives_manager_pickle(tmp_path):
    project_dir = tmp_path / "demo"
    session_id = "344655c1-1111-1111-1111-111111111111"
    write_saved_session(project_dir, session_id, [saved_entry(0, "user", "hello")])
    (project_dir / "sessions" / "index.json").write_text(
        json.dumps(
            [
                {
                    "session_id": session_id,
                    "title": "pickle",
                    "description": "",
                    "kind": "default",
                    "created_at": 1.0,
                    "updated_at": 2.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    state = project_session_state(project_dir, "ffffffff-0000-0000-0000-000000000000")
    component = pm.ProjectSessionBuffers(clock=lambda: 3.0)
    component(state)

    restored = pickle.loads(pickle.dumps(state.buffer_manager))
    state.buffer_manager = restored
    state.__dict__.pop(pm.PROJECT_SESSION_RENDER_ATTR, None)
    state.__dict__.pop(pm.PROJECT_SESSION_INDEX_RENDER_ATTR, None)
    component(state)

    compact = state.buffer_manager.resolve_for_read(state, "ses_344655c1")
    assert "USER [entry 0, lines " in compact.text
    provider = state.buffer_manager._special_buffer_namespaces["ses_"]
    assert type(provider).__module__ == "agent_utils.files.buffer_manager"


def test_record_memory_backtests_messages_and_tool_results(project_db):
    state = state_for(project_db, step=4)
    store = pm.ProjectMemoryStore()
    recall = pm.ProjectMemoryRecall(store)
    record = pm.RecordMemory(store, recall)
    add_message(state, "user", "The release failed during staging.", step=1)
    add_message(
        state,
        "assistant",
        "",
        step=2,
        tool_calls=[
            {
                "id": "call-inspect",
                "type": "function",
                "function": {
                    "name": "inspect_release",
                    "arguments": '{"environment": "staging"}',
                },
            }
        ],
    )
    add_message(state, "tool", "deployment failed in staging", step=3, tool_call_id="call-inspect")

    result = call_tool(
        record,
        state,
        content="Remember the release recovery procedure.",
        trigger=r"(?:release|deployment)\s+failed",
    )

    assert "Backtest: trigger matched 2 events" in result
    assert "- entry 3: Tool result: deployment failed in staging" in result
    assert "- entry 1: user message: The release failed during staging." in result


def test_record_memory_reports_no_backtest_matches(project_db):
    state = state_for(project_db, step=3)
    store = pm.ProjectMemoryStore()
    recall = pm.ProjectMemoryRecall(store)
    record = pm.RecordMemory(store, recall)
    add_message(state, "user", "The build completed successfully.", step=2)

    result = call_tool(
        record,
        state,
        content="Remember the deployment checklist.",
        trigger=r"NEVER_MATCH_[0-9]+",
    )

    assert "Backtest: no messages or tool calls matched in the active context window." in result


def test_record_memory_backtest_respects_harness_handoff_boundary(project_db):
    state = state_for(project_db, step=8)
    store = pm.ProjectMemoryStore()
    recall = pm.ProjectMemoryRecall(store)
    record = pm.RecordMemory(store, recall)
    state.entries = [
        entry(0, "system", "system", step=0),
        entry(1, "assistant", "Old NEEDLE123 history", step=1),
        entry(2, "assistant", "old history", step=2),
        entry(3, "user", "Current NEEDLE123 remains relevant", step=8),
    ]
    state.context_compaction = SimpleNamespace(
        handoff_docs=[
            {
                "text": "Summary without the old match.",
                "source_end_turn": 7,
                "source_end_entry_index": 2,
                "preserved_tail_start_entry_index": 3,
            }
        ]
    )

    result = call_tool(
        record,
        state,
        content="Remember the current deployment state.",
        trigger=r"NEEDLE[0-9]+",
    )

    assert "Backtest: trigger matched 1 event" in result
    assert "- entry 3: user message: Current NEEDLE123 remains relevant" in result
    assert "entry 1" not in result


def test_record_memory_backtest_respects_provider_compaction_boundary(project_db):
    state = state_for(project_db, step=9)
    store = pm.ProjectMemoryStore()
    recall = pm.ProjectMemoryRecall(store)
    record = pm.RecordMemory(store, recall)
    state.entries = [
        entry(0, "system", "system", step=0),
        entry(1, "assistant", "Old NEEDLE123 history", step=1),
        entry(
            2,
            "assistant",
            "",
            step=8,
            provider_tool_calls=[
                {"type": "compaction", "encrypted_content": "opaque-provider-state"}
            ],
        ),
        entry(3, "user", "Current NEEDLE123 remains relevant", step=9),
    ]

    result = call_tool(
        record,
        state,
        content="Remember the current deployment state.",
        trigger=r"NEEDLE[0-9]+",
    )

    assert "Backtest: trigger matched 1 event" in result
    assert "- entry 3: user message: Current NEEDLE123 remains relevant" in result
    assert "entry 1" not in result

def test_common_memories_are_runtime_only_stable_and_buffered(project_db):
    state = state_for(project_db)
    store = pm.ProjectMemoryStore(
        {
            "frustration_reflection": {
                "enabled": True,
                "content": "Reflect before continuing.",
                "trigger": r"frustration",
            }
        }
    )

    store(state)
    configured = store.list_common()
    assert len(configured) == 1
    memory = configured[0]
    assert memory["id"].startswith("mem_")
    assert len(memory["id"]) == 10
    assert store.list_all(state) == []

    store(state)
    assert store.list_common()[0]["id"] == memory["id"]

    pm.ProjectMemoryBuffer(store)(state)
    rendered = state.buffer_manager.resolve_for_read(state, "project_memory").text
    assert "# Common configured memories" in rendered
    assert memory["id"] in rendered
    assert "Enabled: yes" in rendered
    assert "Reflect before continuing." in rendered


def test_enabled_common_memory_is_recalled_without_project_row(project_db):
    state = state_for(project_db, step=2)
    store = pm.ProjectMemoryStore(
        {
            "configured": {
                "enabled": True,
                "content": "Configured guidance.",
                "trigger": r"configured topic",
            }
        }
    )
    add_message(state, "user", "Discuss the configured topic now", step=2)

    pm.ProjectMemoryRecall(store)(state)

    configured = store.list_common()[0]
    assert state.project_memory_pending_recalls == [
        {"id": configured["id"], "content": configured["content"]}
    ]
    assert store.list_all(state) == []


def test_disabled_common_memory_is_not_recalled_but_remains_visible(project_db):
    state = state_for(project_db, step=2)
    store = pm.ProjectMemoryStore(
        {
            "disabled": {
                "enabled": False,
                "content": "Disabled guidance.",
                "trigger": r"disabled topic",
            }
        }
    )
    add_message(state, "user", "Discuss disabled topic now", step=2)

    pm.ProjectMemoryRecall(store)(state)

    assert state.project_memory_pending_recalls == []
    assert store.list_common()[0]["enabled"] is False


def test_common_memory_crud_is_read_only(project_db):
    state = state_for(project_db)
    store = pm.ProjectMemoryStore(
        {
            "configured": {
                "content": "Edit this in YAML.",
                "trigger": r"configured",
            }
        }
    )
    store(state)
    memory_id = store.list_common()[0]["id"]

    with pytest.raises(ValueError, match="read-only"):
        store.update(state, memory_id, content="Project override")
    with pytest.raises(ValueError, match="read-only"):
        store.delete(state, memory_id)
    assert store.list_all(state) == []


def test_common_memory_config_merges_bundled_and_user_entries(monkeypatch):
    monkeypatch.setattr(
        pm.ProjectMemorySystemPrompt,
        "_load_bundled_config",
        lambda: {
            "project_memory": {
                "common_memories": {
                    "bundled": {"content": "Bundled", "trigger": "bundled"},
                    "shared": {"content": "Old", "trigger": "old"},
                }
            }
        },
    )
    monkeypatch.setattr(
        pm.ProjectMemorySystemPrompt,
        "_load_installed_config",
        lambda: {
            "project_memory": {
                "common_memories": {
                    "installed": {"content": "Installed", "trigger": "installed"},
                }
            }
        },
    )

    section = pm.ProjectMemorySystemPrompt.effective_section(
        {
            "project_memory": {
                "common_memories": {
                    "explicit": {"content": "Explicit", "trigger": "explicit"},
                    "shared": {"content": "New", "trigger": "new"},
                }
            }
        }
    )
    common = pm.ProjectMemorySystemPrompt.common_memories(section)

    assert set(common) == {"bundled", "installed", "explicit", "shared"}
    assert common["shared"]["content"] == "New"

    disabled_section = {
        **section,
        "include_bundled_memories": False,
    }
    disabled_common = pm.ProjectMemorySystemPrompt.common_memories(disabled_section)
    assert set(disabled_common) == {"installed", "explicit", "shared"}


def test_recall_buffers_after_user_and_delivers_after_assistant(project_db):
    state = state_for(project_db, step=4)
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "The deploy command is pixi run deploy.", r"deploy")
    add_message(state, "user", "How do I deploy this project?")

    recall = pm.ProjectMemoryRecall(store)
    delivery = pm.ProjectMemoryDelivery()
    recall(state)
    assert state.project_memory_pending_recalls == [
        {"id": memory["id"], "content": memory["content"]}
    ]
    delivery(state)
    assert state.entries[-1].role == "user"
    assert state.entries[-1].system_generated is False

    add_message(state, "assistant", "Let me check.")
    delivery(state)
    recalled = state.entries[-1]
    assert recalled.system_generated is True
    assert recalled.messages[0]["content"] == (
        f"[Memory recalled | {memory['id']}]\n{memory['content']}"
    )
    assert state.done is False


def test_recall_matches_tool_calls_and_results_without_duplicates(project_db):
    state = state_for(project_db, step=2)
    store = pm.ProjectMemoryStore()
    call_memory = store.create(state, "Check generated artifacts after builds.", r"build_project")
    result_memory = store.create(state, "The flaky test is documented in issue 42.", r"FAIL.*issue 42")
    add_message(
        state,
        "assistant",
        "",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "build_project", "arguments": '{"release": true}'},
            }
        ],
    )
    add_message(state, "tool", "FAIL: see issue 42", tool_call_id="call-1")

    recall = pm.ProjectMemoryRecall(store)
    recall(state)
    assert {item["id"] for item in state.project_memory_pending_recalls} == {
        call_memory["id"],
        result_memory["id"],
    }

    recall(state)
    assert len(state.project_memory_pending_recalls) == 2


def test_memory_already_in_active_context_is_not_recalled(project_db):
    state = state_for(project_db, step=3)
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "Always run the formatter before commit.", r"commit")
    add_message(state, "assistant", memory["content"], step=2)
    add_message(state, "user", "I am ready to commit.", step=3)

    pm.ProjectMemoryRecall(store)(state)
    assert state.project_memory_pending_recalls == []


def test_harness_handoff_defines_active_context_defensively(project_db):
    state = state_for(project_db, step=8)
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "Legacy deployment uses blue-green.", r"deployment")
    state.entries = [
        entry(0, "system", "system", step=0),
        entry(1, "assistant", memory["content"], step=1),
        entry(2, "assistant", "old history", step=2),
        entry(3, "user", "Discuss deployment", step=8),
    ]
    state.context_compaction = SimpleNamespace(
        handoff_docs=[
            {
                "text": "Summary without the remembered fact.",
                "source_end_turn": 7,
                "source_end_entry_index": 2,
                "preserved_tail_start_entry_index": 3,
            }
        ]
    )

    pm.ProjectMemoryRecall(store)(state)
    assert [item["id"] for item in state.project_memory_pending_recalls] == [memory["id"]]

    state.project_memory_pending_recalls.clear()
    state.project_memory_seen_events.clear()
    state.project_memory_scan_bootstrapped = False
    state.context_compaction.handoff_docs[0]["text"] = memory["content"]
    pm.ProjectMemoryRecall(store)(state)
    assert state.project_memory_pending_recalls == []


def test_provider_encrypted_compaction_defines_active_context(project_db):
    state = state_for(project_db, step=9)
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "Use the v2 migration checklist.", r"migration")
    state.entries = [
        entry(0, "system", "system", step=0),
        entry(1, "assistant", memory["content"], step=1),
        entry(
            2,
            "assistant",
            "",
            step=8,
            provider_tool_calls=[
                {"type": "compaction", "encrypted_content": "opaque-provider-state"}
            ],
        ),
        entry(3, "user", "Start the migration", step=9),
    ]

    pm.ProjectMemoryRecall(store)(state)
    assert [item["id"] for item in state.project_memory_pending_recalls] == [memory["id"]]


def test_suppress_memory_removes_pending_and_expires_by_turn(project_db):
    state = state_for(project_db, step=10)
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "Prefer the local fixture.", r"fixture")
    state.project_memory_pending_recalls.append(
        {"id": memory["id"], "content": memory["content"]}
    )

    message = pm.SuppressMemory(store).execute(state, id=memory["id"], turns=2)
    assert "through step 11" in message
    assert state.project_memory_pending_recalls == []

    add_message(state, "assistant", "fixture one", step=10)
    recall = pm.ProjectMemoryRecall(store)
    recall(state)
    assert state.project_memory_pending_recalls == []

    state.step = 12
    add_message(state, "assistant", "fixture two", step=12)
    recall(state)
    assert [item["id"] for item in state.project_memory_pending_recalls] == [memory["id"]]


def test_update_and_delete_reconcile_pending_state(project_db):
    state = state_for(project_db)
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "Old content", "old")
    state.project_memory_pending_recalls.append({"id": memory["id"], "content": "Old content"})
    state.project_memory_suppressions[memory["id"]] = 99

    pm.UpdateMemory(store).execute(state, id=memory["id"], content="New content")
    assert state.project_memory_pending_recalls[0]["content"] == "New content"

    pm.DeleteMemory(store).execute(state, id=memory["id"])
    assert state.project_memory_pending_recalls == []
    assert memory["id"] not in state.project_memory_suppressions


def test_new_memory_from_another_session_is_seen_on_next_scan(project_db):
    first = state_for(project_db, run_id="first", step=5)
    second = state_for(project_db, run_id="second", step=5)
    store = pm.ProjectMemoryStore()
    add_message(first, "assistant", "baseline")
    recall = pm.ProjectMemoryRecall(store)
    recall(first)

    memory = store.create(second, "Shared memory arrived.", r"shared topic")
    first.step = 6
    add_message(first, "assistant", "Discuss shared topic now", step=6)
    recall(first)
    assert [item["id"] for item in first.project_memory_pending_recalls] == [memory["id"]]


def test_system_prompt_render_transform_is_idempotent_and_model_only():
    state = State(token_budget=10_000)
    state.entries = [entry(0, "system", "Base prompt", step=0)]
    component = pm.ProjectMemorySystemPrompt("Remember deliberately.")
    target = state.entries[0]
    target.initialize_pending_render_channels()

    component(state)
    component(state)
    assert target.render_transform_names(pending=True) == (
        "project memory system prompt",
    )

    target.finalize_render_channels()
    rendered = target.render(state)
    assert rendered[0]["content"].count("# Project Memory") == 1
    assert "Remember deliberately." in rendered[0]["content"]
    assert target.messages[0]["content"] == "Base prompt"


def test_explicit_config_overrides_installed_config(tmp_path, monkeypatch):
    config_path = tmp_path / "project_memory.yaml"
    config_path.write_text(
        "project_memory:\n  enabled: false\n  system_prompt: installed\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(pm.CONFIG_PATH_ENV, str(config_path))

    section = pm.ProjectMemorySystemPrompt.effective_section(
        {"project_memory": {"enabled": True, "system_prompt": "launch"}}
    )
    assert section == {"enabled": True, "system_prompt": "launch"}


def test_register_features_can_be_disabled(monkeypatch):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})

    class Builder:
        def __init__(self):
            self.features = []

        def add(self, feature: Feature):
            self.features.append(feature)
            return self

    builder = Builder()
    pm.register_features(
        builder,
        session=SimpleNamespace(),
        config={"project_memory": {"enabled": False}},
    )
    assert builder.features == []




def test_register_features_degrades_without_session_namespace_support(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    monkeypatch.setattr(pm, "_session_namespaces_supported", lambda: False)
    monkeypatch.setattr(pm, "_SESSION_NAMESPACE_WARNING_EMITTED", False)

    class Builder:
        def __init__(self):
            self.features = []

        def add(self, feature: Feature):
            self.features.append(feature)
            return self

    builder = Builder()
    with caplog.at_level("WARNING"):
        pm.register_features(builder, session=SimpleNamespace(), config={})

    feature = builder.features[0]
    types = {type(component) for component in feature.components}
    assert pm.ProjectMemoryStore in types
    assert pm.ProjectMemoryBuffer in types
    assert pm.RecordMemory in types
    assert pm.ProjectSessionBuffers not in types
    prompt = next(
        component
        for component in feature.components
        if isinstance(component, pm.ProjectMemorySystemPrompt)
    )
    assert "project_sessions" not in prompt.text
    assert "Core project-memory tools remain active" in caplog.text


def test_feature_contains_all_tools_and_special_buffer(monkeypatch):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})

    class Builder:
        def __init__(self):
            self.features = []

        def add(self, feature: Feature):
            self.features.append(feature)
            return self

    builder = Builder()
    pm.register_features(builder, session=SimpleNamespace(), config={})
    feature = builder.features[0]
    assert feature.name == "project_memory"
    types = {type(component) for component in feature.components}
    assert {
        pm.ProjectMemoryStore,
        pm.ProjectMemoryBuffer,
        pm.ProjectSessionBuffers,
        pm.RecordMemory,
        pm.UpdateMemory,
        pm.SuppressMemory,
        pm.DeleteMemory,
        pm.ProjectMemoryRecall,
        pm.ProjectMemoryDelivery,
        pm.ProjectMemorySystemPrompt,
    } <= types


def _install_only_project_memory(builder, *, session, config, **_kwargs):
    pm.register_features(builder, session=session, config=config)


@pytest.mark.parametrize(
    ("builder", "modes"),
    [
        (pipelines.build_default_pipeline, default_modes),
        (pipelines.build_readonly_rlm_pipeline, rlm_modes),
    ],
)
def test_real_harness_pipelines_build_with_project_memory(
    builder, modes, monkeypatch
):
    monkeypatch.setattr(pipelines, "apply_pipeline_plugins", _install_only_project_memory)
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})

    with Session(system_prompt="test") as session:
        pipeline = builder(
            session,
            config={"project_memory": {"system_prompt": "Use memories."}},
            skill_paths=[],
            scheduler_type="local",
            max_idle=0,
            mode_defs=modes(),
            initial_mode="execution",
            terminal_backend="headless",
            tmux_tools_enabled=False,
        )

    types = [type(component) for component in pipeline]
    assert pm.ProjectMemoryBuffer in types
    assert pm.ProjectSessionBuffers in types
    assert pm.ProjectMemoryRecall in types
    assert pm.ProjectMemoryDelivery in types
    assert pm.ProjectMemorySystemPrompt in types
    assert types.index(pm.ProjectMemoryRecall) < types.index(pm.ProjectMemoryDelivery)
    assert types.index(pm.ProjectMemoryDelivery) < types.index(pm.ProjectMemorySystemPrompt)
    tool_names = {
        component.name
        for component in pipeline
        if isinstance(component, pm.ProjectMemoryTool)
    }
    assert tool_names == {
        "record_memory",
        "update_memory",
        "suppress_memory",
        "delete_memory",
    }


def test_first_recall_scan_baselines_existing_transcript(project_db):
    state = state_for(project_db, step=1)
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "Startup convention.", r"startup")
    add_message(state, "assistant", "Discuss startup convention.", step=0)
    recall = pm.ProjectMemoryRecall(store)

    recall(state)
    assert state.project_memory_pending_recalls == []
    assert state.project_memory_scan_bootstrapped is True

    state.step = 2
    add_message(state, "assistant", "New startup discussion.", step=2)
    recall(state)
    assert [item["id"] for item in state.project_memory_pending_recalls] == [memory["id"]]


def test_pre_restore_pass_does_not_bootstrap_recall_state(project_db):
    state = state_for(project_db, step=0)
    store = pm.ProjectMemoryStore()
    memory = store.create(state, "Historic startup guidance.", r"historic startup")
    recall = pm.ProjectMemoryRecall(store)

    state._rlm_restore_ready = False
    recall(state)
    assert state.project_memory_scan_bootstrapped is False
    assert state.project_memory_seen_events == []

    state.entries = [
        entry(0, "system", "restored system prompt", step=0),
        entry(1, "assistant", "Historic startup discussion.", step=5),
    ]
    state.step = 5
    state._rlm_restore_ready = True
    recall(state)

    assert state.project_memory_pending_recalls == []
    assert state.project_memory_scan_bootstrapped is True

    state.step = 6
    add_message(state, "assistant", "New historic startup discussion.", step=6)
    recall(state)
    assert [item["id"] for item in state.project_memory_pending_recalls] == [memory["id"]]
