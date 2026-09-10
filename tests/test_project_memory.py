from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from agent_utils import Entry, Feature, Session, State
from agent_utils.types import ToolCall
from agent_zoo import pipelines
from agent_zoo.modes import default_modes, rlm_modes
from agent_utils.files.buffer_manager import BufferManager

import project_memory as pm


def entry(index: int, role: str, content: str, *, step: int | None = None, **message):
    return Entry(
        messages=[{"role": role, "content": content, **message}],
        index=index,
        step=index if step is None else step,
    )


@pytest.fixture
def project_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_LOCAL_STATE_ROOT", str(tmp_path / "local"))
    project = tmp_path / "shared" / "projects" / "demo"
    project.mkdir(parents=True)
    return project


def state_for(db: Path, *, run_id: str = "run-1", step: int = 1) -> State:
    state = State(token_budget=10_000)
    state._session_id = run_id
    state._agent_zoo_context = {"project_name": "demo", "project_dir": str(db), "session_id": run_id}
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
