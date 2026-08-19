from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_utils import Entry, Feature, Session, State
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


def test_local_crud_invalidates_same_step_buffer_cache(project_db):
    state = state_for(project_db)
    store = pm.ProjectMemoryStore()
    pm.ProjectMemoryBuffer(store)(state)
    assert "Count: 0" in state.buffer_manager.resolve_for_read(state, "project_memory").text

    memory = store.create(state, "Current immediately.", "immediately")
    current = state.buffer_manager.resolve_for_read(state, "project_memory")
    assert memory["id"] in current.text


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


def test_system_prompt_wrapper_is_idempotent_and_model_only():
    state = State(token_budget=10_000)
    state.entries = [entry(0, "system", "Base prompt", step=0)]
    component = pm.ProjectMemorySystemPrompt("Remember deliberately.")

    component(state)
    component(state)
    rendered = state.entries[0].render_fn(state)
    assert rendered[0]["content"].count("# Project Memory") == 1
    assert "Remember deliberately." in rendered[0]["content"]
    assert state.entries[0].messages[0]["content"] == "Base prompt"


def test_explicit_config_overrides_installed_config(tmp_path, monkeypatch):
    config_path = tmp_path / "project_memory.json"
    config_path.write_text(
        '{"project_memory": {"enabled": false, "system_prompt": "installed"}}',
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
