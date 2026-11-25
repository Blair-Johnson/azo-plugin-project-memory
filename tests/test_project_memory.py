from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_utils.core.events import RenderRequested
from agent_utils.core.session import Session
from agent_utils.core.state import Entry, State
from agent_zoo import pipelines
from agent_zoo.features.modes import default_modes, rlm_modes
from agent_zoo.features.settings import Setting, SettingsContext, SettingsRegistry
from agent_zoo.plugins import loader as plugin_loader
from agent_zoo.testing.sans_io_guard import assert_pipeline_sans_io, audit_module_sans_io

import project_memory as pm


def test_system_prompt_render_transform_is_idempotent_and_model_only():
    state = State(token_budget=10_000)
    state.entries = [Entry(
        messages=[{"role": "system", "content": "Base prompt"}], index=0, step=0,
    )]
    component = pm.ProjectMemorySystemPrompt("Consult session history deliberately.")
    target = state.entries[0]
    target.initialize_pending_render_channels(("model", "user"))

    component.handle(RenderRequested(), state, lambda _ev: None)
    component.handle(RenderRequested(), state, lambda _ev: None)
    assert target.render_transform_names(pending=True) == (
        "project memory system prompt",
    )

    target.finalize_render_channels()
    rendered = target.render(state)
    assert rendered[0]["content"].count("# Project Memory") == 1
    assert "Consult session history deliberately." in rendered[0]["content"]
    assert target.render(state, channel="user") == target.messages
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


def test_feature_contains_history_and_live_buffers(monkeypatch):
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})
    features = []
    pm.register_features(SimpleNamespace(add=features.append), config={})

    assert len(features) == 2
    assert features[0].name == "project_memory_settings"
    assert features[-1].name == "project_memory"
    assert {type(component) for component in features[-1].components} == {
        pm.ProjectSessionBuffers,
        pm.ProjectLiveSessionBuffer,
        pm.ProjectMemorySystemPrompt,
    }
    assert pm.ProjectHistoryDriver in features[-1].drivers


def _install_only_project_memory(builder, *, session, config, **_kwargs):
    del session
    pm.register_features(builder, config=config)


@pytest.mark.parametrize(
    ("builder", "modes"),
    [
        (pipelines.build_default_pipeline, default_modes),
        (pipelines.build_readonly_rlm_pipeline, rlm_modes),
    ],
)
def test_real_harness_pipelines_build_with_project_memory(builder, modes, monkeypatch, tmp_path):
    monkeypatch.setenv(pm.CONFIG_PATH_ENV, str(tmp_path / "settings.yaml"))
    monkeypatch.setattr(plugin_loader, "apply_pipeline_plugins", _install_only_project_memory)
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})

    session = Session(system_prompt="test")
    pipeline = builder(
        session,
        config={"project_memory": {"system_prompt": "Consult session history."}},
        skill_paths=[],
        mode_defs=modes(),
        initial_mode="execution",
        terminal_backend="headless",
        tmux_tools_enabled=False,
        include_llm_drivers=False,
    )

    types = {type(component) for component in pipeline}
    assert pm.ProjectSessionBuffers in types
    assert pm.ProjectLiveSessionBuffer in types
    assert pm.ProjectMemorySystemPrompt in types
    assert_pipeline_sans_io(pipeline)
    registry = SettingsRegistry([component for component in pipeline if isinstance(component, Setting)])
    catalog = registry.handle(
        {"type": "plugin_settings", "request_id": "test", "action": "list"},
        SettingsContext(state=session.state, emit=lambda _ev: None),
    )
    assert catalog["ok"] and catalog["error"] is None
    pm_settings = [row for row in catalog["settings"] if row["key"].startswith("project_memory.")]
    assert len(pm_settings) == 3
    assert all(row["error"] is None for row in catalog["settings"])

    def position(kind):
        return next(i for i, component in enumerate(pipeline) if isinstance(component, kind))

    assert position(pm.RegisterSpecialBuffers) < position(pm.ProjectSessionBuffers)
    assert position(pm.ProjectSessionBuffers) < position(pm.ProjectLiveSessionBuffer)
    assert position(pm.ProjectLiveSessionBuffer) < position(pm.ToolRunner)
    tool_names = {getattr(component, "name", None) for component in pipeline}
    assert tool_names.isdisjoint({
        "record_memory", "update_memory", "suppress_memory", "delete_memory",
    })


@pytest.mark.parametrize("sessions_enabled,live_enabled", [
    (True, False), (False, True), (True, True),
])
@pytest.mark.parametrize("origin", ["installed", "launch", "empty"])
def test_custom_prompt_respected_with_independent_features(
    monkeypatch, sessions_enabled, live_enabled, origin,
):
    installed = {"project_memory": {"system_prompt": "Installed guidance."}}
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: installed)
    section = {"project_sessions": sessions_enabled, "live_sessions": live_enabled}
    expected = {"installed": "Installed guidance.", "launch": "Launch guidance.", "empty": ""}[origin]
    if origin != "installed":
        section["system_prompt"] = expected
    features = []
    pm.register_features(SimpleNamespace(add=features.append),
                         config={"project_memory": section})
    prompt = next(component for component in features[-1].components
                  if isinstance(component, pm.ProjectMemorySystemPrompt))
    assert prompt.text == expected


def test_config_template_preserves_detailed_session_guidance(monkeypatch):
    from pathlib import Path

    template = Path(__file__).parents[1] / "config" / "project_memory.yaml"
    monkeypatch.setenv(pm.CONFIG_PATH_ENV, str(template))
    features = []
    pm.register_features(SimpleNamespace(add=features.append), config={})
    prompt = next(component.text for component in features[-1].components
                  if isinstance(component, pm.ProjectMemorySystemPrompt))
    for guidance in (
        "substantial contiguous sequence of turns", "do not stop at the first turn",
        "older decision", "ses_<id>_full", "newest entries first",
        "Source Commit", "Source Instance", "not journal file lines",
        "live_sessions", "endpoints are not connectivity checks",
        "config/project_memory.yaml as a template", "plugin-configs/azo-plugin-project-memory",
        "Reinstall after source changes", "/reload after installed-guidance changes",
    ):
        assert guidance in prompt
    for removed_tool in ("record_memory", "update_memory", "suppress_memory", "delete_memory"):
        assert removed_tool not in prompt


def test_session_restored_holds_readiness_until_hydration_completes(tmp_path):
    from agent_utils.core.events import IoCompleted, SessionRestored
    from agent_utils.core.io import SubmitOp
    from agent_utils.core.pipeline import PipelineBuilder
    from agent_utils.features.core import core_feature
    from agent_zoo.events import DiscoverSessions, DiscoverSessionsResult

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    session = Session()
    builder = PipelineBuilder().add(core_feature())
    pm.register_features(builder, config={"project_memory": {"enabled": True}})
    session.pipeline = builder.build(suppress_warnings=True)
    session.ensure_initialized()
    session.state._agent_zoo_context = {
        "project_name": "demo",
        "project_dir": str(project_dir),
        "project_store_root": str(project_dir / "coordination"),
    }

    effects = session.step(SessionRestored(reason="resume"))
    cat_eff = next(
        e for e in effects if isinstance(e, SubmitOp) and isinstance(e.request, pm.LoadProjectCatalogRequest)
    )
    live_eff = next(
        e for e in effects if isinstance(e, SubmitOp) and isinstance(e.request, DiscoverSessions)
    )
    assert not session.state.readiness.is_ready
    assert any(k.startswith("project_memory:") for k in session.state.readiness.holds)

    session.step(
        IoCompleted(
            op_id=cat_eff.op_id,
            value=pm.ProjectCatalogResult(
                project_name="demo",
                project_dir=str(project_dir),
                status="absent",
                sessions=(),
                error="",
            ),
        )
    )
    assert not session.state.readiness.is_ready

    session.step(
        IoCompleted(
            op_id=live_eff.op_id,
            value=DiscoverSessionsResult(status="ready", rows=(), observed_at=1.0),
        )
    )
    assert session.state.readiness.is_ready
