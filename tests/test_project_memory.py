from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_utils import Entry, Session, State
from agent_zoo import pipelines
from agent_zoo.modes import default_modes, rlm_modes

import project_memory as pm


def test_system_prompt_render_transform_is_idempotent_and_model_only():
    state = State(token_budget=10_000)
    state.entries = [Entry(
        messages=[{"role": "system", "content": "Base prompt"}], index=0, step=0,
    )]
    component = pm.ProjectMemorySystemPrompt("Consult session history deliberately.")
    target = state.entries[0]
    target.initialize_pending_render_channels(("model", "user"))

    component(state)
    component(state)
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
    pm.register_features(SimpleNamespace(add=features.append), session=None, config={})

    assert len(features) == 1
    assert features[0].name == "project_memory"
    assert {type(component) for component in features[0].components} == {
        pm.ProjectSessionBuffers,
        pm.ProjectLiveSessionBuffer,
        pm.ProjectMemorySystemPrompt,
    }


def _install_only_project_memory(builder, *, session, config, **_kwargs):
    pm.register_features(builder, session=session, config=config)


@pytest.mark.parametrize(
    ("builder", "modes"),
    [
        (pipelines.build_default_pipeline, default_modes),
        (pipelines.build_readonly_rlm_pipeline, rlm_modes),
    ],
)
def test_real_harness_pipelines_build_with_project_memory(builder, modes, monkeypatch):
    monkeypatch.setattr(pipelines, "apply_pipeline_plugins", _install_only_project_memory)
    monkeypatch.setattr(pm.ProjectMemorySystemPrompt, "_load_installed_config", lambda: {})

    with Session(system_prompt="test") as session:
        pipeline = builder(
            session,
            config={"project_memory": {"system_prompt": "Consult session history."}},
            skill_paths=[],
            scheduler_type="local",
            max_idle=0,
            mode_defs=modes(),
            initial_mode="execution",
            terminal_backend="headless",
            tmux_tools_enabled=False,
        )

    types = {type(component) for component in pipeline}
    assert pm.ProjectSessionBuffers in types
    assert pm.ProjectLiveSessionBuffer in types
    assert pm.ProjectMemorySystemPrompt in types

    def position(kind):
        return next(i for i, component in enumerate(pipeline) if isinstance(component, kind))

    assert position(pm.RegisterSpecialBuffers) < position(pm.ProjectSessionBuffers)
    assert position(pm.ProjectSessionBuffers) < position(pm.ProjectLiveSessionBuffer)
    assert position(pm.ProjectLiveSessionBuffer) < position(pm.ToolDispatchStart)
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
    pm.register_features(SimpleNamespace(add=features.append), session=None,
                         config={"project_memory": section})
    prompt = next(component for component in features[0].components
                  if isinstance(component, pm.ProjectMemorySystemPrompt))
    assert prompt.text == expected


def test_config_template_preserves_detailed_session_guidance(monkeypatch):
    from pathlib import Path

    template = Path(__file__).parents[1] / "config" / "project_memory.yaml"
    monkeypatch.setenv(pm.CONFIG_PATH_ENV, str(template))
    features = []
    pm.register_features(SimpleNamespace(add=features.append), session=None, config={})
    prompt = next(component.text for component in features[0].components
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
