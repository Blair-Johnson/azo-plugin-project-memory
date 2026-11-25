"""Pure Sans-I/O /config settings tests via SettingsDispatcher + PluginStoreDriver."""
from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest
import yaml
from agent_utils.core.events import IoFailed
from agent_utils.core.io import SubmitOp
from agent_utils.core.pipeline import PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.features.core import core_feature
from agent_zoo.durability import PluginStoreDriver, StoreUpdate
from agent_zoo.events import ReplyFrontendRequest, SettingsRequest
from agent_zoo.features.settings import Setting, settings_feature
from agent_zoo.testing.sans_io_guard import assert_pipeline_sans_io

import project_memory as pm


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path))
    monkeypatch.delenv(pm.CONFIG_PATH_ENV, raising=False)
    return tmp_path / "plugin-configs" / pm.PLUGIN_NAME / "config" / pm.CONFIG_FILENAME


def build_settings_session(config=None):
    session = Session()
    builder = PipelineBuilder().add(core_feature()).add(settings_feature())
    pm.register_features(builder, session=session, config=config or {})
    session.pipeline = builder.build(suppress_warnings=True)
    session.ensure_initialized()
    assert_pipeline_sans_io(session.pipeline)
    return session


def dispatch(session, *, action="list", fail_persist=None, **kwargs):
    ev = SettingsRequest(request_id="test", action=action, **kwargs)
    effects = session.step(ev)
    reply = None

    def _collect(effs):
        nonlocal reply
        next_events = []
        for eff in effs:
            if not isinstance(eff, SubmitOp):
                continue
            req = eff.request
            if isinstance(req, ReplyFrontendRequest):
                reply = req.payload
            elif isinstance(req, StoreUpdate):
                if fail_persist is not None:
                    next_events.append(IoFailed(op_id=eff.op_id, error=fail_persist))
                else:
                    drv = PluginStoreDriver()
                    inbox: queue.Queue = queue.Queue()
                    drv.start(inbox.put)
                    try:
                        drv.submit(eff.op_id, req)
                        io_ev = inbox.get(timeout=5.0)
                    finally:
                        drv.close()
                    next_events.append(io_ev)
        return next_events

    pending = _collect(effects)
    while pending:
        ev2 = pending.pop(0)
        pending.extend(_collect(session.step(ev2)))
    assert reply is not None
    return reply


def pm_rows(result):
    return [row for row in result["settings"] if row["key"].startswith("project_memory.")]


def test_defaults_metadata_and_typed_persistence(config_path):
    session = build_settings_session()
    result = dispatch(session)
    rows = pm_rows(result)
    assert result["ok"] and result["error"] is None
    assert {row["key"] for row in rows} == {
        "project_memory.enabled",
        "project_memory.project_sessions.enabled",
        "project_memory.live_sessions.enabled",
    }
    assert all(
        row["kind"] == "bool"
        and row["apply"] == "reload"
        and row["value"] is True
        and row["writable"]
        and row["error"] is None
        for row in rows
    )
    assert not config_path.exists()
    for invalid in (1, "false", None):
        assert not dispatch(
            session, action="set", binding=result["binding"], key="project_memory.enabled", value=invalid
        )["ok"]
        assert not config_path.exists()
    binding = result["binding"]
    for row in rows:
        res = dispatch(session, action="set", binding=binding, key=row["key"], value=False)
        assert res["ok"]
        binding = res["binding"]

    fresh = build_settings_session()
    fresh_rows = pm_rows(dispatch(fresh))
    assert all(row["value"] is False for row in fresh_rows)
    assert pm.ProjectMemorySystemPrompt.effective_section({}) == {
        "enabled": False,
        "project_sessions": {"enabled": False},
        "live_sessions": {"enabled": False},
    }
    features = []
    pm.register_features(SimpleNamespace(add=features.append), session=None, config={})
    assert [feature.name for feature in features] == ["project_memory_settings"]

    fresh2 = build_settings_session()
    listing2 = dispatch(fresh2)
    r1 = dispatch(fresh2, action="set", binding=listing2["binding"], key="project_memory.enabled", value=True)
    assert r1["ok"]
    r2 = dispatch(
        fresh2, action="set", binding=r1["binding"], key="project_memory.project_sessions.enabled", value=True
    )
    assert r2["ok"]
    pm.register_features(SimpleNamespace(add=features.append), session=None, config={})
    assert features[-1].name == "project_memory"


def test_preserves_prompt_unknown_values_and_absent_defaults(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '''other: &shared
  enabled: true
  nested: [1, {text: "keep"}]
project_memory:
  system_prompt: |-
    Detailed non-regex guidance.
    Keep every line and punctuation: exactly.
  project_sessions: *shared
  unknown: {enabled: true, delay: 1.25}
''',
        encoding="utf-8",
    )
    original = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    session = build_settings_session()
    listing = dispatch(session)
    assert dispatch(
        session,
        action="set",
        binding=listing["binding"],
        key="project_memory.project_sessions.enabled",
        value=False,
    )["ok"]
    expected = yaml.safe_load(yaml.safe_dump(original))
    expected["project_memory"]["project_sessions"] = {**expected["other"], "enabled": False}
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize("raw", ["project_memory: [", "[]", "null", "", "project_memory: false", "\udcff"])
def test_invalid_file_is_not_overwritten(config_path, raw):
    config_path.parent.mkdir(parents=True)
    original = raw.encode("utf-8", errors="surrogateescape")
    config_path.write_bytes(original)
    session = build_settings_session()
    listing = dispatch(session)
    result = dispatch(
        session,
        action="set",
        binding=listing["binding"],
        key="project_memory.enabled",
        value=False,
    )
    assert not result["ok"]
    assert pm_rows(result)[0]["error"] is not None
    assert config_path.read_bytes() == original


def test_unreadable_and_failed_persist_rollback_are_safe(config_path):
    config_path.mkdir(parents=True)
    session = build_settings_session()
    listing = dispatch(session)
    assert not dispatch(
        session,
        action="set",
        binding=listing["binding"],
        key="project_memory.enabled",
        value=False,
    )["ok"]
    config_path.rmdir()
    config_path.write_text("project_memory: {enabled: true}\n", encoding="utf-8")
    original = config_path.read_bytes()

    session2 = build_settings_session()
    listing2 = dispatch(session2)
    result = dispatch(
        session2,
        action="set",
        binding=listing2["binding"],
        key="project_memory.enabled",
        value=False,
        fail_persist=PermissionError("private path or detail"),
    )
    assert not result["ok"] and "private" not in str(result)
    assert config_path.read_bytes() == original
    # Rolled back in memory to True
    enabled_row = next(r for r in pm_rows(result) if r["key"] == "project_memory.enabled")
    assert enabled_row["value"] is True


def test_saved_defaults_are_editable_without_changing_launch_override(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text("project_memory: {enabled: false, live_sessions: false}\n", encoding="utf-8")
    config = {"project_memory": {"enabled": True, "live_sessions": {"other": 1}}}
    session = build_settings_session(config)
    listing = dispatch(session)
    rows = {row["key"]: row for row in pm_rows(listing)}
    binding = listing["binding"]
    for key in ("project_memory.enabled", "project_memory.live_sessions.enabled"):
        assert rows[key]["value"] is False and rows[key]["writable"]
        assert "override" in rows[key]["help"]
        r1 = dispatch(session, action="set", binding=binding, key=key, value=True)
        assert r1["ok"]
        binding = r1["binding"]
        r2 = dispatch(session, action="set", binding=binding, key=key, value=False)
        assert r2["ok"]
        binding = r2["binding"]
    assert config == {"project_memory": {"enabled": True, "live_sessions": {"other": 1}}}
    effective = pm.ProjectMemorySystemPrompt.effective_section(config)
    assert effective["enabled"] is True
    assert pm._subfeature_enabled(effective, "live_sessions") is True
    reloaded = {row["key"]: row for row in pm_rows(dispatch(build_settings_session()))}
    assert reloaded["project_memory.enabled"]["value"] is False
    assert reloaded["project_memory.live_sessions.enabled"]["value"] is False
    assert rows["project_memory.project_sessions.enabled"]["writable"]


@pytest.mark.parametrize("value", [False, "off", {"enabled": False}])
def test_legacy_subfeature_shapes(config_path, value):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({"project_memory": {"project_sessions": value}}), encoding="utf-8")
    session = build_settings_session()
    listing = dispatch(session)
    rows = {row["key"]: row for row in pm_rows(listing)}
    assert rows["project_memory.project_sessions.enabled"]["value"] is False
    assert dispatch(
        session,
        action="set",
        binding=listing["binding"],
        key="project_memory.project_sessions.enabled",
        value=True,
    )["ok"]
    assert pm._subfeature_enabled(
        pm.ProjectMemorySystemPrompt.effective_section({}), "project_sessions"
    )


def test_state_home_ignores_install_root_and_xdg_data_home(tmp_path, monkeypatch):
    monkeypatch.delenv(pm.CONFIG_PATH_ENV, raising=False)
    monkeypatch.delenv("AGENT_ZOO_HOME", raising=False)
    monkeypatch.delenv("AGENT_ZOO_STATE_ROOT", raising=False)
    monkeypatch.setenv("AGENT_ZOO_INSTALL_ROOT", str(tmp_path / "install"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    # Must resolve via canonical resolve_state_root(), NOT install_root or XDG_DATA_HOME.
    assert pm.ProjectMemorySystemPrompt._state_root() != (tmp_path / "install")
    assert pm.ProjectMemorySystemPrompt._state_root() != (tmp_path / "xdg" / "agent-zoo")

    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path))
    path = tmp_path / "plugin-configs" / pm.PLUGIN_NAME / "config" / pm.CONFIG_FILENAME
    assert pm.ProjectMemorySystemPrompt._config_path() == path
    assert not path.exists()
    session = build_settings_session()
    listing = dispatch(session)
    assert dispatch(
        session,
        action="set",
        binding=listing["binding"],
        key="project_memory.enabled",
        value=False,
    )["ok"]
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"project_memory": {"enabled": False}}
