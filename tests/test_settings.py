from types import SimpleNamespace

import pytest
import yaml
from agent_utils import PipelineBuilder, Session
from agent_zoo.runtime.settings import Setting, SettingError, SettingsContext, SettingsRegistry

import project_memory as pm


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config" / "project_memory.yaml"
    monkeypatch.setenv(pm.CONFIG_PATH_ENV, str(path))
    return path


def registry(config=None):
    features = []
    pm.register_features(SimpleNamespace(add=features.append), session=None, config=config or {})
    with Session() as session:
        builder = PipelineBuilder(session)
        builder.add(next(feature for feature in features if feature.name == "project_memory_settings"))
        pipeline = builder.build(suppress_warnings=True)
    return SettingsRegistry([component for component in pipeline if isinstance(component, Setting)])


def request(reg, **kwargs):
    action = "set" if kwargs else "list"
    if kwargs:
        kwargs["binding"] = reg.binding
    return reg.handle({"type": "plugin_settings", "request_id": "test", "action": action, **kwargs},
                      SettingsContext(SimpleNamespace(state=SimpleNamespace())))


def test_defaults_metadata_and_typed_persistence(config_path):
    reg = registry()
    result = request(reg)
    assert result["ok"] and result["error"] is None
    assert {row["key"] for row in result["settings"]} == {
        "project_memory.enabled", "project_memory.project_sessions.enabled",
        "project_memory.live_sessions.enabled",
    }
    assert all(row["kind"] == "bool" and row["apply"] == "reload" and row["value"] is True
               and row["writable"] and row["error"] is None for row in result["settings"])
    assert not config_path.exists()
    for invalid in (1, "false", None):
        assert not request(reg, key="project_memory.enabled", value=invalid)["ok"]
        assert not config_path.exists()
    for row in result["settings"]:
        assert request(reg, key=row["key"], value=False)["ok"]
    assert all(row["value"] is False for row in request(registry())["settings"])
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert pm.ProjectMemorySystemPrompt.effective_section({}) == {
        "enabled": False, "project_sessions": {"enabled": False},
        "live_sessions": {"enabled": False},
    }
    features = []
    pm.register_features(SimpleNamespace(add=features.append), session=None, config={})
    assert [feature.name for feature in features] == ["project_memory_settings"]
    assert request(registry(), key="project_memory.enabled", value=True)["ok"]
    assert request(registry(), key="project_memory.project_sessions.enabled", value=True)["ok"]
    pm.register_features(SimpleNamespace(add=features.append), session=None, config={})
    assert features[-1].name == "project_memory"


def test_preserves_prompt_unknown_values_absent_defaults_and_permissions(config_path):
    config_path.parent.mkdir()
    config_path.write_text('''other: &shared
  enabled: true
  nested: [1, {text: "keep"}]
project_memory:
  system_prompt: |-
    Detailed non-regex guidance.
    Keep every line and punctuation: exactly.
  project_sessions: *shared
  unknown: {enabled: true, delay: 1.25}
''')
    config_path.chmod(0o640)
    original = yaml.safe_load(config_path.read_text())
    assert request(registry(), key="project_memory.project_sessions.enabled", value=False)["ok"]
    expected = yaml.safe_load(yaml.safe_dump(original))
    expected["project_memory"]["project_sessions"] = {**expected["other"], "enabled": False}
    assert yaml.safe_load(config_path.read_text()) == expected
    assert config_path.stat().st_mode & 0o777 == 0o640
    assert set(config_path.parent.iterdir()) == {config_path}


@pytest.mark.parametrize("raw", ["project_memory: [", "[]", "null", "", "project_memory: false", "\udcff"])
def test_invalid_file_is_not_overwritten(config_path, raw):
    config_path.parent.mkdir()
    original = raw.encode("utf-8", errors="surrogateescape")
    config_path.write_bytes(original)
    # Registration's pre-existing forgiving loader is separate from strict settings reads.
    reg = SettingsRegistry(pm._settings({}))
    result = request(reg, key="project_memory.enabled", value=False)
    assert not result["ok"]
    assert result["settings"][0]["error"] is not None
    assert config_path.read_bytes() == original
    with pytest.raises(SettingError):
        pm._write_setting("enabled", False)
    assert config_path.read_bytes() == original


def test_unreadable_and_failed_replace_are_safe(config_path, monkeypatch):
    config_path.mkdir(parents=True)
    reg = SettingsRegistry(pm._settings({}))
    assert not request(reg, key="project_memory.enabled", value=False)["ok"]
    config_path.rmdir()
    config_path.write_text("project_memory: {enabled: true}\n")
    original = config_path.read_bytes()

    def fail(*args):
        raise PermissionError("private path or detail")

    monkeypatch.setattr(pm.os, "replace", fail)
    result = request(reg, key="project_memory.enabled", value=False)
    assert not result["ok"] and "private" not in str(result)
    assert config_path.read_bytes() == original
    assert set(config_path.parent.iterdir()) == {config_path}


def test_saved_defaults_are_editable_without_changing_launch_override(config_path):
    config_path.parent.mkdir()
    config_path.write_text("project_memory: {enabled: false, live_sessions: false}\n")
    config = {"project_memory": {"enabled": True, "live_sessions": {"other": 1}}}
    reg = registry(config)
    rows = {row["key"]: row for row in request(reg)["settings"]}
    for key in ("project_memory.enabled", "project_memory.live_sessions.enabled"):
        assert rows[key]["value"] is False and rows[key]["writable"]
        assert "override" in rows[key]["help"]
        assert request(reg, key=key, value=True)["ok"]
        assert request(reg, key=key, value=False)["ok"]
    assert config == {"project_memory": {"enabled": True, "live_sessions": {"other": 1}}}
    effective = pm.ProjectMemorySystemPrompt.effective_section(config)
    assert effective["enabled"] is True
    assert pm._subfeature_enabled(effective, "live_sessions") is True
    reloaded = {row["key"]: row for row in request(registry())["settings"]}
    assert reloaded["project_memory.enabled"]["value"] is False
    assert reloaded["project_memory.live_sessions.enabled"]["value"] is False
    assert rows["project_memory.project_sessions.enabled"]["writable"]


@pytest.mark.parametrize("value", [False, "off", {"enabled": False}])
def test_legacy_subfeature_shapes(config_path, value):
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump({"project_memory": {"project_sessions": value}}))
    reg = registry()
    assert request(reg)["settings"][1]["value"] is False
    assert request(reg, key="project_memory.project_sessions.enabled", value=True)["ok"]
    assert pm._subfeature_enabled(pm.ProjectMemorySystemPrompt.effective_section({}), "project_sessions")


def test_state_home_path_and_symlink_are_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv(pm.CONFIG_PATH_ENV, raising=False)
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path))
    path = tmp_path / "plugin-configs" / pm.PLUGIN_NAME / "config" / pm.CONFIG_FILENAME
    assert not path.exists()
    assert request(registry(), key="project_memory.enabled", value=False)["ok"]
    assert yaml.safe_load(path.read_text()) == {"project_memory": {"enabled": False}}
    link = tmp_path / "linked.yaml"
    link.symlink_to(path)
    monkeypatch.setenv(pm.CONFIG_PATH_ENV, str(link))
    assert request(registry(), key="project_memory.enabled", value=True)["ok"]
    assert link.is_symlink()
    assert yaml.safe_load(path.read_text()) == {"project_memory": {"enabled": True}}
