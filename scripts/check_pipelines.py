"""Validate installed plugin discovery without guessing sibling source paths."""
import json
import os
from pathlib import Path
import sys

from agent_utils import Session
from agent_zoo import pipelines
from agent_zoo.modes import default_modes, rlm_modes


for name, builder, modes in (
    ("default", pipelines.build_default_pipeline, default_modes),
    ("rlm", pipelines.build_readonly_rlm_pipeline, rlm_modes),
):
    with Session(system_prompt="plugin build check") as session:
        pipeline = builder(
            session, config={}, skill_paths=[], scheduler_type="local",
            max_idle=0, mode_defs=modes(), initial_mode="execution",
            terminal_backend="headless", tmux_tools_enabled=False,
        )
    components = [type(component).__name__ for component in pipeline]
    tools = {getattr(component, "name", "") for component in pipeline}
    assert "ProjectMemoryStore" in components, components
    assert "ProjectSessionBuffers" in components, components
    assert {"record_memory", "update_memory", "suppress_memory", "delete_memory"} <= tools
    print(json.dumps({"pipeline": name, "python": sys.executable,
                      "home": os.environ.get("AGENT_ZOO_HOME"),
                      "source": str(Path(pipelines.__file__).resolve()),
                      "project_memory": "present", "tools": sorted(tools & {
                          "record_memory", "update_memory", "suppress_memory", "delete_memory"})}))
