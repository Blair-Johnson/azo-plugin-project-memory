# azo-plugin-project-memory

An Agent Zoo userspace plugin for project-scoped, agent-authored regex memories.

The plugin stores memories in the current Agent Zoo project's shared SQLite database. Each memory has a short `mem_xxxxxx` identifier, content to recall, and a regular-expression trigger. On every pipeline pass, after tool results have been consolidated, the plugin tests newly appended transcript messages against current project memories. A matching memory is delivered as a standard interrupt only when its content is not already present in the active context after the newest harness handoff or provider compaction boundary.

The feature provides `record_memory`, `update_memory`, `suppress_memory`, and `delete_memory`. Suppression is session-local and measured in pipeline turns; memory records themselves propagate between running sessions because the database is queried on every pass.
The feature also registers a live readonly special buffer named `project_memory`. Read it with the normal buffer tools to see every project memory’s `mem_xxxxxx` ID, trigger regex, and content; it queries the shared project database whenever the buffer is materialized.

Install with:

```bash
pixi run azo-plugin install .
```

Installation creates an editable config at:

```text
~/.local/share/agent-zoo/plugin-configs/azo-plugin-project-memory/config/project_memory.yaml
```

The `project_memory.system_prompt` value controls the guidance appended to the model-visible system prompt. Launch-time `--config-set project_memory.system_prompt=...` values override the installed file. Set `project_memory.enabled=false` to disable registration, then run `/reload` in an existing TUI session.

Run tests with:

```bash
pixi run --environment test test
```
