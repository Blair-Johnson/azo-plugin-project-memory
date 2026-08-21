# azo-plugin-project-memory

An Agent Zoo userspace plugin for project-scoped, agent-authored regex memories.

The plugin stores memories in the current Agent Zoo project's shared SQLite database. Each memory has a short `mem_xxxxxx` identifier, content to recall, and a regular-expression trigger. On every pipeline pass, after tool results have been consolidated, the plugin tests newly appended transcript messages against current project memories. A matching memory is delivered as a standard interrupt only when its content is not already present in the active context after the newest harness handoff or provider compaction boundary.

The feature provides `record_memory`, `update_memory`, `suppress_memory`, and `delete_memory`. Suppression is session-local and measured in pipeline turns; memory records themselves propagate between running sessions because the database is queried on every pass.

The plugin registers three readonly buffer surfaces. `project_memory` shows the project’s regex memories. `project_sessions` lists saved previous sessions newest-first, excluding the active session and every RLM session. Each listed `ses_<prefix>` ID lazily renders a compact reverse-chronological projection: complete turn blocks appear newest-first while each block preserves the user message followed by its final assistant response. Message delimiters retain source entry indexes and line ranges into the authoritative `session.json` file.

Compact session projections are cached as disposable atomic sidecars under the project’s `plugin-data/project-memory/compact` directory. The source transcript remains authoritative; its content digest and indexed session metadata invalidate stale sidecars. Rendering the session index never parses transcripts or creates cache files.

Session-history buffers require Agent Utils support for replaceable lazy special-buffer namespaces. On an older host, the plugin logs a warning and keeps the core memory tools and `project_memory` buffer active. Set `project_memory.project_sessions.enabled=false` to disable only the session-history surfaces.

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
