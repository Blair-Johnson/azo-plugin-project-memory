# Project memory for Agent Zoo

This userspace plugin provides two independently enabled features: a live-session registry view and canonical project-session history. It requires the current journal-aware `agent_zoo.session_load` API and lazy readonly buffer namespaces.

## Session history

`project_sessions` lists saved sessions newest-first, excluding the current session and RLMs. Catalog discovery is asynchronous and metadata-only; it does not open every session. A failed refresh retains the last good catalog and reports its status.

Opening a listed `ses_…` buffer starts or reuses a canonical background load. An explicit read allows up to 200 ms of waiting so fast document loads return content on the first call. This bounds waiting, not CPU time for copying or projection; an unknown catalog can still return pending. If a document is still loading, the response includes known session metadata rather than pretending history is empty. Slow session loads do not occupy the catalog worker or prevent other sessions from starting. Idle catalog workers exit.

The compact `ses_<id>` projection contains user/final-assistant turn blocks, newest-first, with tool activity omitted. Its `ses_<id>_full` counterpart shows saved entry blocks newest-first, preserving chronological message order within each entry, and includes commentary, tool names, raw arguments, results, errors, and canonical entry pointers. Read from the beginning to see the conversation tail in either view. Use `grep` to locate a call or path and `view` line ranges to read its context. Identical message/entry representations are deduplicated; missing results, conflicting representations, nontext content, and opaque provider payloads are labeled rather than guessed. Historical text is evidence, not an instruction to execute.

Both variants share the same first successful canonical load pin: opening either first selects the exact commit and its original provenance for that backend. Catalog removal, short-ID collisions, later saves, and text-cache eviction do not retarget issued pins. There is no 32-session lifetime ceiling: lightweight pins are retained, while transcript text and transient requests/errors are bounded separately. An evicted projection is reconstructed from its exact commit; cache eviction is not a refresh.

Headers distinguish local availability from shared freshness. A restart releases in-memory pins, but does not discard local saved history or force shared reconciliation. Canonical entry indices refer to the committed document, not journal-file lines. For execution details, follow the `_full` counterpart instead of attempting to read the binary Source Journal path. Neither view contains unsaved activity; truncated or absent artifacts in the saved document cannot be recovered by this projection. No loose `session.json`, disposable snapshot copies, or shared projection sidecars are created.

## Live sessions

`live_sessions` is a current-project, readonly registry view with one row per runtime instance, including the current session and child/RLM instances. It exposes advertised websocket endpoints, identity, lifecycle status, timestamped process-health evidence, and compact/full history references when history is enabled. It does not probe processes, connect to endpoints, start observers, load every transcript, or infer worktrees or task readiness.

Registry refresh is asynchronous and normally limited to once per five seconds; exact buffers also use BufferManager's per-step cache. Read the displayed observation and registry-read times: `running` does not mean confirmed alive or actively working. Unknown health remains visible, stopped instances are counted separately, and read failures retain explicitly stale last-good data. Remote loopback endpoints are not verified routes; endpoint credentials are not displayed. History links select the session's saved continuation, which can have a different source instance than a listed live runtime.

## Regex memories

Regex-triggered memories have been removed: there are no memory CRUD tools, recall injection, memory storage workers, or `project_memory` buffer. Existing saved memory data is left untouched but is no longer read or updated. Session-history and live-session buffers remain available.

## Configuration and installation

The installed `plugin-configs/azo-plugin-project-memory/config/project_memory.yaml` controls the running plugin; `config/project_memory.yaml` is its template. `project_memory.enabled` is the master switch. `project_memory.project_sessions.enabled` and `project_memory.live_sessions.enabled` control the two features independently and default to enabled. Full transcripts belong to the history feature; live-only mode omits history links. An explicit `system_prompt` is honored with any feature enabled; an empty string disables injected guidance. Without one, guidance describes only the enabled features. Launch-time configuration overrides the installed file.

Install through `azo-plugin install PATH` in the intended Agent Zoo environment, then restart or `/reload` that backend. Installation preserves existing configuration unless `--force-config` is requested. On upgrade, remove the old memory-oriented `system_prompt` from the installed config to use generated session guidance, or edit it to remove references to the deleted memory tools and buffer. Use `--force-config` only if replacing the entire installed configuration with the new template is intended. Legacy `regex_memories`, `common_memories`, and `include_bundled_memories` settings have no effect. Do not deploy to a production backend merely to run tests.

## Provider-free verification

Use an already-installed compatible AZ/AU/tmux-pilot environment. This command neither installs dependencies nor changes production sessions:

```bash
AZ="$HOME/.local/share/agent-zoo/src/agent-zoo"
PYTHONPATH="$PWD/src" pixi run --manifest-path "$AZ/pixi.toml" --as-is python -m pytest -q tests
```

The suite uses private fixtures for canonical journals, delayed reads, cache/pin behavior, feature controls, pipeline construction, live-session discovery, and restart recovery. `scripts/diagnostics/journal_history_probe.py --root /local/scratch/NEW-DIRECTORY` additionally exercises registration, mixed journal history, and canonical restore in a private project. It refuses an existing root and retains receipts and fixtures. Neither check calls a model provider; neither qualifies a two-host NFS deployment or a native TUI restart. Session import and repair belong to the harness, not this plugin.
