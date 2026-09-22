# Project memory for Agent Zoo

This userspace plugin provides two independently enabled features: canonical project-session history and regex-triggered durable memories. It requires the current journal-aware `agent_zoo.session_load` API and lazy readonly buffer namespaces.

## Session history

`project_sessions` lists saved sessions newest-first, excluding the current session and RLMs. Catalog discovery is asynchronous and metadata-only; it does not open every session. A failed refresh retains the last good catalog and reports its status.

Opening a listed `ses_…` buffer starts or reuses a canonical background load. An explicit read allows up to 200 ms of waiting so fast document loads return content on the first call. This bounds waiting, not CPU time for copying or projection; an unknown catalog can still return pending. If a document is still loading, the response includes known session metadata rather than pretending history is empty. Slow session loads do not occupy the catalog worker or prevent other sessions from starting. Idle catalog workers exit.

Projections contain user/final-assistant turn blocks, newest-first, with tool activity omitted. The first successful canonical load pins the exact commit and its original provenance for that backend. Catalog removal, short-ID collisions, later saves, and text-cache eviction do not retarget issued pins. There is no 32-session lifetime ceiling: lightweight pins are retained, while transcript text and transient requests/errors are bounded separately. An evicted projection is reconstructed from its exact commit; cache eviction is not a refresh.

Headers distinguish local availability from shared freshness. A restart releases in-memory pins, but does not discard local saved history or force shared reconciliation. Canonical entry indices refer to the committed document, not journal-file lines. For omitted details, use `load_session` with the header's session and commit identity. No loose `session.json`, disposable snapshot copies, or shared projection sidecars are created.

## Regex memories

`record_memory`, `update_memory`, `suppress_memory`, and `delete_memory` manage project-specific guidance with selective regex triggers. Configured memories are readonly; suppression is session-local. Explicit writes persist locally before shared publication, with revision checks against concurrent edits. Inspect `project_memory` for local/shared status and conflicts. Shared failures never erase the last good memory snapshot.

Publication acknowledgments stay in the existing local operation record until cleanup completes. Recovery retries acknowledged cleanup rather than overwriting a newer remote edit. Local-only changes are unavailable on other hosts until publication succeeds; the local spool must be on reliable local storage.

Recall is best-effort: individual searches have a 5 ms budget, a pass 50 ms, event text 64 KiB, and pending recalls eight memories. Backtests report incomplete evaluation. Refine expensive or noisy triggers rather than relying on exhaustive matching.

## Configuration and installation

The installed `plugin-configs/azo-plugin-project-memory/config/project_memory.yaml` controls the running plugin; `config/project_memory.yaml` is its template. `project_memory.enabled` is the master switch. `project_memory.regex_memories.enabled` and `project_memory.project_sessions.enabled` control the two features independently. An explicit `system_prompt` is honored with either feature enabled; an empty string disables injected guidance. Without one, guidance describes only the enabled features. Launch-time configuration overrides the installed file.

Install through `azo-plugin install PATH` in the intended Agent Zoo environment, then restart or `/reload` that backend. Installation preserves existing configuration unless `--force-config` is requested. Review custom prompts when disabling a feature: explicit text is not rewritten to remove references to disabled tools. Do not deploy to a production backend merely to run tests.

## Provider-free verification

Use an already-installed compatible AZ/AU/tmux-pilot environment. This command neither installs dependencies nor changes production sessions:

```bash
AZ="$HOME/.local/share/agent-zoo/src/agent-zoo"
PYTHONPATH="$PWD/src" pixi run --manifest-path "$AZ/pixi.toml" --as-is python -m pytest -q tests
```

The suite uses private fixtures for canonical journals, delayed reads, cache/pin behavior, feature controls, pipeline construction, publication conflicts, and restart recovery. `scripts/diagnostics/journal_history_probe.py --root /local/scratch/NEW-DIRECTORY` additionally exercises registration, mixed journal history, canonical restore, and storage outages in a private project. It refuses an existing root and retains receipts and fixtures. Neither check calls a model provider; neither qualifies a two-host NFS deployment or a native TUI restart.

The optional `scripts/live_dev_probe.py` is a separate real-model test requiring credentials and an explicitly chosen development root. Historical acceptance numbers are not a substitute for running the current suite.

Old SQLite memories can be imported offline with `scripts/import_sqlite_memories.py`; stop old writers first. Import is explicit and conflict-preserving, never a runtime fallback. Session import and repair belong to the harness, not this plugin.
