# Project memory for Agent Zoo

This branch targets the canonical journal filesystem lifecycle harness (`agent_zoo.session_load` plus Agent Utils canonical documents). It keeps the `record_memory`, `update_memory`, `suppress_memory`, and `delete_memory` tools, regex-triggered recall, and the `project_memory`, `project_sessions`, and `ses_…` readonly buffers. It does not use the old AgentDB SQLite connection or require an orchestrator.

## Behavior

Project memories are independent records. Explicit writes are saved locally before being queued for publication to shared storage. Tool receipts distinguish local persistence from shared publication. Concurrent edits are checked against their expected revisions rather than silently overwriting another writer. Configured memories remain readonly; suppression remains session-local.

Recall uses an in-memory last-good memory snapshot while a background worker refreshes it. Shared storage errors are not interpreted as an empty memory collection. Routine synchronization is quiet; inspect `project_memory` for status. Local-only changes are not available on other hosts until publication succeeds. The local spool must be on a reliable local filesystem if it is to remain usable during shared-filesystem outages.

Prior-session discovery reads the project catalog without probing every session repository. Opening a `ses_…` buffer uses `agent_zoo.session_load.load_session` in the background, projects `LoadedSession.state.document` directly, and pins the exact `state.commit_id`. Journal-only sessions and mixed legacy/journal repositories use the same canonical selection rules as the harness; genuine ambiguity is reported, never resolved by timestamp or silently retried through the historical checkpoint loader. Pending or unavailable reads are reported as such. Projection caches are in memory, not shared writes or temporary JSON snapshots. Projections contain complete user/final-assistant turn blocks in reverse chronological order, omit tool activity, and retain exact source commit, instance, and entry references. Current and RLM sessions are excluded from the index.

Regex evaluation has a computation budget, and automatic recall is best-effort: individual searches are bounded to 5 ms, a recall pass to 50 ms, each event to 64 KiB, and pending recall to eight memories. Expensive/noisy triggers can therefore miss matches and should be refined. Backtests report incomplete evaluation. Transcript-tail tracking prevents bounded event-ID retention from replaying old history in long sessions; existing compaction-aware context deduplication is retained.

## Development install

Use the dev-root wrapper and explicitly select its state root. Inherited production environment variables otherwise override the intended location. Substitute your own checkout and install paths:

```bash
DEV="$HOME/.local/share/agent-zoo-dev"
PLUGIN="$HOME/Documents/GitHub/azo-plugin-project-memory-worktrees/dev-filesystem"
env -u PYTHONPATH AGENT_ZOO_STATE_ROOT="$DEV" AGENT_ZOO_HOME="$DEV" \
  "$DEV/bin/azo-plugin" install "$PLUGIN" --force
```

The installer declares dependencies through Pixi and preserves existing installed configuration unless `--force-config` is requested. Restart or `/reload` the dev backend after installation. The installed YAML lives under `plugin-configs/azo-plugin-project-memory/config/project_memory.yaml`; repository YAML is its template.

## Historical verification

The existing suite and live-model probe below predate canonical journal history. They are not the September 17 stabilization gate; use the explicitly authorized private provider-free workflow below with matching Agent Zoo, Agent Utils, and tmux-pilot sources. No tests, providers, builds, or installs were run for this reader adaptation.

`scripts/live_dev_probe.py --dev-root PATH` uses the installed launcher and a real Luna model to exercise memory recording, cross-session recall, immutable history, editing, suppression, deletion, and restart. It injects shared plugin-storage unavailability, verifies local cached reads and deletion across restart, then checks the actual shared tombstone after recovery. It creates a uniquely named test project in the dev root and prints retained backend/tool evidence. Model credentials are required. Its observation deadlines belong to the diagnostic, not the plugin's operation lifecycle.

Historical, pre-journal-cutover evidence only: on September 10, 2026, the combined plugin suite passed 46 tests (`rx r16`), both installed pipeline builds passed (`r18`), and the installed real-model acceptance passed in 87 seconds (`r17`). The live run used project `memory-live-35f4446e` and exercised two sessions through four backend lifetimes. The earlier old-SQLite plugin failed its actual `record_memory` call against the new harness (`r2`), establishing the integration failure this migration fixes.

## Canonical history diagnostic

The September 17, 2026 private provider-free component/filesystem workflow passed for the canonical reader, including mixed history, stable commit pins, independent feature controls and memory CRUD across a storage outage and restore. Its temporary bulk was removed. This does not qualify native backend activation or NFS; the earlier September 10 acceptance does not validate journal history. `scripts/diagnostics/journal_history_probe.py` is an opt-in provider-free workflow for a compatible existing environment. It creates a small private project/home/spool, uses real plugin registration, journal persistence, catalog and buffer APIs, and retains JSON receipts on stdout plus the private files. It refuses an existing `--root`:

```bash
# Select already-installed compatible AZ/AU/TP sources; do not install an environment.
# $PYTHON is that environment's Python. Include this checkout's src on PYTHONPATH.
PYTHONPATH="$PLUGIN/src:$PYTHONPATH" "$PYTHON" \
  "$PLUGIN/scripts/diagnostics/journal_history_probe.py" \
  --root /local/scratch/new-project-memory-probe
```

The workflow checks journal-only and mixed-history newest turns, exact old-commit restore, unchanged `ses_…` content after later saves, fresh-reader advancement, independent regex/history/master controls, regex recall and suppression, memory CRUD, last-good cache and locally queued edits across canonical restore during a private storage outage, and shared deletion after recovery. The mixed fixture deliberately uses the historical immutable publisher to create its legacy ancestor; that is fixture preparation, not a runtime fallback. It does not load the managed plugin, call a provider, create an environment, or change production data. It is component/filesystem acceptance, **not** a full backend/TUI restart, divergent-branch repair, multi-host, or NFS qualification. Run it only at the planned private integration gate; remove its explicitly chosen private root after inspecting the evidence.

History headers show source location, repository, journal (when applicable), source format, instance, and exact commit. Canonical entry indices are not journal-file line numbers; no loose `session.json` is promised. For omitted detail, use `load_session` with the header's commit/session identity and inspect its canonical document. The canonical loader's `freshness` and `warning` fields are shown with the pinned source receipt. Local-first success is not proof of shared freshness. A restart releases pins but does not discard local saved history, force shared refresh, or resolve divergent continuations.

## Migration and limitations

Activation is separate from this source change. Do not replace a running old SQLite plugin blindly: stop its writers, preserve its configuration and data, validate the compatible core/plugin sources privately, then coordinate a fresh backend activation. Keep `project_memory.regex_memories` and `project_memory.project_sessions` independent. Existing custom prompts referring to physical Source Transcript paths should be reviewed for canonical commit/entry wording; configuration is not automatically overwritten.

Initial import of the old `azo_project_memories` SQLite table is an explicit offline operation, not a runtime fallback. Do not point the new plugin at old mutable `session.json` files: import those sessions through the harness's initial checkpoint importer first.

With the old writers stopped, import the project's old database into the plugin's dedicated store (not the harness store). IDs and timestamps are retained, reruns skip identical memories, and conflicting destination records are reported rather than overwritten:

```bash
pixi run --manifest-path "$DEV/src/agent-zoo/pixi.toml" python \
  "$PLUGIN/scripts/import_sqlite_memories.py" /path/to/old-project.sqlite \
  "$DEV/projects/PROJECT/plugin-data/project-memory" --dry-run --json
# Inspect the report, then repeat without --dry-run to import.
```

The SQLite source is opened read-only. No production memories are imported automatically. Local history projections are bounded to 32 pinned buffers per backend; an exhausted pin budget reports capacity instead of silently changing an existing buffer's meaning. A backend restart releases these in-memory pins.

Local delayed/unavailable-storage tests do not qualify a particular two-host NFS deployment; that still requires testing on the intended mount.

The live backend processes emitted Python multiprocessing semaphore-cleanup warnings at shutdown. This run does not establish their cause or resolve that harness cleanup issue; no plugin tool error or failed acceptance step accompanied them.
