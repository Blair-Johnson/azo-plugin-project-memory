# Project memory for Agent Zoo

This branch targets the current filesystem lifecycle harness. It keeps the `record_memory`, `update_memory`, `suppress_memory`, and `delete_memory` tools, regex-triggered recall, and the `project_memory`, `project_sessions`, and `ses_…` readonly buffers. It does not use the old AgentDB SQLite connection or require an orchestrator.

## Behavior

Project memories are independent records. Explicit writes are saved locally before being queued for publication to shared storage. Tool receipts distinguish local persistence from shared publication. Concurrent edits are checked against their expected revisions rather than silently overwriting another writer. Configured memories remain readonly; suppression remains session-local.

Recall uses an in-memory last-good memory snapshot while a background worker refreshes it. Shared storage errors are not interpreted as an empty memory collection. Routine synchronization is quiet; inspect `project_memory` for status. Local-only changes are not available on other hosts until publication succeeds. The local spool must be on a reliable local filesystem if it is to remain usable during shared-filesystem outages.

Prior-session discovery reads the project catalog without probing every checkpoint. Opening a `ses_…` buffer loads a saved immutable revision in the background and pins its projection. Pending or unavailable reads are reported as such. Projection caches are local, not shared writes. Projections contain complete user/final-assistant turn blocks in reverse chronological order, omit tool activity, and retain source revision and entry references. Current and RLM sessions are excluded from the index.

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

## Verification

Run tests using the environment that contains the matching Agent Zoo, Agent Utils, and tmux-pilot revisions, with this repository's `src` on `PYTHONPATH`. The plugin's own test environment does not silently select sibling production checkouts.

`scripts/live_dev_probe.py --dev-root PATH` uses the installed launcher and a real Luna model to exercise memory recording, cross-session recall, immutable history, editing, suppression, deletion, and restart. It creates a uniquely named test project in the dev root and prints retained backend/tool evidence. Model credentials are required. Its observation deadlines belong to the diagnostic, not the plugin's operation lifecycle.

## Migration and limitations

Initial import of the old `azo_project_memories` SQLite table is an explicit offline operation, not a runtime fallback. Do not point the new plugin at old mutable `session.json` files: import those sessions through the harness's initial checkpoint importer first.

Validation evidence and the final importer invocation are recorded after integration. Local delayed/unavailable-storage tests do not qualify a particular two-host NFS deployment; that still requires testing on the intended mount.
