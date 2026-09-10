"""Exercise the installed plugin through real model/tool/WebSocket sessions."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

from agent_zoo.wsctl import AzoWs


def emit(phase, **values):
    print(json.dumps({"phase": phase, "at": time.time(), **values}), flush=True)


def results(client, name):
    return [str(tool.get("result", ""))
            for item in client.observation().transcript_items()
            for tool in item.get("tools", []) if tool.get("name") == name]

async def turn(client, prompt, marker):
    await client.send_user(prompt + f" End your final answer with {marker}.")
    await client.wait_until(
        lambda obs: any(str(item.get("body", "")).strip().endswith(marker)
                        and item.get("role") == "assistant"
                        for item in obs.transcript_items()), timeout=150,
    )
    return "\n".join(results(client, "view"))


async def run(args):
    project = "memory-live-" + uuid.uuid4().hex[:8]
    clients = []
    with tempfile.TemporaryDirectory(prefix="azo-memory-live-") as directory:
        root = Path(directory)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("AGENT_ZOO_", "AZO_WIRE_", "AZO_TUI_")) and k != "PYTHONPATH"}
        env["AGENT_ZOO_LOCAL_STATE_ROOT"] = str(root / "local")
        env["PYTHONPATH"] = str(Path(__file__).parent.resolve() / "memory_fault_hook")
        env["AZO_MEMORY_TEST_OUTAGE"] = str(root / "memory-sharing-offline")
        env["AZO_MEMORY_TEST_PROJECT"] = project
        command = [str(args.dev_root / "bin/azo-start"), "--home", str(args.dev_root),
                   "--project", project, "--workdir", str(root), "--host", "127.0.0.1",
                   "--port", "0", "--model", "gpt-5.6-luna-max"]
        try:
            await exercise(args, command, env, clients, project)
        finally:
            for client in reversed(clients):
                emit("backend_evidence", session_id=client.session_id,
                     stderr=list(client.stderr), transcript=client.observation().transcript_items())
                await client.force_close()


async def connect(command, env, clients):
    client = AzoWs(command=command, env=env, modern=True)
    clients.append(client)
    await asyncio.wait_for(client.__aenter__(), 60)
    await client.wait_until(lambda obs: obs.state.get("core.transcript"), timeout=30)
    emit("connected", session_id=client.session_id)
    return client


# Live exercise stages are kept below the process lifecycle helpers.


async def exercise(args, command, env, clients, project):
    first = await connect(command, env, clients)
    text = await turn(first,
        'Call record_memory exactly once with content="The cobalt launch checklist requires two reviewers." '
        'and trigger="(?i)cobalt launch". Do not use other tools.', "MEMORY_RECORDED")
    match = re.search(r"Recorded\s+(mem_[a-zA-Z0-9_-]+)", "\n".join(results(first, "record_memory")))
    assert match, "No successful record_memory tool receipt"
    memory_id = match.group(1)
    await turn(first, 'Use view(buffer="project_memory") to inspect the saved memory. '
               'If loading is pending, read it again. Report its storage status.', "MEMORY_INSPECTED")
    first_id = first.session_id
    await first.slash("/session save")
    await first.wait_until(lambda obs: (obs.state.get("core.status", {}).get("persistence") or {}).get("shared_saved"), timeout=45)
    emit("memory_recorded", memory_id=memory_id, session_id=first_id)
    await asyncio.wait_for(first.close(), 60)

    second = await connect(command, env, clients)
    await turn(second, "We are discussing the cobalt launch. Tell me any relevant project guidance. "
               "Do not inspect files or invoke tools in this turn.", "RECALL_CHECKED")
    await second.wait_until(lambda obs: "[Memory recalled | " + memory_id in obs.transcript_text(), timeout=30)
    emit("cross_session_recall", memory_id=memory_id)
    text = await turn(second, 'Use view(buffer="project_sessions"), then '
        f'view(buffer="ses_{first_id.replace("-", "")[:8]}"). '
        'If a buffer is pending/loading, retry until it is readable. '
        'Report the saved transcript revision and the prior final reply.', "HISTORY_CHECKED")
    assert "MEMORY_RECORDED" in text and ("Revision" in text or "revision" in text), "History projection not observed"
    emit("revision_history", source_session=first_id)
    await turn(second, f'Call update_memory(id="{memory_id}", content="The cobalt launch checklist requires three reviewers."). '
               f'Then call suppress_memory(id="{memory_id}", turns=2).', "MEMORY_UPDATED")
    await asyncio.wait_for(second.close(), 60)

    outage = Path(env["AZO_MEMORY_TEST_OUTAGE"])
    outage.touch()
    resumed = await connect(command + ["--resume", first_id], env, clients)
    text = await turn(resumed, 'Use view(buffer="project_memory") and report the current reviewer rule. '
                      'If the snapshot is loading, retry.', "MEMORY_RESTARTED")
    assert "three reviewers" in text, "Updated memory missing after restart"
    text = await turn(resumed, f'Call delete_memory(id="{memory_id}"). Then view project_memory to confirm deletion.', "MEMORY_DELETED")
    assert any("Deleted" in result for result in results(resumed, "delete_memory")), "Missing deletion receipt"
    assert results(second, "update_memory") and results(second, "suppress_memory"), "Missing mutation tools"
    assert any("local queued" in result.lower() for result in results(resumed, "delete_memory")), "Expected local-only outage receipt"
    emit("offline_restart_and_delete", session_id=first_id, memory_id=memory_id)
    await asyncio.wait_for(resumed.close(), 60)
    outage.unlink()
    recovered = await connect(command + ["--resume", first_id], env, clients)
    text = await turn(recovered, 'Use view(buffer="project_memory"). If sharing is still pending, retry. '
        'Confirm that the deleted memory is absent and report synchronization status.', "MEMORY_RECOVERED")
    assert "## " + memory_id not in results(recovered, "view")[-1], "Deleted memory reappeared"
    from tmux_pilot.fs_store import RecordStore
    shared = RecordStore(args.dev_root / "projects" / project / "plugin-data/project-memory", create=False)
    deadline = time.monotonic() + 30
    while True:
        record = shared.get("project_memory", memory_id)
        if record and record.get("deleted"):
            break
        assert time.monotonic() < deadline, "Offline deletion was not published after recovery"
        await asyncio.sleep(0.1)
    emit("shared_delete_verified", memory_id=memory_id, revision=record["revision"])
    await asyncio.wait_for(recovered.close(), 60)
    emit("passed", project=project, memory_id=memory_id, restarted_session=first_id)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-root", type=Path, required=True)
    args = parser.parse_args()
    if args.dev_root.resolve() == (Path.home() / ".local/share/agent-zoo").resolve():
        parser.error("Use a disposable dev install, never production")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
