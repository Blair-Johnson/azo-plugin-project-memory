"""Pure saved-transcript projection; no live services or session restoration."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

import project_memory as pm


SESSION = dict(session_id="344655c1-1111-1111-1111-111111111111",
               title="Saved work", description="No paths in this description",
               created_at=1, updated_at=2)


def render(entries=None, *, document=None):
    if document is None:
        document = {"entries": entries or []}
    return pm._build_full_transcript(
        SESSION, document, commit_id="pinned-commit", instance_id="source-instance",
        source_format="journal", generated_at=3,
    )


def request(cid="call-1", name="terminal_run", args='{"command":"pwd"}'):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def record(cid="call-1", name="terminal_run", args='{"command":"pwd"}', **extra):
    return dict(id=cid, name=name, raw_args=args, parsed_args={}, **extra)


def test_newest_entries_first_with_chronological_messages_provenance_and_stored_events():
    entries = [
        {"index": 8, "step": 2, "event_time": {"wall_time": 123.0},
         "messages": [{"role": "user", "content": "first question"},
                      {"role": "assistant", "channel": "commentary", "content": "checking first"}]},
        {"index": 11, "compressed": True, "forgotten": True, "system_generated": True,
         "messages": [{"role": "system", "content": "stored interrupt event"},
                      {"role": "assistant", "_responses_phase": "final", "content": "last answer"}]},
    ]
    before = deepcopy(entries)
    text = render(entries)
    assert entries == before
    assert text.index("stored interrupt event") < text.index("last answer") < text.index("first question") < text.index("checking first")
    for expected in ("Newest entry first", "messages within each entry remain chronological",
                     "entry 8, message 1", "ENTRY 11 [document position 1]",
                     'Message channel: "commentary"', 'Message _responses_phase: "final"',
                     "Entry compressed: true", "Entry forgotten: true", "Entry system_generated: true",
                     "123.0", "pinned-commit", "source-instance", "Source Format: journal",
                     "not journal file lines", "unsaved/live activity"):
        assert expected in text


@pytest.mark.parametrize("historical", [False, True])
def test_tail_is_visible_in_first_200_lines(historical):
    entries = [
        {"index": index, "messages": [{"role": "user", "content": f"question-{index:03d}"}]}
        for index in range(100)
    ]
    document = {"entries": entries}
    if historical:
        document = {"state": document}
    text = render(document=document)
    first_page = "\n".join(text.splitlines()[:200])
    assert "question-099" in first_page
    assert "question-000" not in first_page
    assert [line for line in text.splitlines() if line.startswith("=== ENTRY")] == [
        f"=== ENTRY {index} [document position {index}] ===" for index in reversed(range(100))
    ]


def test_reverse_document_order_preserves_fallback_positions_and_unsupported_entries():
    text = render([
        {"index": 40, "messages": [{"role": "user", "content": "oldest"}]},
        None,
        {"messages": [{"role": "assistant", "content": "middle"}]},
        {"index": 3, "messages": [{"role": "assistant", "content": "newest"}]},
    ])
    markers = ["=== ENTRY 3 [document position 3] ===",
               "=== ENTRY 2 [document position 2] ===",
               "[document entry position 1: unsupported entry omitted]",
               "=== ENTRY 40 [document position 0] ==="]
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "ASSISTANT [entry 2, message 0]" in text


def test_historical_state_shape_and_empty_document():
    text = render(document={"state": {"entries": [
        {"index": 0, "messages": [{"role": "user", "content": "historical shape"}]},
    ]}})
    assert "historical shape" in text
    assert "No saved entries" in render(document={"state": {}})


def test_paths_only_in_raw_tool_arguments_and_results_are_searchable_and_not_duplicated():
    path = "/private/worktrees/topic one"
    args = json.dumps({"command": f"git worktree add '{path}' topic"})
    change_args = json.dumps({"path": path})
    text = render([
        {"index": 5, "messages": [
            {"role": "assistant", "content": "Preparing checkout", "tool_calls": [request(args=args)]},
            {"role": "tool", "tool_call_id": "call-1", "content": f"Created checkout at {path}"},
            {"role": "assistant", "content": "Entering checkout", "tool_calls": [request("cd", "change_directory", change_args)]},
            {"role": "tool", "tool_call_id": "cd", "content": "directory changed"},
        ], "tool_calls": [
            record(args=args, result=f"Created checkout at {path}", error=False),
            record("cd", "change_directory", change_args, result="directory changed"),
        ]},
        {"index": 6, "messages": [{"role": "assistant", "content": "Done"}]},
    ])
    assert text.count(args) == 1
    assert text.count("Created checkout at " + path) == 1
    assert text.count("directory changed") == 1
    assert "change_directory" in text and "git worktree add" in text
    lines = text.splitlines()
    match = next(i for i, line in enumerate(lines) if "git worktree add" in line)
    assert "entry 5, message 0" in "\n".join(lines[max(0, match - 3):match + 1])


def test_raw_invalid_json_and_entry_only_fallback_timing_errors_and_no_truncation():
    args = '{"command": "unterminated\nRAW_ARGUMENT_MARKER'
    result = "result-start\n" + "x" * 180_000 + "\nresult-end"
    text = render([{"index": 7, "messages": [], "tool_calls": [
        record(args=args, result=result, error=True,
               event_time={"wall_time": 10}, started_at={"wall_time": 11}, finished_at={"wall_time": 12}),
    ]}])
    assert args in text and result in text
    assert "entry-level fallback; completion order not recorded" in text
    for expected in ("Stored error: true", "Stored event_time", "Stored started_at", "Stored finished_at"):
        assert expected in text


@pytest.mark.parametrize("value,expected", [("", "[empty string]"), ([], "[empty list]"), ({}, "{}"), (False, "false"), (0, "0")])
def test_empty_and_falsey_results_are_present(value, expected):
    text = render([{"index": 0, "tool_calls": [record(result=value)]}])
    assert "Result (entry-level fallback" in text
    assert expected in text
    assert "Result not recorded at this revision." not in text


@pytest.mark.parametrize("extra", [{}, {"result": None}])
def test_missing_result_is_not_claimed_running(extra):
    text = render([{"index": 0, "tool_calls": [record(**extra)]}])
    assert "Result not recorded at this revision." in text
    assert "not proof a call is running" in text
    assert "still running" not in text


def test_parsed_arguments_are_explicit_fallback_not_replacement_for_raw():
    text = render([{"index": 0, "tool_calls": [
        {"id": "parsed", "name": "f", "parsed_args": {"path": "/fallback/only"}},
        record("raw", args="INVALID RAW", parsed_ignored="not arguments"),
    ]}])
    assert "Parsed arguments (fallback; raw arguments not recorded)" in text
    assert "/fallback/only" in text and "INVALID RAW" in text


def test_different_ids_and_reused_ids_remain_distinct_executions():
    call_args = "IDENTICAL ARGUMENTS"
    messages = []
    for cid in ("one", "two", "one"):
        messages.extend([
            {"role": "assistant", "tool_calls": [request(cid, args=call_args)]},
            {"role": "tool", "tool_call_id": cid, "content": "IDENTICAL RESULT"},
        ])
    text = render([{"index": 0, "messages": messages}])
    assert text.count(call_args) == 3
    assert text.count("IDENTICAL RESULT") == 3
    assert "reused call ID" in text
    assert "request entry 0, message 4, call 0" in text


def test_missing_ids_orphan_results_and_ambiguous_reuse_stay_visible():
    text = render([{"index": 0, "messages": [
        {"role": "tool", "tool_call_id": "orphan", "content": "orphan output"},
        {"role": "assistant", "tool_calls": [request(None, args="no ID argument")]},
        {"role": "tool", "content": "missing ID output"},
        {"role": "assistant", "tool_calls": [request("twice"), request("twice")]},
        {"role": "tool", "tool_call_id": "twice", "content": "ambiguous output"},
    ], "tool_calls": [record(None, args="no ID argument", result="uncorrelated fallback")]}])
    for expected in ("orphan result", "orphan output", "[missing ID]", "missing ID output",
                     "ambiguous reused ID", "ambiguous output", "uncorrelated fallback"):
        assert expected in text
    assert text.count("no ID argument") == 2  # No content-only deduplication.


def test_conflicting_request_and_result_representations_are_not_erased():
    text = render([{"index": 0, "messages": [
        {"role": "assistant", "tool_calls": [request(args="message arguments")]},
        {"role": "tool", "tool_call_id": "call-1", "content": "message result"},
    ], "tool_calls": [record(args="message arguments", result="different durable result"),
                      record(args="different durable arguments", result="other result")]}])
    for expected in ("message arguments", "message result", "different durable result", "different durable arguments", "other result",
                     "conflicting result representations", "conflicting or ambiguous duplicate representation"):
        assert expected in text
    assert text.count("message arguments") == 1


def test_result_in_later_entry_is_message_first_and_enriched_without_duplication():
    text = render([
        {"index": 2, "messages": [{"role": "assistant", "tool_calls": [request()]}],
         "tool_calls": [record(result="UNIQUE_RESULT", error=True)]},
        {"index": 3, "messages": [{"role": "tool", "tool_call_id": "call-1", "content": "UNIQUE_RESULT"}]},
    ])
    assert text.count("UNIQUE_RESULT") == 1
    assert "Stored error: true" in text
    assert "Result represented at entry 3, message 0" in text
    assert text.index("ENTRY 3") < text.index("UNIQUE_RESULT") < text.index("ENTRY 2")
    assert "TOOL RESULT id=call-1 (request entry 2, message 0, call 0)" in text


def test_provider_activity_blocks_artifact_refs_and_opaque_payload_omissions():
    text = render([{"index": 0, "messages": [{
        "role": "assistant", "content": [
            {"type": "text", "text": "visible block"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE_SECRET"}},
            {"type": "thinking", "thinking": "THINKING_SECRET"},
        ], "provider_tool_calls": [
            {"type": "web_search_call", "id": "hosted-search", "status": "completed",
             "action": {"type": "search", "query": "visible search query"}},
            {"type": "compaction", "encrypted_content": "CIPHER_SECRET"},
        ], "_responses_reasoning_items": [{"encrypted_content": "REASONING_SECRET"}],
    }], "tool_calls": [record(result="output stored in buf_old_missing", blocks=[{"tag": "F1:C2", "text": "chunk evidence"}],
                               artifacts=[{"kind": "diff", "format": "diff", "path": "/not/fetched.patch", "body": "-old\n+new"}])]}])
    for expected in ("visible block", "PROVIDER-HOSTED ACTIVITY", "not a local function call", "visible search query",
                     "nontext", "omitted", "buf_old_missing", "F1:C2", "chunk evidence", "/not/fetched.patch", "-old\n+new"):
        assert expected in text
    for forbidden in ("IMAGE_SECRET", "THINKING_SECRET", "CIPHER_SECRET", "REASONING_SECRET"):
        assert forbidden not in text


def test_hidden_analysis_is_not_rendered_but_local_requests_are_observable():
    text = render([{"index": 0, "messages": [{"role": "assistant", "channel": "analysis",
                                             "content": "PRIVATE_REASONING", "tool_calls": [request()]}]}])
    assert "PRIVATE_REASONING" not in text
    assert "hidden reasoning content omitted" in text
    assert "terminal_run" in text


def test_entry_content_blocks_are_not_duplicated_when_also_on_tool_record():
    blocks = [{"tag": "B1", "text": "UNIQUE_BLOCK_TEXT"}]
    text = render([{"index": 0, "tool_calls": [record(blocks=blocks)], "blocks": {"call-1": blocks}}])
    assert text.count("UNIQUE_BLOCK_TEXT") == 1
    assert "identical tool-record blocks" in text


def test_projection_is_pure_and_never_invokes_render_hooks_or_reads_references(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("formatter performed IO or called a render hook")

    document = {"entries": [{"index": 0, "render_fn": forbidden, "compress_fn": forbidden,
                             "messages": [{"role": "user", "content": "plain text"}],
                             "tool_calls": [record(result="See /missing/file and https://invalid.example", artifacts=[
                                 {"kind": "file", "path": "/missing/artifact", "format": "text"},
                             ])]}], "state": {"opaque": "IGNORED_RUNTIME_STATE"}}
    before = deepcopy(document)
    monkeypatch.setattr("builtins.open", forbidden)
    text = render(document=document)
    assert document == before
    assert "plain text" in text and "/missing/artifact" in text
    assert "IGNORED_RUNTIME_STATE" not in text


def test_entry_only_request_and_same_entry_tool_message_share_result():
    text = render([{"index": 0, "messages": [
        {"role": "tool", "tool_call_id": "call-1", "content": "ENTRY_ONLY_UNIQUE_RESULT"},
    ], "tool_calls": [record(result="ENTRY_ONLY_UNIQUE_RESULT")]}])
    assert text.count("ENTRY_ONLY_UNIQUE_RESULT") == 1
    assert "request entry 0, tool record 0" in text
    assert "LOCAL FUNCTION REQUEST" in text
    assert "entry-only request" in text


def test_durable_record_on_later_result_entry_enriches_preceding_request():
    text = render([
        {"index": 0, "messages": [{"role": "assistant", "tool_calls": [request(args="CROSS_ENTRY_ARGUMENTS")]}]},
        {"index": 1, "messages": [{"role": "tool", "tool_call_id": "call-1", "content": "CROSS_ENTRY_RESULT"}],
         "tool_calls": [record(args="CROSS_ENTRY_ARGUMENTS", result="CROSS_ENTRY_RESULT", error=True)]},
    ])
    assert text.count("CROSS_ENTRY_ARGUMENTS") == 1
    assert text.count("CROSS_ENTRY_RESULT") == 1
    assert "Stored error: true" in text
    assert "Request represented at entry 0, message 0, call 0" in text
