from types import SimpleNamespace
import time

from agent_utils import Entry
import project_memory as pm


def state():
    return SimpleNamespace(
        entries=[], step=1, project_memory_seen_events=[],
        project_memory_pending_recalls=[], project_memory_suppressions={},
        project_memory_scan_bootstrapped=False, project_memory_last_entry_count=0,
    )


def message(i, text="old discussion"):
    return Entry(index=i, step=1, messages=[{"role": "assistant", "content": text}])


def test_large_history_does_not_recall_evicted_event_ids():
    s = state()
    s.entries = [message(i) for i in range(5000)]
    recall = pm.ProjectMemoryRecall(pm.ProjectMemoryStore())
    recall._new_events(s, s.entries, baseline_only=True)
    assert recall._new_events(s, s.entries) == []
    s.entries.append(message(5000, "new discussion"))
    events = recall._new_events(s, s.entries)
    assert len(events) == 1 and "new discussion" in events[0][1]


def test_tail_mutation_is_observed():
    s = state()
    s.entries = [message(0)]
    recall = pm.ProjectMemoryRecall(pm.ProjectMemoryStore())
    recall._new_events(s, s.entries, baseline_only=True)
    s.entries[-1].messages.append({"role": "tool", "tool_call_id": "one", "content": "result"})
    assert recall._new_events(s, s.entries) == [("tool-result:one", "Tool result:\nresult")]
    assert recall._new_events(s, s.entries) == []


def test_pathological_regex_is_bounded():
    start = time.monotonic()
    assert not pm._trigger_matches(pm._compile_trigger("(a+)+$"), "a" * 60000 + "!", start + .1)
    assert time.monotonic() - start < 1


def test_backtest_reports_incomplete_expensive_search():
    s = state()
    s.entries = [message(0, "a" * 60000 + "!")]
    matches, truncated = pm.ProjectMemoryRecall(pm.ProjectMemoryStore()).backtest(s, "(a+)+$")
    assert not matches and truncated


def test_ordinary_common_recall_remains_functional():
    s = state()
    s.entries = [message(0, "deploy tonight")]
    store = pm.ProjectMemoryStore({"deploy": {"content": "Use the staging checklist.", "trigger": "deploy"}})
    store.list_all = lambda _: []
    pm.ProjectMemoryRecall(store)(s)
    assert s.project_memory_pending_recalls[0]["content"] == "Use the staging checklist."
