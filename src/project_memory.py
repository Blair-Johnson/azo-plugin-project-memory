"""Project-scoped regex memories for Agent Zoo pipelines."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from agent_utils import Entry, Feature, Tool, estimate_tokens
from agent_utils.components import (
    CompressToolResults,
    ConsolidateToolResults,
    ExcludeForgotten,
    MessageRenderer,
    ToolDispatchStart,
    TurnCounter,
    _wrap_entry_renderer,
    safe_interrupt_text,
)
from agent_utils.failed_tool_calls import FailedToolCallRecorder
from agent_utils.files.buffer_manager import ReadonlyBufferView
from agents.system_prompt import SystemPromptSkillList
from agent_zoo.tools.special_buffers import RegisterSpecialBuffers


log = logging.getLogger("agent_zoo_user_plugin.project_memory")

PLUGIN_NAME = "azo-plugin-project-memory"
CONFIG_SECTION = "project_memory"
CONFIG_PATH_ENV = "AZO_PROJECT_MEMORY_CONFIG"
CONFIG_FILENAME = "project_memory.yaml"
TABLE_NAME = "azo_project_memories"
SUPPRESSIONS_ATTR = "project_memory_suppressions"
SEEN_EVENTS_ATTR = "project_memory_seen_events"
PENDING_RECALLS_ATTR = "project_memory_pending_recalls"
BOOTSTRAPPED_ATTR = "project_memory_scan_bootstrapped"
LAST_ENTRY_COUNT_ATTR = "project_memory_last_entry_count"
MAX_SEEN_EVENTS = 4096
DEFAULT_SYSTEM_PROMPT = (
    "Use project memories for durable project-specific facts, decisions, constraints, "
    "and procedures that should be recalled in future sessions. Call record_memory with "
    "concise standalone content and a selective regular expression that matches future "
    "transcript messages indicating relevance. Refine noisy memories with update_memory "
    "or temporarily quiet them with suppress_memory; delete obsolete memories. Do not "
    "record secrets, transient progress, or facts already maintained in authoritative "
    "project files."
)


class ProjectMemoryStore:
    """Project-database storage component shared by all memory components."""

    optional_reads = {"run_db", "agent_db"}

    def __call__(self, state):
        try:
            self.ensure_schema(state)
        except RuntimeError:
            pass
        return state

    @staticmethod
    def _agent_db(state):
        run_db = getattr(state, "run_db", None)
        agent_db = getattr(run_db, "agent_db", None) or getattr(state, "agent_db", None)
        if agent_db is None or not hasattr(agent_db, "conn"):
            raise RuntimeError("project memory requires a live Agent Zoo project database")
        return agent_db

    def ensure_schema(self, state):
        conn = self._agent_db(state).conn
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id         TEXT PRIMARY KEY,
                content    TEXT NOT NULL,
                trigger    TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_updated_at "
            f"ON {TABLE_NAME}(updated_at)"
        )
        conn.commit()
        return conn

    @staticmethod
    def invalidate_buffer(state) -> None:
        manager = getattr(state, "buffer_manager", None)
        cache = getattr(manager, "_special_cache", None)
        if isinstance(cache, dict):
            cache.pop("project_memory", None)
    @staticmethod
    def _validate_content(content: str) -> str:
        text = str(content or "").strip()
        if not text:
            raise ValueError("memory content must not be empty")
        return text

    @staticmethod
    def _validate_trigger(trigger: str) -> str:
        pattern = str(trigger or "").strip()
        if not pattern:
            raise ValueError("memory trigger must not be empty")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid memory trigger regex: {exc}") from exc
        return pattern

    @staticmethod
    def _memory_from_row(row) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "content": str(row["content"]),
            "trigger": str(row["trigger"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def get(self, state, memory_id: str) -> dict[str, Any] | None:
        conn = self.ensure_schema(state)
        row = conn.execute(
            f"SELECT id, content, trigger, created_at, updated_at "
            f"FROM {TABLE_NAME} WHERE id = ?",
            (str(memory_id or "").strip(),),
        ).fetchone()
        return self._memory_from_row(row)

    def list_all(self, state) -> list[dict[str, Any]]:
        conn = self.ensure_schema(state)
        rows = conn.execute(
            f"SELECT id, content, trigger, created_at, updated_at "
            f"FROM {TABLE_NAME} ORDER BY created_at, id"
        ).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def create(self, state, content: str, trigger: str) -> dict[str, Any]:
        content = self._validate_content(content)
        trigger = self._validate_trigger(trigger)
        conn = self.ensure_schema(state)
        now = time.time()
        for _attempt in range(32):
            memory_id = f"mem_{secrets.token_hex(3)}"
            try:
                conn.execute(
                    f"INSERT INTO {TABLE_NAME} "
                    "(id, content, trigger, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (memory_id, content, trigger, now, now),
                )
                conn.commit()
                memory = {
                    "id": memory_id,
                    "content": content,
                    "trigger": trigger,
                    "created_at": now,
                    "updated_at": now,
                }
                self.invalidate_buffer(state)
                return memory
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("could not allocate a unique project memory id")

    def update(
        self,
        state,
        memory_id: str,
        *,
        content: str | None = None,
        trigger: str | None = None,
    ) -> dict[str, Any]:
        memory_id = str(memory_id or "").strip()
        current = self.get(state, memory_id)
        if current is None:
            raise KeyError(f"unknown project memory {memory_id!r}")
        if content is None and trigger is None:
            raise ValueError("update_memory requires content and/or trigger")
        next_content = current["content"] if content is None else self._validate_content(content)
        next_trigger = current["trigger"] if trigger is None else self._validate_trigger(trigger)
        now = time.time()
        conn = self.ensure_schema(state)
        conn.execute(
            f"UPDATE {TABLE_NAME} SET content = ?, trigger = ?, updated_at = ? WHERE id = ?",
            (next_content, next_trigger, now, memory_id),
        )
        conn.commit()
        memory = {
            **current,
            "content": next_content,
            "trigger": next_trigger,
            "updated_at": now,
        }
        self.invalidate_buffer(state)
        return memory

    def delete(self, state, memory_id: str) -> bool:
        memory_id = str(memory_id or "").strip()
        conn = self.ensure_schema(state)
        cursor = conn.execute(f"DELETE FROM {TABLE_NAME} WHERE id = ?", (memory_id,))
        conn.commit()
        deleted = bool(cursor.rowcount)
        if deleted:
            self.invalidate_buffer(state)
        return deleted


class ProjectMemoryBuffer:
    """Register the live readonly ``project_memory`` special buffer."""

    reads = {"buffer_manager"}
    optional_reads = {"run_db", "agent_db"}
    writes = {"buffer_manager"}

    def __init__(self, store: ProjectMemoryStore) -> None:
        self.store = store

    def __call__(self, state):
        manager = state.buffer_manager
        cache = getattr(manager, "_special_cache", None)
        if isinstance(cache, dict):
            cache.pop("project_memory", None)
        if not manager.is_special("project_memory"):
            manager.register_special_buffer("project_memory", self.render)
        return state

    def render(self, state) -> ReadonlyBufferView:
        try:
            memories = self.store.list_all(state)
        except Exception as exc:
            text = (
                "# Project memories\n\n"
                f"Project database unavailable: {type(exc).__name__}: {exc}\n"
            )
        else:
            lines = ["# Project memories", "", f"Count: {len(memories)}"]
            for memory in memories:
                lines.extend(
                    [
                        "",
                        f"## {memory['id']}",
                        f"Trigger regex: {json.dumps(memory['trigger'], ensure_ascii=False)}",
                        "",
                        "Content:",
                        str(memory["content"]),
                    ]
                )
            text = "\n".join(lines).rstrip() + "\n"
        return ReadonlyBufferView(
            id="project_memory",
            path="memory://project",
            text=text,
        )


class ProjectMemoryTool(Tool):
    """Shared dispatch behavior for project-memory tools."""

    group = "Project Memory"
    optional_reads = {"run_db", "agent_db"}

    def __init__(self, store: ProjectMemoryStore) -> None:
        super().__init__()
        self.store = store

    def transform(self, state):
        for tc, kwargs in self.pending(state):
            try:
                tc.result = str(self.execute(state, **kwargs))
            except Exception as exc:
                tc.result = f"ERROR: {type(exc).__name__}: {exc}"
                tc.error = True
            self._notify(state, tc)
        return state

    def execute(self, state, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _format(memory: Mapping[str, Any], verb: str) -> str:
        return (
            f"{verb} {memory['id']}.\n"
            f"Trigger: {memory['trigger']}\n"
            f"Content: {memory['content']}"
        )


class RecordMemory(ProjectMemoryTool):
    """Save a project-scoped memory with a regular-expression trigger."""

    @staticmethod
    def fn(content: str, trigger: str) -> str:
        """Save durable project knowledge for later regex-triggered recall.

        Args:
            content: Standalone text that should be recalled in a relevant future context.
            trigger: Python regular expression matched against future transcript messages.
        """

    def execute(self, state, *, content: str, trigger: str) -> str:
        return self._format(self.store.create(state, content, trigger), "Recorded")


class UpdateMemory(ProjectMemoryTool):
    """Refine an existing project memory."""

    optional_reads = {"run_db", "agent_db", PENDING_RECALLS_ATTR}
    writes = {PENDING_RECALLS_ATTR, "tool_schemas"}
    init = {PENDING_RECALLS_ATTR: list, "tool_schemas": dict}

    @staticmethod
    def fn(
        id: str,
        content: str | None = None,
        trigger: str | None = None,
    ) -> str:
        """Update a memory's content, trigger regex, or both.

        Args:
            id: Memory identifier such as mem_a1b2c3.
            content: Replacement memory content. Omit to preserve existing content.
            trigger: Replacement Python regex. Omit to preserve the existing trigger.
        """

    def execute(
        self,
        state,
        *,
        id: str,
        content: str | None = None,
        trigger: str | None = None,
    ) -> str:
        memory = self.store.update(state, id, content=content, trigger=trigger)
        for pending in getattr(state, PENDING_RECALLS_ATTR):
            if pending.get("id") == memory["id"]:
                pending["content"] = memory["content"]
        return self._format(memory, "Updated")


class SuppressMemory(ProjectMemoryTool):
    """Temporarily suppress a memory in the current session."""

    optional_reads = {"run_db", "agent_db", PENDING_RECALLS_ATTR}
    writes = {SUPPRESSIONS_ATTR, PENDING_RECALLS_ATTR, "tool_schemas"}
    init = {
        SUPPRESSIONS_ATTR: dict,
        PENDING_RECALLS_ATTR: list,
        "tool_schemas": dict,
    }

    @staticmethod
    def fn(id: str, turns: int) -> str:
        """Suppress an irrelevant memory in this conversation for several turns.

        Args:
            id: Memory identifier such as mem_a1b2c3.
            turns: Positive number of pipeline turns to suppress, beginning now.
        """

    def execute(self, state, *, id: str, turns: int) -> str:
        memory_id = str(id or "").strip()
        if self.store.get(state, memory_id) is None:
            raise KeyError(f"unknown project memory {memory_id!r}")
        try:
            count = int(turns)
        except (TypeError, ValueError) as exc:
            raise ValueError("turns must be a positive integer") from exc
        if count <= 0:
            raise ValueError("turns must be a positive integer")
        expires_at = int(getattr(state, "step", 0)) + count
        suppressions = getattr(state, SUPPRESSIONS_ATTR)
        suppressions[memory_id] = expires_at
        pending = getattr(state, PENDING_RECALLS_ATTR)
        pending[:] = [item for item in pending if item.get("id") != memory_id]
        return (
            f"Suppressed {memory_id} for {count} turn{'s' if count != 1 else ''} "
            f"in this session (through step {expires_at - 1})."
        )


class DeleteMemory(ProjectMemoryTool):
    """Delete an obsolete project memory."""

    optional_reads = {"run_db", "agent_db", SUPPRESSIONS_ATTR, PENDING_RECALLS_ATTR}
    writes = {SUPPRESSIONS_ATTR, PENDING_RECALLS_ATTR, "tool_schemas"}
    init = {
        SUPPRESSIONS_ATTR: dict,
        PENDING_RECALLS_ATTR: list,
        "tool_schemas": dict,
    }

    @staticmethod
    def fn(id: str) -> str:
        """Permanently delete a project memory.

        Args:
            id: Memory identifier such as mem_a1b2c3.
        """

    def execute(self, state, *, id: str) -> str:
        memory_id = str(id or "").strip()
        if not self.store.delete(state, memory_id):
            raise KeyError(f"unknown project memory {memory_id!r}")
        getattr(state, SUPPRESSIONS_ATTR).pop(memory_id, None)
        pending = getattr(state, PENDING_RECALLS_ATTR)
        pending[:] = [item for item in pending if item.get("id") != memory_id]
        return f"Deleted {memory_id}."


class ProjectMemoryRecall:
    """Recall matching memories after transcript tool results are consolidated."""

    reads = {"entries", "step"}
    optional_reads = {"run_db", "agent_db", "context_compaction"}
    writes = {
        SEEN_EVENTS_ATTR,
        PENDING_RECALLS_ATTR,
        SUPPRESSIONS_ATTR,
        BOOTSTRAPPED_ATTR,
        LAST_ENTRY_COUNT_ATTR,
    }
    init = {
        SEEN_EVENTS_ATTR: list,
        PENDING_RECALLS_ATTR: list,
        SUPPRESSIONS_ATTR: dict,
        BOOTSTRAPPED_ATTR: bool,
        LAST_ENTRY_COUNT_ATTR: int,
    }

    def __init__(self, store: ProjectMemoryStore) -> None:
        self.store = store

    def __call__(self, state):
        entries = list(getattr(state, "entries", []) or [])
        events = self._new_events(state, entries)
        if not events:
            return state
        try:
            memories = self.store.list_all(state)
        except Exception:
            log.debug("project memory database unavailable during recall", exc_info=True)
            return state
        if not memories:
            return state

        step = int(getattr(state, "step", 0) or 0)
        suppressions = getattr(state, SUPPRESSIONS_ATTR)
        for memory_id, expires_at in list(suppressions.items()):
            try:
                expired = step >= int(expires_at)
            except (TypeError, ValueError):
                expired = True
            if expired:
                suppressions.pop(memory_id, None)

        active_text = self._active_context_text(state, entries)
        pending = getattr(state, PENDING_RECALLS_ATTR)
        queued_ids = {str(item.get("id") or "") for item in pending if isinstance(item, Mapping)}
        event_texts = [text for _event_id, text in events if text]

        for memory in memories:
            memory_id = memory["id"]
            if memory_id in queued_ids or memory_id in suppressions:
                continue
            content = str(memory.get("content") or "").strip()
            if not content or self._content_is_active(content, active_text):
                continue
            try:
                trigger = re.compile(str(memory.get("trigger") or ""))
            except re.error:
                log.warning("Skipping project memory %s with invalid stored regex", memory_id)
                continue
            if not any(trigger.search(text) for text in event_texts):
                continue
            pending.append({"id": memory_id, "content": content})
            queued_ids.add(memory_id)
        return state

    def _new_events(self, state, entries) -> list[tuple[str, str]]:
        seen_order = [str(item) for item in (getattr(state, SEEN_EVENTS_ATTR) or [])]
        seen = set(seen_order)
        current_ids: list[str] = []
        candidates: list[tuple[str, str]] = []
        bootstrapped = bool(getattr(state, BOOTSTRAPPED_ATTR, False))
        current_step = self._int_value(getattr(state, "step", 0), 0)

        for position, entry in enumerate(entries):
            entry_step = self._int_value(self._get(entry, "step", -1), -1)
            for event_id, text in self._entry_events(entry, position):
                current_ids.append(event_id)
                if event_id in seen:
                    continue
                if bootstrapped or entry_step >= current_step:
                    candidates.append((event_id, text))
                seen.add(event_id)
                seen_order.append(event_id)

        previous_count = self._int_value(getattr(state, LAST_ENTRY_COUNT_ATTR, 0), 0)
        if len(entries) < previous_count:
            current = set(current_ids)
            seen_order = [event_id for event_id in seen_order if event_id in current]

        setattr(state, SEEN_EVENTS_ATTR, seen_order[-MAX_SEEN_EVENTS:])
        setattr(state, BOOTSTRAPPED_ATTR, True)
        setattr(state, LAST_ENTRY_COUNT_ATTR, len(entries))
        return candidates

    def _entry_events(self, entry, position: int) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        entry_index = self._int_value(self._get(entry, "index", position), position)
        entry_step = self._int_value(self._get(entry, "step", -1), -1)
        messages = self._get(entry, "messages", ()) or ()
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "")
            content = self._content_text(message.get("content"))
            if content and role != "system":
                event_id = self._hashed_event_id(
                    "message", entry_index, entry_step, message_index, role, content
                )
                events.append((event_id, f"{role} message:\n{content}"))

            for call_index, call in enumerate(message.get("tool_calls") or ()):
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function") or {}
                if not isinstance(function, Mapping):
                    function = {}
                call_id = str(call.get("id") or "").strip()
                name = str(function.get("name") or "")
                arguments = self._content_text(function.get("arguments"))
                event_id = (
                    f"tool-call:{call_id}"
                    if call_id
                    else self._hashed_event_id(
                        "tool-call", entry_index, entry_step, message_index, call_index, name, arguments
                    )
                )
                events.append((event_id, f"Tool call {name}:\n{arguments}"))

            if role == "tool":
                call_id = str(message.get("tool_call_id") or "").strip()
                event_id = (
                    f"tool-result:{call_id}"
                    if call_id
                    else self._hashed_event_id(
                        "tool-result", entry_index, entry_step, message_index, content
                    )
                )
                events.append((event_id, f"Tool result:\n{content}"))
        return events

    def _active_context_text(self, state, entries) -> str:
        harness_position, handoff_text = self._harness_boundary(state, entries)
        provider_position = self._provider_boundary(entries)

        if provider_position is not None and (
            harness_position is None or provider_position >= harness_position
        ):
            start = provider_position + 1
            prefix = []
        else:
            start = harness_position or 0
            prefix = [handoff_text] if handoff_text else []

        parts = list(prefix)
        for entry in entries[start:]:
            if bool(self._get(entry, "forgotten", False)):
                continue
            for message in self._get(entry, "messages", ()) or ():
                if isinstance(message, Mapping):
                    text = self._message_context_text(message)
                    if text:
                        parts.append(text)
        return "\n\n".join(parts)

    def _harness_boundary(self, state, entries) -> tuple[int | None, str]:
        cc = getattr(state, "context_compaction", None)
        docs = list(self._get(cc, "handoff_docs", ()) or ())
        if not docs or not entries:
            return None, ""
        current_turn = int(getattr(state, "step", 0) or 0)
        current_last_index = max(
            self._int_value(self._get(entry, "index", pos), pos)
            for pos, entry in enumerate(entries)
        )
        chosen = None
        for doc in reversed(docs):
            if self._get(doc, "error", None):
                continue
            source_turn = self._int_value(self._get(doc, "source_end_turn", -1), -1)
            source_index = self._int_value(
                self._get(doc, "source_end_entry_index", -1), -1
            )
            if source_turn <= current_turn and source_index <= current_last_index:
                chosen = doc
                break
        if chosen is None:
            return None, ""
        source_index = self._int_value(
            self._get(chosen, "source_end_entry_index", -1), -1
        )
        tail_index = self._int_value(
            self._get(chosen, "preserved_tail_start_entry_index", source_index + 1),
            source_index + 1,
        )
        for position, entry in enumerate(entries):
            if self._int_value(self._get(entry, "index", position), position) >= tail_index:
                return position, str(self._get(chosen, "text", "") or "")
        return len(entries), str(self._get(chosen, "text", "") or "")

    def _provider_boundary(self, entries) -> int | None:
        latest = None
        for position, entry in enumerate(entries):
            messages = list(self._get(entry, "messages", ()) or ())
            direct_calls = self._get(entry, "provider_tool_calls", ()) or ()
            if direct_calls:
                messages.append({"provider_tool_calls": direct_calls})
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                provenance = message.get("_provenance")
                if isinstance(provenance, Mapping):
                    api = str(provenance.get("api") or "")
                    if api and api != "responses":
                        continue
                for item in message.get("provider_tool_calls") or ():
                    if not isinstance(item, Mapping):
                        continue
                    kind = str(item.get("type") or "")
                    if kind in {"compaction", "compaction_summary"} and item.get(
                        "encrypted_content"
                    ):
                        latest = position
        return latest

    def _message_context_text(self, message: Mapping[str, Any]) -> str:
        parts = []
        content = self._content_text(message.get("content"))
        if content:
            parts.append(content)
        for call in message.get("tool_calls") or ():
            if not isinstance(call, Mapping):
                continue
            function = call.get("function") or {}
            if not isinstance(function, Mapping):
                continue
            parts.append(str(function.get("name") or ""))
            parts.append(self._content_text(function.get("arguments")))
        for item in message.get("provider_tool_calls") or ():
            if not isinstance(item, Mapping):
                continue
            if str(item.get("type") or "") in {"compaction", "compaction_summary"}:
                continue
            cleaned = {
                str(key): value
                for key, value in item.items()
                if key not in {"encrypted_content", "data"}
            }
            parts.append(self._content_text(cleaned))
        return "\n".join(part for part in parts if part)

    @classmethod
    def _content_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for key in ("text", "content", "output", "arguments"):
                if key in value:
                    text = cls._content_text(value.get(key))
                    if text:
                        return text
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return "\n".join(filter(None, (cls._content_text(item) for item in value)))
        return str(value)

    @staticmethod
    def _content_is_active(content: str, active_text: str) -> bool:
        return bool(content.strip() and content.strip() in active_text)

    @staticmethod
    def _hashed_event_id(*parts: Any) -> str:
        payload = json.dumps(parts, ensure_ascii=False, default=str, separators=(",", ":"))
        return "message:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _get(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)

    @staticmethod
    def _int_value(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


class ProjectMemoryDelivery:
    """Deliver queued recalls as standard system-generated user entries."""

    reads = {"entries", "step"}
    optional_reads = {PENDING_RECALLS_ATTR}
    writes = {"entries", "done", PENDING_RECALLS_ATTR}
    init = {PENDING_RECALLS_ATTR: list}

    def __call__(self, state):
        pending = getattr(state, PENDING_RECALLS_ATTR)
        if not pending:
            return state
        if state.entries and getattr(state.entries[-1], "role", "") == "user":
            return state

        delivered = list(pending)
        pending.clear()
        for memory in delivered:
            memory_id = str(memory.get("id") or "")
            content = str(memory.get("content") or "")
            text = safe_interrupt_text(f"[Memory recalled | {memory_id}]\n{content}")
            state.entries.append(
                Entry(
                    messages=[{"role": "user", "content": text}],
                    index=len(state.entries),
                    step=state.step,
                    tokens=estimate_tokens(text),
                    system_generated=True,
                )
            )
        state.done = False
        return state


class ProjectMemorySystemPrompt:
    """Append configured project-memory guidance to the system prompt."""

    reads = {"entries"}
    writes = {"entries"}
    marker_attr = "_azo_project_memory_system_prompt_wrapper"

    def __init__(self, text: str) -> None:
        self.text = str(text or "").strip()

    def __call__(self, state):
        if not self.text or not state.entries:
            return state
        entry = state.entries[0]
        block = "# Project Memory\n" + self.text

        def factory(previous_render, _block=block):
            def render(render_state):
                messages = previous_render(render_state)
                if not messages:
                    return messages
                first = dict(messages[0])
                first["content"] = self._append_content(first.get("content"), _block)
                return [first, *messages[1:]]

            return render

        _wrap_entry_renderer(entry, factory, self.marker_attr)
        return state

    @staticmethod
    def _append_content(content: Any, block: str) -> Any:
        if isinstance(content, str):
            return content.rstrip() + "\n\n" + block
        if content is None:
            return block
        if isinstance(content, list):
            rendered = list(content)
            uses_input_blocks = any(
                isinstance(item, Mapping)
                and str(item.get("type") or "").startswith("input_")
                for item in rendered
            )
            rendered.append(
                {"type": "input_text" if uses_input_blocks else "text", "text": block}
            )
            return rendered
        return str(content).rstrip() + "\n\n" + block

    @classmethod
    def effective_section(cls, config: Any) -> dict[str, Any]:
        installed = cls._load_installed_config().get(CONFIG_SECTION, {})
        explicit = (config or {}).get(CONFIG_SECTION, {})
        merged = dict(installed) if isinstance(installed, Mapping) else {}
        if isinstance(explicit, Mapping):
            merged.update(explicit)
        return merged

    @staticmethod
    def enabled(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return value is not False

    @classmethod
    def _load_installed_config(cls) -> dict[str, Any]:
        try:
            loaded = yaml.safe_load(cls._config_path().read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, yaml.YAMLError):
            log.warning("Could not load project-memory config", exc_info=True)
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @classmethod
    def _config_path(cls) -> Path:
        override = str(os.environ.get(CONFIG_PATH_ENV, "") or "").strip()
        if override:
            return Path(override).expanduser()
        return cls._state_root() / "plugin-configs" / PLUGIN_NAME / "config" / CONFIG_FILENAME

    @staticmethod
    def _state_root() -> Path:
        for name in ("AGENT_ZOO_HOME", "AGENT_ZOO_STATE_ROOT", "AGENT_ZOO_INSTALL_ROOT"):
            value = str(os.environ.get(name, "") or "").strip()
            if value:
                return Path(value).expanduser()
        xdg = str(os.environ.get("XDG_DATA_HOME", "") or "").strip()
        base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
        return base / "agent-zoo"


def register_features(builder, *, session, config):
    """Register the project_memory feature in every Agent Zoo pipeline."""
    del session
    section = ProjectMemorySystemPrompt.effective_section(config)
    if not ProjectMemorySystemPrompt.enabled(section.get("enabled", True)):
        return
    prompt = str(section.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
    store = ProjectMemoryStore()
    builder.add(
        Feature(
            name="project_memory",
            components=[
                store,
                ProjectMemoryBuffer(store),
                RecordMemory(store),
                UpdateMemory(store),
                SuppressMemory(store),
                DeleteMemory(store),
                ProjectMemoryRecall(store),
                ProjectMemoryDelivery(),
                ProjectMemorySystemPrompt(prompt),
            ],
            order=[
                TurnCounter,
                RegisterSpecialBuffers,
                ProjectMemoryStore,
                ProjectMemoryBuffer,
                ToolDispatchStart,
                ConsolidateToolResults,
                FailedToolCallRecorder,
                ProjectMemoryRecall,
                ProjectMemoryDelivery,
                CompressToolResults,
                SystemPromptSkillList,
                ProjectMemorySystemPrompt,
                ExcludeForgotten,
                MessageRenderer,
            ],
        )
    )
