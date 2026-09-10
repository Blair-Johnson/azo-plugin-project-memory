"""Project-scoped regex memories for Agent Zoo pipelines."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import operator
import os
import queue
import re
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import regex

from agent_utils import Entry, Feature, MODEL_RENDER_CHANNEL, Tool, estimate_tokens
from agent_utils.components import (
    CompressToolResults,
    ConsolidateToolResults,
    ExcludeForgotten,
    MessageRenderer,
    RenderTransformRegistrar,
    ToolDispatchStart,
    TurnCounter,
    safe_interrupt_text,
)
from agent_utils.failed_tool_calls import FailedToolCallRecorder
from agent_utils.files.buffer_manager import (
    BufferManager,
    ReadonlyBufferView,
    StateBoundSpecialBufferNamespaceProvider,
)
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
DEFERRED_EVENTS_ATTR = "project_memory_deferred_events"
MAX_SEEN_EVENTS = 4096
REGEX_SECONDS = 0.005
RECALL_SECONDS = 0.05
MAX_MATCH_TEXT = 65536
MAX_RECALLS = 8


@functools.lru_cache(maxsize=256)
def _compile_trigger(pattern: str):
    return regex.compile(pattern)


def _trigger_matches(pattern, text: str, deadline: float) -> bool:
    remaining = min(REGEX_SECONDS, deadline - time.monotonic())
    if remaining <= 0:
        return False
    try:
        return pattern.search(text[:MAX_MATCH_TEXT], timeout=remaining) is not None
    except TimeoutError:
        return False


MAX_BACKTEST_MATCHES = 12
MAX_BACKTEST_EXCERPT_CHARS = 600
PROJECT_SESSIONS_BUFFER_ID = "project_sessions"
SESSION_BUFFER_PREFIX = "ses_"
MIN_SESSION_PREFIX_LENGTH = 8
PROJECT_SESSION_RENDER_ATTR = "_project_session_memory_render"
PROJECT_SESSION_INDEX_RENDER_ATTR = "_project_session_memory_index_render"
CORE_SYSTEM_PROMPT = (
    "Use project memories for durable project-specific facts, decisions, constraints, "
    "and procedures that should be recalled in future sessions. Call record_memory with "
    "concise standalone content and a selective regular expression that matches future "
    "transcript messages indicating relevance. Refine noisy memories with update_memory "
    "or temporarily quiet them with suppress_memory; delete obsolete memories. Do not "
    "record secrets, transient progress, or facts already maintained in authoritative "
    "project files."
)
SESSION_SYSTEM_PROMPT = (
    "When a request appears to depend on work from an earlier project session and the "
    "necessary context is not already available, view project_sessions. It indexes prior "
    "sessions by recent activity and lists a ses_ buffer for each; use the titles, "
    "descriptions, and timestamps to choose the most relevant session or sessions. "
    "Opening a listed ses_ buffer produces a compact reverse-chronological projection "
    "containing the user message and final assistant response from each turn, with newer "
    "turns first and tool activity omitted. Read from the beginning through a substantial "
    "contiguous sequence of turns until you have enough context; do not stop at the first "
    "turn or treat it as a summary. If a recent turn depends on an older decision, "
    "continue downward into earlier turns. For an unusually long buffer, use grep only to "
    "locate a relevant area, then read the surrounding turn blocks. Use the Source "
    "Transcript path and entry/line pointers when omitted tool activity or full-session "
    "detail is needed."
)
DEFAULT_SYSTEM_PROMPT = f"{CORE_SYSTEM_PROMPT} {SESSION_SYSTEM_PROMPT}"


def _config_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return value is not False



def _register_exact_special_buffer(manager, buffer_id: str, provider: Any) -> None:
    """Replace one exact provider using the current BufferManager contract."""
    manager.register_special_buffer(buffer_id, provider, replace=True)


def _format_utc(value: Any) -> str:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return "(unknown)"
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                return raw
        else:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        parsed = datetime.fromtimestamp(float(value), timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return "(unknown)"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _one_line_text(value: Any, fallback: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text or fallback


def _session_key(session_id: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(session_id or "").lower())


# These caps are deliberately small: session history is an observational aid,
# not a second persistence system.  A single daemon drains the bounded queue.
MAX_PROJECT_SESSION_ROWS = 1000
SESSION_LOAD_QUEUE_LIMIT = 8
SESSION_PROJECTION_CACHE_LIMIT = 16
SESSION_BUFFER_RECORD_LIMIT = 32
SESSION_ERROR_CHARS = 300
SESSION_INDEX_REFRESH_S = 5.0
SESSION_LOAD_RETRY_S = 5.0


def _session_buffer_ids(sessions: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    keyed = {
        str(item.get("session_id") or ""): _session_key(str(item.get("session_id") or ""))
        for item in sessions
    }
    result: dict[str, str] = {}
    for session_id, key in keyed.items():
        if not session_id or len(key) < MIN_SESSION_PREFIX_LENGTH:
            continue
        length = MIN_SESSION_PREFIX_LENGTH
        while length < len(key):
            prefix = key[:length]
            if sum(other.startswith(prefix) for other in keyed.values()) == 1:
                break
            length += 1
        result[session_id] = SESSION_BUFFER_PREFIX + key[:length]
    return result


def _session_source(project_dir: Path, session_id: str) -> Path:
    """Return the logical checkpoint locator without touching shared storage."""
    if (
        not session_id
        or "\x00" in session_id
        or Path(session_id).name != session_id
        or session_id in {".", ".."}
    ):
        raise ValueError(f"Invalid session id: {session_id!r}")
    # ``abspath`` is lexical here; unlike ``resolve`` it does not inspect a
    # shared mount or follow a session directory while the pipeline is running.
    return Path(os.path.abspath(os.fspath(Path(project_dir) / "sessions" / session_id / "session.json")))


def _timestamp_sort_value(value: Any) -> float:
    if isinstance(value, str):
        raw = value.strip()
        try:
            return float(raw)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return 0.0
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    try:
        return float(value or 0.0)
    except (TypeError, ValueError, OSError, OverflowError):
        return 0.0


def _eligible_project_sessions(
    entries: Sequence[Mapping[str, Any]],
    current_session_id: str,
) -> list[dict[str, Any]]:
    """Filter only catalog metadata; never probe one shared row at a time."""
    sessions: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = str(current_session_id or "").strip()
    for raw in entries:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        session_id = str(item.get("session_id") or "").strip()
        kind = str(item.get("kind") or "").strip().lower()
        if (
            not session_id
            or "\x00" in session_id
            or Path(session_id).name != session_id
            or session_id in seen
            or session_id == current
            or kind.startswith("rlm")
        ):
            continue
        item["session_id"] = session_id
        # Old rows may contain a mutable source path.  It is not inventory
        # metadata and must never become the authority for a history buffer.
        item.pop("source_transcript", None)
        seen.add(session_id)
        sessions.append(item)
    sessions.sort(
        key=lambda item: (_timestamp_sort_value(item.get("updated_at")), str(item["session_id"])),
        reverse=True,
    )
    return sessions[:MAX_PROJECT_SESSION_ROWS]


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content"):
            if key in value:
                text = _content_text(value.get(key))
                if text:
                    return text
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(filter(None, (_content_text(item) for item in value)))
    return "" if value is None else str(value)


def _entry_line_ranges(source_text: str) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    in_entries = False
    start_line: int | None = None
    entry_index: int | None = None
    index_pattern = re.compile(r'^\s{6}"index":\s*(-?\d+),?\s*$')
    for line_number, line in enumerate(source_text.splitlines(), start=1):
        stripped = line.rstrip("\r\n")
        if not in_entries:
            if stripped == '  "entries": [':
                in_entries = True
            continue
        if start_line is None:
            if stripped == "    {":
                start_line = line_number
                entry_index = None
                continue
            if stripped == "  ],":
                break
            continue
        match = index_pattern.match(stripped)
        if match:
            entry_index = int(match.group(1))
        if stripped in {"    },", "    }"}:
            if entry_index is not None:
                ranges[entry_index] = (start_line, line_number)
            start_line = None
            entry_index = None
    return ranges


def _compact_turns(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep complete user/final-assistant turn blocks, oldest-to-newest."""
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or bool(entry.get("system_generated", False)):
            continue
        try:
            entry_index = int(entry.get("index", position))
        except (TypeError, ValueError):
            entry_index = position
        messages = [
            message
            for message in (entry.get("messages") or [])
            if isinstance(message, Mapping)
        ]
        assistant_messages = [
            message for message in messages if str(message.get("role") or "") == "assistant"
        ]
        for message in messages:
            role = str(message.get("role") or "")
            text = _content_text(message.get("content")).strip()
            if role == "user" and text:
                if current is not None:
                    turns.append(current)
                current = {
                    "user_entry": entry_index,
                    "user_text": text,
                    "assistant_entry": None,
                    "assistant_text": "",
                }
                continue
            if role != "assistant" or current is None or not text:
                continue
            has_message_tool_calls = bool(
                message.get("tool_calls") or message.get("provider_tool_calls")
            )
            only_assistant_owns_entry_calls = (
                len(assistant_messages) == 1 and bool(entry.get("tool_calls"))
            )
            if has_message_tool_calls or only_assistant_owns_entry_calls:
                continue
            # Later plain assistant messages in the same turn are the final
            # answer for projection purposes.
            current["assistant_entry"] = entry_index
            current["assistant_text"] = text
    if current is not None:
        turns.append(current)
    return turns


def _entry_delimiter(role: str, entry_index: int, ranges: Mapping[int, tuple[int, int]]) -> str:
    source_range = ranges.get(entry_index)
    if source_range is None:
        return f"{role} [entry {entry_index}]"
    return f"{role} [entry {entry_index}, lines {source_range[0]}-{source_range[1]}]"


def _build_compact_transcript(
    session: Mapping[str, Any],
    source: Path,
    *,
    revision_id: str = "",
    generated_at: float,
    source_text: str | None = None,
) -> str:
    """Build a projection from an AU-validated immutable payload."""
    if source_text is None:
        source_text = source.read_text(encoding="utf-8")
    document = json.loads(source_text)
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list):
        state = document.get("state")
        raw_entries = state.get("entries") if isinstance(state, Mapping) else []
    entries = [item for item in (raw_entries or []) if isinstance(item, Mapping)]
    ranges = _entry_line_ranges(source_text)
    lines = [
        f"Session ID: {session['session_id']}",
        f"Title: {_one_line_text(session.get('title'), '(untitled)')}",
        f"Description: {_one_line_text(session.get('description'), '(none)')}",
        f"Created (UTC): {_format_utc(session.get('created_at'))}",
        f"Updated (UTC): {_format_utc(session.get('updated_at'))}",
        f"Buffer Generated (UTC): {_format_utc(generated_at)}",
        "Order: Newest turn first",
        f"Source Transcript: {source}",
        f"Source Revision: {revision_id or '(unknown immutable revision)'}",
    ]
    for turn in reversed(_compact_turns(entries)):
        lines.extend(
            [
                "",
                _entry_delimiter("USER", turn["user_entry"], ranges),
                turn["user_text"],
            ]
        )
        if turn["assistant_entry"] is not None:
            lines.extend(
                [
                    "",
                    _entry_delimiter("ASSISTANT", turn["assistant_entry"], ranges),
                    turn["assistant_text"],
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _projection_cache_key(session: Mapping[str, Any], revision_id: str) -> tuple[str, str, str]:
    metadata = {
        key: session.get(key)
        for key in ("title", "description", "created_at", "updated_at", "kind")
    }
    return (
        str(session.get("session_id") or ""),
        str(revision_id or ""),
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str),
    )


def _bounded_history_error(error: object) -> str:
    text = str(error or "").replace("\x00", " ").strip()
    if len(text) > SESSION_ERROR_CHARS:
        return text[:SESSION_ERROR_CHARS].rstrip() + "…"
    return text or "unknown history read failure"


class ProjectMemoryStore:
    """Project-database storage component shared by all memory components."""

    def __init__(
        self,
        common_memories: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.common_memories = {
            str(key): dict(value)
            for key, value in (common_memories or {}).items()
            if isinstance(value, Mapping)
        }

    @staticmethod
    def _common_memory_id(registry_key: str) -> str:
        digest = hashlib.sha256(
            f"{PLUGIN_NAME}:common:{registry_key}".encode("utf-8")
        ).hexdigest()[:6]
        return f"mem_{digest}"

    def list_common(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        memories = []
        for raw_key, spec in self.common_memories.items():
            registry_key = str(raw_key or "").strip()
            if not registry_key:
                log.warning("Skipping configured project memory with empty registry key")
                continue
            try:
                content = self._validate_content(spec.get("content"))
                trigger = self._validate_trigger(spec.get("trigger"))
            except (AttributeError, ValueError) as exc:
                log.warning(
                    "Skipping invalid configured project memory %s: %s",
                    registry_key or "<empty>",
                    exc,
                )
                continue
            enabled = _config_enabled(spec.get("enabled", True))
            if enabled_only and not enabled:
                continue
            memories.append(
                {
                    "id": self._common_memory_id(registry_key),
                    "content": content,
                    "trigger": trigger,
                    "registry_key": registry_key,
                    "source": "config",
                    "enabled": enabled,
                }
            )
        return memories

    def get_common(self, memory_id: str) -> dict[str, Any] | None:
        memory_id = str(memory_id or "").strip()
        return next(
            (memory for memory in self.list_common() if memory["id"] == memory_id),
            None,
        )

    optional_reads = {"run_db", "agent_db"}

    def __call__(self, state):
        # Startup and restore must not take a project-database write lock.
        # CRUD methods create the schema lazily when a mutation is requested.
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

    def _existing_schema(self, state):
        conn = self._agent_db(state).conn
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (TABLE_NAME,),
        ).fetchone()
        return conn if exists is not None else None

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
        memory_id = str(memory_id or "").strip()
        conn = self._existing_schema(state)
        if conn is None:
            return self.get_common(memory_id)
        row = conn.execute(
            f"SELECT id, content, trigger, created_at, updated_at "
            f"FROM {TABLE_NAME} WHERE id = ?",
            (memory_id,),
        ).fetchone()
        memory = self._memory_from_row(row)
        return memory if memory is not None else self.get_common(memory_id)

    def list_all(self, state) -> list[dict[str, Any]]:
        conn = self._existing_schema(state)
        if conn is None:
            return []
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
        reserved_ids = {memory["id"] for memory in self.list_common()}
        for _attempt in range(32):
            memory_id = f"mem_{secrets.token_hex(3)}"
            if memory_id in reserved_ids:
                continue
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
        if self.get_common(memory_id) is not None:
            raise ValueError(
                f"configured common memory {memory_id!r} is read-only; edit the YAML registry"
            )
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
        if self.get_common(memory_id) is not None:
            raise ValueError(
                f"configured common memory {memory_id!r} is read-only; edit the YAML registry"
            )
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

        # Keep the dynamically loaded plugin instance out of the manager's
        # provider table. Pipeline initialization recreates this runtime binding.
        setattr(state, "_project_memory_render", functools.partial(self.render, state))
        _register_exact_special_buffer(
            manager,
            "project_memory",
            operator.methodcaller("_project_memory_render"),
        )
        return state

    def render(self, state) -> ReadonlyBufferView:
        db_error = None
        try:
            memories = self.store.list_all(state)
        except Exception as exc:
            memories = []
            db_error = f"Project database unavailable: {type(exc).__name__}: {exc}"

        common_memories = self.store.list_common()
        lines = ["# Project memories", "", f"Count: {len(memories)}"]
        if db_error:
            lines.extend(["", db_error])
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

        if common_memories:
            lines.extend(
                [
                    "",
                    "# Common configured memories",
                    "",
                    f"Count: {len(common_memories)}",
                ]
            )
            for memory in common_memories:
                lines.extend(
                    [
                        "",
                        f"## {memory['id']}",
                        f"Registry key: {memory['registry_key']}",
                        f"Enabled: {'yes' if memory['enabled'] else 'no'}",
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


class ProjectSessionBuffers:
    """Expose previous project sessions through immutable revision projections."""

    reads = {"buffer_manager"}
    optional_reads = {"_agent_zoo_context", "_session_id", "_local_state_root", "run_db", "agent_db"}
    writes = {"buffer_manager"}

    def __init__(self, *, clock=time.time) -> None:
        self.clock = clock
        self._project_name = ""
        self._session_id = ""
        self._lock = threading.RLock()
        self._work: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(
            maxsize=SESSION_LOAD_QUEUE_LIMIT
        )
        self._worker: threading.Thread | None = None
        self._inventories: dict[str, dict[str, Any]] = {}
        self._buffer_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._session_results: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._pinned_revisions: dict[tuple[str, str], dict[str, str]] = {}
        self._projection_cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()

    def bind_session(
        self,
        session_id: str,
        *,
        project_name: str | None = None,
        db: Any | None = None,
    ) -> None:
        del db
        self._session_id = str(session_id or "")
        if project_name is not None:
            self._project_name = str(project_name or "default")

    def _project_key(self, project_name: str, project_dir: Path) -> str:
        return f"{project_name}\x00{os.path.abspath(os.fspath(project_dir))}"

    @staticmethod
    def _session_scope_key(project_key: str, buffer_id: str) -> tuple[str, str]:
        return project_key, str(buffer_id)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            worker = threading.Thread(
                target=self._worker_loop,
                name="azo-project-session-history",
                daemon=True,
            )
            self._worker = worker
            worker.start()

    def _enqueue(self, kind: str, payload: dict[str, Any]) -> bool:
        self._ensure_worker()
        try:
            self._work.put_nowait((kind, payload))
        except queue.Full:
            return False
        return True

    def _worker_loop(self) -> None:
        while True:
            kind, payload = self._work.get()
            try:
                if kind == "inventory":
                    self._read_inventory(payload)
                elif kind == "session":
                    self._read_session(payload)
                else:  # pragma: no cover - defensive queue integrity guard
                    log.warning("Unknown project-session history task %r", kind)
            except Exception:
                log.debug("Project-session history task failed", exc_info=True)
            finally:
                self._work.task_done()

    def _new_inventory(self) -> dict[str, Any]:
        return {
            "status": "pending",
            "sessions": [],
            "error": "initial metadata read is pending",
            "pending": False,
            "has_snapshot": False,
            "requested_at": 0.0,
            "completed_at": 0.0,
        }

    def _inventory_snapshot(self, key: str) -> dict[str, Any]:
        with self._lock:
            record = self._inventories.get(key)
            if record is None:
                record = self._new_inventory()
                self._inventories[key] = record
            return {
                **record,
                "sessions": [dict(item) for item in record.get("sessions", [])],
            }

    def _request_inventory(
        self,
        project_name: str,
        project_dir: Path,
        current_session_id: str,
    ) -> str:
        key = self._project_key(project_name, project_dir)
        now = float(self.clock())
        with self._lock:
            record = self._inventories.setdefault(key, self._new_inventory())
            if record.get("pending"):
                return key
            requested_at = float(record.get("requested_at") or 0.0)
            if record.get("has_snapshot") and now - requested_at < SESSION_INDEX_REFRESH_S:
                return key
            record["pending"] = True
            record["requested_at"] = now
            if not record.get("has_snapshot"):
                record["status"] = "pending"
                record["error"] = "initial metadata read is pending"

        payload = {
            "key": key,
            "project_name": project_name,
            "project_dir": project_dir,
            "current_session_id": current_session_id,
        }
        if self._enqueue("inventory", payload):
            return key
        with self._lock:
            record = self._inventories[key]
            record["pending"] = False
            record["status"] = "unavailable"
            record["error"] = "background history queue is full"
        return key

    @staticmethod
    def _read_catalog_snapshot(project_name: str, project_dir: Path):
        from agent_zoo import projects

        catalog_path = Path(
            os.path.abspath(os.fspath(Path(project_dir) / "sessions" / "index.json"))
        )
        data = projects._read_observational_json(catalog_path)
        if data is projects._MISSING_DOCUMENT:
            return projects.SessionIndexSnapshot(project_name, catalog_path, [], "absent")
        return projects.SessionIndexSnapshot(
            project_name,
            catalog_path,
            projects._validate_session_index(data, catalog_path),
            "present",
        )

    def _read_inventory(self, payload: Mapping[str, Any]) -> None:
        key = str(payload["key"])
        project_name = str(payload["project_name"])
        project_dir = Path(payload["project_dir"])
        current_session_id = str(payload.get("current_session_id") or "")
        try:
            # This is the one atomic catalog read.  It deliberately does not
            # inspect a session directory or checkpoint row for each entry.
            snapshot = self._read_catalog_snapshot(project_name, project_dir)
            sessions = _eligible_project_sessions(snapshot.entries, current_session_id)
            status = "absent" if snapshot.absent else "ready"
            error = "session catalog is absent" if snapshot.absent else ""
        except Exception as exc:
            with self._lock:
                record = self._inventories.setdefault(key, self._new_inventory())
                record["pending"] = False
                record["status"] = "unavailable"
                record["error"] = _bounded_history_error(exc)
                record["completed_at"] = float(self.clock())
            return

        with self._lock:
            record = self._inventories.setdefault(key, self._new_inventory())
            record["pending"] = False
            record["status"] = status
            record["error"] = error
            record["sessions"] = [dict(item) for item in sessions]
            record["has_snapshot"] = True
            record["completed_at"] = float(self.clock())
            buffer_ids = _session_buffer_ids(sessions)
            for session in sessions:
                session_id = str(session["session_id"])
                buffer_id = buffer_ids.get(session_id)
                if not buffer_id:
                    continue
                self._buffer_records[buffer_id] = {
                    "project_key": key,
                    "session": dict(session),
                }
                self._buffer_records.move_to_end(buffer_id)
            while len(self._buffer_records) > SESSION_BUFFER_RECORD_LIMIT:
                self._buffer_records.popitem(last=False)

    def _local_state_root(self, state: Any, context: Mapping[str, Any]) -> str | None:
        value = (
            context.get("local_state_root")
            or getattr(state, "_local_state_root", "")
            or getattr(state, "local_state_root", "")
        )
        text = str(value or "").strip()
        return text or None

    def __call__(self, state):
        manager = state.buffer_manager
        context = dict(getattr(state, "_agent_zoo_context", {}) or {})
        context_project_name = str(context.get("project_name") or "").strip()
        if context_project_name:
            self._project_name = context_project_name
        context_session_id = str(getattr(state, "_session_id", "") or context.get("session_id") or "").strip()
        if context_session_id:
            self._session_id = context_session_id

        setattr(state, PROJECT_SESSION_INDEX_RENDER_ATTR, functools.partial(self.render_index, state))
        setattr(state, PROJECT_SESSION_RENDER_ATTR, functools.partial(self.render_session, state))
        manager.register_special_buffer(
            PROJECT_SESSIONS_BUFFER_ID,
            operator.methodcaller(PROJECT_SESSION_INDEX_RENDER_ATTR),
            replace=True,
        )
        manager.register_special_buffer_namespace(
            SESSION_BUFFER_PREFIX,
            StateBoundSpecialBufferNamespaceProvider(PROJECT_SESSION_RENDER_ATTR),
            replace=True,
        )
        project_name, project_dir, current_session_id = self._runtime_context(state)
        if project_dir is not None:
            self._request_inventory(project_name, project_dir, current_session_id)
        return state

    def _runtime_context(self, state) -> tuple[str, Path | None, str]:
        context = dict(getattr(state, "_agent_zoo_context", {}) or {})
        project_name = self._project_name or str(context.get("project_name") or "default")
        current_session_id = str(
            getattr(state, "_session_id", "")
            or self._session_id
            or context.get("session_id")
            or ""
        )
        project_dir_raw = str(context.get("project_dir") or "").strip()
        if project_dir_raw:
            project_dir = Path(project_dir_raw)
        else:
            index_raw = str(context.get("project_session_index") or "").strip()
            if index_raw:
                project_dir = Path(index_raw).parent.parent
            else:
                run_db = getattr(state, "run_db", None)
                agent_db = getattr(run_db, "agent_db", None) or getattr(state, "agent_db", None)
                db_path = str(getattr(agent_db, "path", "") or "").strip()
                if not db_path:
                    return project_name, None, current_session_id
                project_dir = Path(db_path).parent
        return project_name, project_dir, current_session_id

    @staticmethod
    def _inventory_status_lines(record: Mapping[str, Any]) -> list[str]:
        status = str(record.get("status") or "pending")
        sessions = list(record.get("sessions", []) or [])
        if status == "pending":
            if sessions:
                return ["Inventory: refresh pending; showing the last good metadata snapshot."]
            return ["Inventory: pending; initial metadata read is still in progress."]
        if status == "unavailable":
            detail = _bounded_history_error(record.get("error"))
            if sessions:
                return [
                    f"Inventory: unavailable ({detail}); showing the last good metadata snapshot."
                ]
            return [f"Inventory: unavailable ({detail})."]
        if status == "absent":
            return ["Inventory: absent; no published session catalog is available."]
        return ["Inventory: ready."]

    def render_index(self, state) -> ReadonlyBufferView:
        project_name, project_dir, current_session_id = self._runtime_context(state)
        if project_dir is None:
            return ReadonlyBufferView(
                id=PROJECT_SESSIONS_BUFFER_ID,
                path="project-memory://sessions/pending",
                text=(
                    "Project Sessions\n"
                    f"Project: {project_name}\n"
                    "Inventory: pending; Agent Zoo project context is not attached yet.\n"
                ),
            )
        key = self._request_inventory(project_name, project_dir, current_session_id)
        record = self._inventory_snapshot(key)
        sessions = record["sessions"]
        buffer_ids = _session_buffer_ids(sessions)
        lines = [
            "Project Sessions",
            f"Project: {project_name}",
            f"Buffer Generated (UTC): {_format_utc(self.clock())}",
            "Order: Updated (UTC), newest first",
            "Current session omitted; RLM sessions excluded.",
            *self._inventory_status_lines(record),
            f"Sessions: {len(buffer_ids)}",
        ]
        listed = 0
        for session in sessions:
            session_id = str(session["session_id"])
            buffer_id = buffer_ids.get(session_id)
            if not buffer_id:
                continue
            listed += 1
            lines.extend(
                [
                    "",
                    (
                        f"{listed}. Buffer: {buffer_id} | Session ID: {session_id} | "
                        f"Updated (UTC): {_format_utc(session.get('updated_at'))} | "
                        f"Created (UTC): {_format_utc(session.get('created_at'))}"
                    ),
                    (
                        f"   Title: {_one_line_text(session.get('title'), '(untitled)')} | "
                        f"Description: {_one_line_text(session.get('description'), '(none)')}"
                    ),
                ]
            )
        return ReadonlyBufferView(
            id=PROJECT_SESSIONS_BUFFER_ID,
            path="project-memory://sessions",
            text="\n".join(lines).rstrip() + "\n",
        )

    @staticmethod
    def _status_view(buffer_id: str, status: str, detail: str) -> ReadonlyBufferView:
        label = "pending" if status == "pending" else "unavailable"
        return ReadonlyBufferView(
            id=buffer_id,
            path=f"project-memory://sessions/{buffer_id}/{label}",
            text=(
                f"Session buffer {buffer_id}: {label}.\n"
                f"{detail}\n"
            ),
        )

    def _queue_session_load(
        self,
        *,
        project_key: str,
        project_name: str,
        project_dir: Path,
        buffer_id: str,
        session: Mapping[str, Any],
        local_state_root: str | None,
        revision_id: str | None,
    ) -> bool:
        session_id = str(session["session_id"])
        scope_key = self._session_scope_key(project_key, buffer_id)
        now = float(self.clock())
        with self._lock:
            binding = self._pinned_revisions.get(scope_key)
            if binding is not None:
                if binding["session_id"] != session_id:
                    self._session_results[scope_key] = {
                        "status": "unavailable",
                        "error": (
                            f"buffer is pinned to session {binding['session_id']}; "
                            f"catalog now maps it to {session_id}"
                        ),
                        "project_key": project_key,
                        "session_id": binding["session_id"],
                        "retry_at": None,
                    }
                    return False
                revision_id = binding["revision_id"]
            elif revision_id is None and len(self._pinned_revisions) >= SESSION_BUFFER_RECORD_LIMIT:
                self._session_results[scope_key] = {
                    "status": "unavailable",
                    "error": "session history pin capacity reached; refusing to retarget a buffer",
                    "project_key": project_key,
                    "session_id": session_id,
                    "retry_at": None,
                }
                return False
            self._session_results[scope_key] = {
                "status": "pending",
                "requested_at": now,
                "project_key": project_key,
                "session_id": session_id,
            }
            self._session_results.move_to_end(scope_key)
        payload = {
            "project_key": project_key,
            "project_name": project_name,
            "project_dir": project_dir,
            "buffer_id": buffer_id,
            "session": dict(session),
            "local_state_root": local_state_root,
            "revision_id": revision_id,
        }
        if self._enqueue("session", payload):
            return True
        with self._lock:
            self._session_results[scope_key] = {
                "status": "unavailable",
                "error": "background history queue is full",
                "project_key": project_key,
                "session_id": session_id,
                "retry_at": now + SESSION_LOAD_RETRY_S,
            }
        return False

    def _read_session(self, payload: Mapping[str, Any]) -> None:
        project_key = str(payload["project_key"])
        buffer_id = str(payload["buffer_id"])
        scope_key = self._session_scope_key(project_key, buffer_id)
        session = dict(payload["session"])
        session_id = str(session["session_id"])
        revision_id = str(payload.get("revision_id") or "").strip() or None
        try:
            from agent_zoo.checkpoint_sync import PendingCheckpointLoad, load_checkpoint

            locator = _session_source(Path(payload["project_dir"]), session_id)
            loaded = load_checkpoint(
                locator,
                session_id=session_id,
                revision_id=revision_id,
                local_state_root=payload.get("local_state_root"),
                # The worker owns the wait.  A zero observer budget prevents a
                # caller-facing view from blocking on shared storage.
                shared_timeout_s=0.0,
            )
            if isinstance(loaded, PendingCheckpointLoad):
                loaded = loaded.wait(timeout_s=None)
            loaded_revision = getattr(loaded, "revision", None)
            actual_revision_id = str(getattr(loaded_revision, "revision_id", "") or "").strip()
            payload_path = Path(getattr(loaded, "payload_path", ""))
            if not actual_revision_id or not payload_path.name:
                raise RuntimeError("checkpoint loader returned no immutable revision payload")
            if revision_id is not None and actual_revision_id != revision_id:
                raise RuntimeError(
                    f"checkpoint loader selected revision {actual_revision_id!r}, expected {revision_id!r}"
                )

            cache_key = _projection_cache_key(session, actual_revision_id)
            with self._lock:
                text = self._projection_cache.get(cache_key)
                if text is not None:
                    self._projection_cache.move_to_end(cache_key)
            if text is None:
                # load_checkpoint has already run the real AU revision validator,
                # including session.json/blob references.  This read is only the
                # immutable payload used for the compact projection.
                source_text = payload_path.read_text(encoding="utf-8")
                text = _build_compact_transcript(
                    session,
                    payload_path,
                    revision_id=actual_revision_id,
                    generated_at=self.clock(),
                    source_text=source_text,
                )
                with self._lock:
                    self._projection_cache[cache_key] = text
                    self._projection_cache.move_to_end(cache_key)
                    while len(self._projection_cache) > SESSION_PROJECTION_CACHE_LIMIT:
                        self._projection_cache.popitem(last=False)

            with self._lock:
                binding = self._pinned_revisions.get(scope_key)
                if binding is not None:
                    if binding["session_id"] != session_id:
                        raise RuntimeError(
                            f"buffer is pinned to session {binding['session_id']}; "
                            f"loaded {session_id} instead"
                        )
                    if binding["revision_id"] != actual_revision_id:
                        raise RuntimeError(
                            f"buffer is pinned to revision {binding['revision_id']}; "
                            f"loaded {actual_revision_id} instead"
                        )
                else:
                    if len(self._pinned_revisions) >= SESSION_BUFFER_RECORD_LIMIT:
                        raise RuntimeError(
                            "session history pin capacity reached; refusing to retarget a buffer"
                        )
                    self._pinned_revisions[scope_key] = {
                        "revision_id": actual_revision_id,
                        "session_id": session_id,
                    }
                self._session_results[scope_key] = {
                    "status": "ready",
                    "text": text,
                    "revision_id": actual_revision_id,
                    "project_key": project_key,
                    "session_id": session_id,
                    "cache_key": cache_key,
                }
                self._session_results.move_to_end(scope_key)
                while len(self._session_results) > SESSION_BUFFER_RECORD_LIMIT:
                    evictable = next(
                        (
                            key
                            for key in self._session_results
                            if key not in self._pinned_revisions
                        ),
                        None,
                    )
                    if evictable is None:
                        break
                    self._session_results.pop(evictable, None)
        except Exception as exc:
            error = _bounded_history_error(exc)
            retry_at = (
                None
                if "pin capacity" in error or "buffer is pinned" in error
                else float(self.clock()) + SESSION_LOAD_RETRY_S
            )
            with self._lock:
                self._session_results[scope_key] = {
                    "status": "unavailable",
                    "error": error,
                    "project_key": project_key,
                    "session_id": session_id,
                    "retry_at": retry_at,
                }
                self._session_results.move_to_end(scope_key)

    def render_session(self, state, buffer_id: str) -> ReadonlyBufferView | None:
        requested = str(buffer_id or "").strip().lower()
        if not re.fullmatch(r"ses_[0-9a-z]{8,}", requested):
            return None
        project_name, project_dir, current_session_id = self._runtime_context(state)
        if project_dir is None:
            return self._status_view(
                requested,
                "pending",
                "Agent Zoo project context is not attached yet; retry this buffer after startup.",
            )
        project_key = self._request_inventory(project_name, project_dir, current_session_id)
        record = self._inventory_snapshot(project_key)
        sessions = record["sessions"]
        buffer_ids = _session_buffer_ids(sessions)
        session = next(
            (
                item
                for item in sessions
                if buffer_ids.get(str(item["session_id"])) == requested
            ),
            None,
        )
        if session is None:
            with self._lock:
                remembered = self._buffer_records.get(requested)
                if remembered is not None and remembered.get("project_key") == project_key:
                    session = dict(remembered.get("session") or {})
                    self._buffer_records.move_to_end(requested)
        if session is None:
            if record["status"] in {"pending", "unavailable"}:
                detail = (
                    "The metadata inventory has not completed yet. Re-read this buffer shortly."
                    if record["status"] == "pending"
                    else f"The metadata inventory failed: {_bounded_history_error(record.get('error'))}"
                )
                return self._status_view(requested, record["status"], detail)
            return None

        scope_key = self._session_scope_key(project_key, requested)
        pinned = None
        binding = None
        result = None
        with self._lock:
            binding = self._pinned_revisions.get(scope_key)
            result = self._session_results.get(scope_key)
            if binding is not None:
                pinned = binding["revision_id"]
                if binding["session_id"] != str(session["session_id"]):
                    return self._status_view(
                        requested,
                        "unavailable",
                        (
                            f"This buffer is pinned to session {binding['session_id']}; "
                            f"the catalog now maps it to {session['session_id']}."
                        ),
                    )
        if isinstance(result, Mapping) and result.get("status") == "ready":
            if (
                result.get("project_key") == project_key
                and result.get("session_id") == str(session["session_id"])
            ):
                return ReadonlyBufferView(
                    id=requested,
                    path=(
                        f"project-memory://sessions/{session['session_id']}"
                        f"/revisions/{result['revision_id']}"
                    ),
                    # Do not rebuild headers from a newer catalog row: a pinned
                    # projection is the exact text first returned to the caller.
                    text=str(result["text"]),
                )
        if isinstance(result, Mapping) and result.get("status") == "pending":
            return self._status_view(
                requested,
                "pending",
                "The immutable revision is loading in the bounded background worker.",
            )
        if isinstance(result, Mapping) and result.get("status") == "unavailable":
            error = _bounded_history_error(result.get("error"))
            retry_at = result.get("retry_at")
            if "pin capacity" in error or "buffer is pinned" in error:
                return self._status_view(requested, "unavailable", error)
            if retry_at is not None and float(self.clock()) < float(retry_at):
                return self._status_view(
                    requested,
                    "unavailable",
                    f"The immutable revision could not be loaded: {error}",
                )

        queued = self._queue_session_load(
            project_key=project_key,
            project_name=project_name,
            project_dir=project_dir,
            buffer_id=requested,
            session=session,
            local_state_root=self._local_state_root(
                state,
                dict(getattr(state, "_agent_zoo_context", {}) or {}),
            ),
            revision_id=pinned,
        )
        if not queued:
            with self._lock:
                failed = self._session_results.get(scope_key)
            if isinstance(failed, Mapping) and failed.get("status") == "unavailable":
                return self._status_view(
                    requested,
                    "unavailable",
                    _bounded_history_error(failed.get("error")),
                )
        return self._status_view(
            requested,
            "pending",
            "The immutable preferred revision is queued for local-first loading.",
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

    reads = {"entries", "step"}
    optional_reads = {"run_db", "agent_db", "context_compaction"}

    def __init__(self, store: ProjectMemoryStore, recall: "ProjectMemoryRecall") -> None:
        super().__init__(store)
        self.recall = recall

    @staticmethod
    def fn(content: str, trigger: str) -> str:
        """Save durable project knowledge for later regex-triggered recall.

        Args:
            content: Standalone text that should be recalled in a relevant future context.
            trigger: Python regular expression matched against future transcript messages.
        """

    def execute(self, state, *, content: str, trigger: str) -> str:
        memory = self.store.create(state, content, trigger)
        current_call_ids = {
            f"tool-call:{getattr(tool_call, 'id', '')}"
            for tool_call in getattr(state, "pending_tool_calls", ())
            if getattr(tool_call, "name", "") == self.name and getattr(tool_call, "id", None)
        }
        matches, truncated = self.recall.backtest(
            state,
            trigger,
            exclude_event_ids=current_call_ids,
        )
        result = self._format(memory, "Recorded")
        if not matches:
            return result + "\nBacktest: no messages or tool calls matched in the active context window."

        count = len(matches)
        unit = "event" if count == 1 else "events"
        suffix = (
            f"; showing the most recent {MAX_BACKTEST_MATCHES}"
            if truncated
            else ""
        )
        lines = [
            result,
            f"Backtest: trigger matched {count} {unit} in the active context window"
            f" (most recent first{suffix}):",
        ]
        for match in matches:
            excerpt = " ".join(str(match["text"]).strip().split())
            if len(excerpt) > MAX_BACKTEST_EXCERPT_CHARS:
                excerpt = excerpt[:MAX_BACKTEST_EXCERPT_CHARS].rstrip() + "..."
            lines.append(f"- {match['label']}: {excerpt}")
        return "\n".join(lines)


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
    optional_reads = {
        "run_db",
        "agent_db",
        "context_compaction",
        "_rlm_restore_ready",
    }
    writes = {
        SEEN_EVENTS_ATTR,
        PENDING_RECALLS_ATTR,
        SUPPRESSIONS_ATTR,
        BOOTSTRAPPED_ATTR,
        LAST_ENTRY_COUNT_ATTR,
        DEFERRED_EVENTS_ATTR,
    }
    init = {
        SEEN_EVENTS_ATTR: list,
        PENDING_RECALLS_ATTR: list,
        SUPPRESSIONS_ATTR: dict,
        BOOTSTRAPPED_ATTR: bool,
        LAST_ENTRY_COUNT_ATTR: int,
        DEFERRED_EVENTS_ATTR: list,
    }

    def __init__(self, store: ProjectMemoryStore) -> None:
        self.store = store
        self._restore_deferred = False

    def __call__(self, state):
        if getattr(state, "_rlm_restore_ready", True) is False:
            self._restore_deferred = True
            return state
        entries = list(getattr(state, "entries", []) or [])
        if self._restore_deferred:
            self._restore_deferred = False
            self._new_events(state, entries, baseline_only=True)
            return state
        events = self._new_events(state, entries)
        deferred = getattr(state, DEFERRED_EVENTS_ATTR, [])
        events = list(dict((str(key), str(text)[:MAX_MATCH_TEXT])
                           for key, text in [*deferred, *events]).items())[-32:]
        try:
            memories = self.store.list_all(state)
            snapshot_ready = self.store.snapshot_status()["ready"]
        except Exception:
            log.debug("project memory snapshot unavailable during recall", exc_info=True)
            memories, snapshot_ready = [], False
        setattr(state, DEFERRED_EVENTS_ATTR, [] if snapshot_ready else events)
        if not events:
            return state
        memories.extend(self.store.list_common(enabled_only=True))
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

        deadline = time.monotonic() + RECALL_SECONDS
        for memory in memories:
            if time.monotonic() >= deadline or len(pending) >= MAX_RECALLS:
                break
            memory_id = memory["id"]
            if memory_id in queued_ids or memory_id in suppressions:
                continue
            content = str(memory.get("content") or "").strip()
            if not content or self._content_is_active(content, active_text):
                continue
            try:
                trigger = _compile_trigger(str(memory.get("trigger") or ""))
            except regex.error:
                log.warning("Skipping project memory %s with invalid stored regex", memory_id)
                continue
            if not any(_trigger_matches(trigger, text, deadline) for text in event_texts):
                continue
            pending.append({"id": memory_id, "content": content})
            queued_ids.add(memory_id)
        return state

    def _new_events(
        self,
        state,
        entries,
        *,
        baseline_only: bool = False,
    ) -> list[tuple[str, str]]:
        seen_order = [str(item) for item in (getattr(state, SEEN_EVENTS_ATTR) or [])]
        seen = set(seen_order)
        current_ids: list[str] = []
        candidates: list[tuple[str, str]] = []
        bootstrapped = bool(getattr(state, BOOTSTRAPPED_ATTR, False))
        current_step = self._int_value(getattr(state, "step", 0), 0)

        previous_count = self._int_value(getattr(state, LAST_ENTRY_COUNT_ATTR, 0), 0)
        # Revisit the mutable tail for tool completion, not the entire saved history.
        # A truncation/replacement is a new baseline, not thousands of new events.
        reset = len(entries) < previous_count
        start = max(0, previous_count - 2) if bootstrapped and not reset else 0
        if reset:
            bootstrapped = False
        for position in range(start, len(entries)):
            entry = entries[position]
            entry_step = self._int_value(self._get(entry, "step", -1), -1)
            for event_id, text in self._entry_events(entry, position):
                current_ids.append(event_id)
                if event_id in seen:
                    continue
                if not baseline_only and (bootstrapped or entry_step >= current_step):
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
            if content and role not in {"system", "tool"}:
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

    def backtest(
        self,
        state,
        trigger: str,
        *,
        exclude_event_ids: Sequence[str] = (),
    ) -> tuple[list[dict[str, str]], bool]:
        """Test a trigger against the active transcript context.

        Results are returned most-recent-first and are capped so a recording
        tool result cannot overwhelm the model context.  The optional event
        exclusion is used to avoid matching the active ``record_memory`` call
        against its own trigger argument.
        """
        pattern = _compile_trigger(str(trigger))
        entries = list(getattr(state, "entries", []) or [])
        start, handoff_text = self._active_context_window(state, entries)
        excluded = {str(event_id) for event_id in exclude_event_ids}
        candidates: list[tuple[str, str]] = []
        if handoff_text:
            candidates.append(("handoff document", handoff_text))

        for position in range(start, len(entries)):
            entry = entries[position]
            if bool(self._get(entry, "forgotten", False)):
                continue
            entry_index = self._int_value(self._get(entry, "index", position), position)
            for event_id, text in self._entry_events(entry, position):
                if event_id not in excluded and text:
                    candidates.append((f"entry {entry_index}", text))

        matches: list[dict[str, str]] = []
        truncated = False
        deadline = time.monotonic() + RECALL_SECONDS
        for label, text in reversed(candidates):
            remaining = min(REGEX_SECONDS, deadline - time.monotonic())
            if remaining <= 0:
                truncated = True
                break
            if len(text) > MAX_MATCH_TEXT:
                truncated = True
            try:
                matched = pattern.search(text[:MAX_MATCH_TEXT], timeout=remaining)
            except TimeoutError:
                truncated = True
                continue
            if not matched:
                continue
            if len(matches) >= MAX_BACKTEST_MATCHES:
                truncated = True
                break
            matches.append({"label": label, "text": text})
        return matches, truncated

    def _active_context_window(self, state, entries) -> tuple[int, str]:
        harness_position, handoff_text = self._harness_boundary(state, entries)
        provider_position = self._provider_boundary(entries)

        if provider_position is not None and (
            harness_position is None or provider_position >= harness_position
        ):
            return provider_position + 1, ""
        return harness_position or 0, handoff_text

    def _active_context_text(self, state, entries) -> str:
        start, handoff_text = self._active_context_window(state, entries)
        parts = [handoff_text] if handoff_text else []
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


class ProjectMemorySystemPrompt(RenderTransformRegistrar):
    """Register configured project-memory guidance on the model channel."""

    reads = {"entries"}
    transform_name = "project memory system prompt"

    def __init__(self, text: str) -> None:
        self.text = str(text or "").strip()

    def __call__(self, state):
        if not self.text or not state.entries:
            return state
        entry = state.entries[0]
        if self.transform_name in entry.render_transform_names(
            MODEL_RENDER_CHANNEL,
            pending=True,
        ):
            return state
        block = "# Project Memory\n" + self.text

        def transform(messages, _state, _block=block):
            if not messages:
                return messages
            first = dict(messages[0])
            first["content"] = ProjectMemorySystemPrompt._append_content(
                first.get("content"),
                _block,
            )
            return [first, *messages[1:]]

        entry.append_render_transform(
            MODEL_RENDER_CHANNEL,
            transform,
            name=self.transform_name,
        )
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

    @staticmethod
    def _normalize_common_memories(value: Any) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        if isinstance(value, Mapping):
            for key, spec in value.items():
                if isinstance(spec, Mapping):
                    normalized[str(key).strip()] = dict(spec)
            return {key: spec for key, spec in normalized.items() if key}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                key = str(item.get("key") or item.get("name") or "").strip()
                if not key:
                    continue
                spec = dict(item)
                spec.pop("key", None)
                spec.pop("name", None)
                normalized[key] = spec
        return normalized

    @classmethod
    def common_memories(cls, section: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        bundled_section = cls._load_bundled_config().get(CONFIG_SECTION, {})
        bundled = (
            bundled_section.get("common_memories")
            if isinstance(bundled_section, Mapping)
            else None
        )
        configured = section.get("common_memories") if isinstance(section, Mapping) else None
        include_bundled = (
            section.get("include_bundled_memories", True)
            if isinstance(section, Mapping)
            else True
        )
        result = {}
        if cls.enabled(include_bundled):
            result.update(cls._normalize_common_memories(bundled))
        result.update(cls._normalize_common_memories(configured))
        return result

    @classmethod
    def _load_bundled_config(cls) -> dict[str, Any]:
        path = Path(__file__).resolve().parents[1] / "config" / CONFIG_FILENAME
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, yaml.YAMLError):
            log.warning("Could not load bundled project-memory config", exc_info=True)
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @classmethod
    def effective_section(cls, config: Any) -> dict[str, Any]:
        installed = cls._load_installed_config().get(CONFIG_SECTION, {})
        explicit = (config or {}).get(CONFIG_SECTION, {})
        merged = dict(installed) if isinstance(installed, Mapping) else {}
        if isinstance(explicit, Mapping):
            for key, value in explicit.items():
                if key == "common_memories":
                    registry = cls._normalize_common_memories(merged.get(key))
                    registry.update(cls._normalize_common_memories(value))
                    merged[key] = registry
                else:
                    merged[key] = value
        return merged

    @staticmethod
    def enabled(value: Any) -> bool:
        return _config_enabled(value)

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

    session_config = section.get("project_sessions", True)
    if isinstance(session_config, Mapping):
        sessions_requested = ProjectMemorySystemPrompt.enabled(
            session_config.get("enabled", True)
        )
    else:
        sessions_requested = ProjectMemorySystemPrompt.enabled(session_config)
    sessions_enabled = sessions_requested

    prompt = str(section.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
    if not sessions_enabled:
        prompt = prompt.replace(SESSION_SYSTEM_PROMPT, "").strip()

    store = ProjectMemoryStore(ProjectMemorySystemPrompt.common_memories(section))
    recall = ProjectMemoryRecall(store)
    components = [
        store,
        ProjectMemoryBuffer(store),
        RecordMemory(store, recall),
        UpdateMemory(store),
        SuppressMemory(store),
        DeleteMemory(store),
        recall,
        ProjectMemoryDelivery(),
        ProjectMemorySystemPrompt(prompt),
    ]
    order = [
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
    ]
    if sessions_enabled:
        components.insert(2, ProjectSessionBuffers())
        order.insert(order.index(ToolDispatchStart), ProjectSessionBuffers)

    builder.add(
        Feature(
            name="project_memory",
            components=components,
            order=order,
        )
    )
