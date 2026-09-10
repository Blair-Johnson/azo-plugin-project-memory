"""Project-scoped regex memories for Agent Zoo pipelines."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import operator
import os
import re
import secrets
import sqlite3
import tempfile
import time
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
try:
    from agent_utils.files.buffer_manager import (
        BufferManager,
        ReadonlyBufferView,
        StateBoundSpecialBufferNamespaceProvider,
    )
except ImportError:  # pragma: no cover - compatibility error is raised at registration
    from agent_utils.files.buffer_manager import BufferManager, ReadonlyBufferView

    StateBoundSpecialBufferNamespaceProvider = None
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
COMPACT_CACHE_VERSION = 3
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




def _session_namespaces_supported() -> bool:
    if StateBoundSpecialBufferNamespaceProvider is None:
        return False
    namespace_register = getattr(BufferManager, "register_special_buffer_namespace", None)
    exact_register = getattr(BufferManager, "register_special_buffer", None)
    if not callable(namespace_register) or not callable(exact_register):
        return False
    try:
        return (
            "replace" in inspect.signature(namespace_register).parameters
            and "replace" in inspect.signature(exact_register).parameters
        )
    except (TypeError, ValueError):
        return False




def _register_exact_special_buffer(manager, buffer_id: str, provider: Any) -> None:
    register = manager.register_special_buffer
    try:
        supports_replace = "replace" in inspect.signature(register).parameters
    except (TypeError, ValueError):
        supports_replace = False
    if supports_replace:
        register(buffer_id, provider, replace=True)
        return
    if manager.is_special(buffer_id):
        providers = getattr(manager, "_special_buffers", None)
        if isinstance(providers, dict):
            providers[buffer_id] = provider
            cache = getattr(manager, "_special_cache", None)
            if isinstance(cache, dict):
                cache.pop(buffer_id, None)
            return
    register(buffer_id, provider)


_SESSION_NAMESPACE_WARNING_EMITTED = False


def _warn_session_namespaces_unavailable() -> None:
    global _SESSION_NAMESPACE_WARNING_EMITTED
    if _SESSION_NAMESPACE_WARNING_EMITTED:
        return
    _SESSION_NAMESPACE_WARNING_EMITTED = True
    log.warning(
        "Project session buffers are disabled: the loaded Agent Utils build lacks "
        "replaceable special-buffer namespace support. Core project-memory tools remain active."
    )


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


def _load_session_index(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [dict(item) for item in data if isinstance(item, Mapping)]


def _session_source(project_dir: Path, session_id: str) -> Path:
    if not session_id or Path(session_id).name != session_id or session_id in {".", ".."}:
        raise ValueError(f"Invalid session id: {session_id!r}")
    sessions_root = (project_dir / "sessions").resolve()
    source = (sessions_root / session_id / "session.json").resolve()
    try:
        source.relative_to(sessions_root)
    except ValueError as exc:
        raise ValueError(f"Session path escapes project: {session_id!r}") from exc
    return source


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
    project_dir: Path,
    current_session_id: str,
) -> list[dict[str, Any]]:
    sessions = []
    seen: set[str] = set()
    for item in _load_session_index(project_dir / "sessions" / "index.json"):
        session_id = str(item.get("session_id") or "").strip()
        kind = str(item.get("kind") or "").strip().lower()
        if (
            not session_id
            or session_id in seen
            or session_id == current_session_id
            or kind.startswith("rlm")
        ):
            continue
        seen.add(session_id)
        try:
            source = _session_source(project_dir, session_id)
        except ValueError:
            continue
        if not source.is_file():
            continue
        item["session_id"] = session_id
        item["source_transcript"] = str(source)
        sessions.append(item)
    sessions.sort(
        key=lambda item: _timestamp_sort_value(item.get("updated_at")),
        reverse=True,
    )
    return sessions


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


def _entry_has_tool_activity(entry: Mapping[str, Any]) -> bool:
    if entry.get("tool_calls"):
        return True
    for message in entry.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "") == "tool":
            return True
        if message.get("tool_calls") or message.get("provider_tool_calls"):
            return True
    return False


def _role_text(entry: Mapping[str, Any], role: str) -> str:
    parts = []
    for message in entry.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "") != role:
            continue
        text = _content_text(message.get("content")).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _compact_turns(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
    generated_at: float,
    source_text: str | None = None,
) -> str:
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
        f"Source Transcript: {source.resolve()}",
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


def _cache_path(project_dir: Path, session_id: str) -> Path:
    raw_id = str(session_id or "")
    readable = _session_key(raw_id)[:24] or "session"
    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
    return (
        project_dir
        / "plugin-data"
        / "project-memory"
        / "compact"
        / f"{readable}-{digest}.json"
    )


def _source_stat_key(source_stat: os.stat_result) -> tuple[int, int, int, int]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
    )


def _read_source_snapshot(source: Path) -> tuple[str, os.stat_result]:
    for _ in range(3):
        before = source.stat()
        source_text = source.read_text(encoding="utf-8")
        after = source.stat()
        if _source_stat_key(before) == _source_stat_key(after):
            return source_text, after
    raise OSError(f"Session transcript changed repeatedly while reading: {source}")


def _session_cache_metadata(
    session: Mapping[str, Any],
    source: Path,
    source_stat: os.stat_result,
) -> dict[str, Any]:
    return {
        "version": COMPACT_CACHE_VERSION,
        "source": str(source.resolve()),
        "source_device": source_stat.st_dev,
        "source_inode": source_stat.st_ino,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "title": str(session.get("title") or ""),
        "description": str(session.get("description") or ""),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
    }


def _read_compact_cache(path: Path, expected: Mapping[str, Any]) -> str | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cached, Mapping) or cached.get("metadata") != dict(expected):
        return None
    text = cached.get("text")
    return text if isinstance(text, str) else None


def _atomic_write_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _cached_compact_transcript(
    project_dir: Path,
    session: Mapping[str, Any],
    source: Path,
    *,
    generated_at: float,
) -> str:
    cache = _cache_path(project_dir, str(session["session_id"]))
    before = source.stat()
    cached = _read_compact_cache(
        cache,
        _session_cache_metadata(session, source, before),
    )
    after = source.stat()
    if cached is not None and _source_stat_key(before) == _source_stat_key(after):
        return cached

    source_text, source_stat = _read_source_snapshot(source)
    metadata = _session_cache_metadata(session, source, source_stat)
    text = _build_compact_transcript(
        session,
        source,
        generated_at=generated_at,
        source_text=source_text,
    )
    try:
        _atomic_write_cache(cache, {"metadata": metadata, "text": text})
    except OSError:
        log.debug("Could not cache compact session transcript %s", source, exc_info=True)
    return text


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
    """Expose previous project sessions as compact readonly buffers."""

    reads = {"buffer_manager"}
    optional_reads = {"_agent_zoo_context", "_session_id", "run_db", "agent_db"}
    writes = {"buffer_manager"}

    def __init__(self, *, clock=time.time) -> None:
        self.clock = clock
        self._project_name = ""
        self._session_id = ""

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

    def __call__(self, state):
        if not _session_namespaces_supported():
            log.warning(
                "Project session buffers are unavailable because this Agent Utils "
                "build lacks special-buffer namespace support"
            )
            return state
        manager = state.buffer_manager
        context = dict(getattr(state, "_agent_zoo_context", {}) or {})
        if not self._project_name:
            self._project_name = str(context.get("project_name") or "")
        if not self._session_id:
            self._session_id = str(getattr(state, "_session_id", "") or "")

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
        return state

    def _runtime_context(self, state) -> tuple[str, Path, str]:
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
                    raise RuntimeError("Agent Zoo project directory is unavailable")
                project_dir = Path(db_path).parent
        return project_name, project_dir, current_session_id

    def render_index(self, state) -> ReadonlyBufferView:
        project_name, project_dir, current_session_id = self._runtime_context(state)
        sessions = _eligible_project_sessions(project_dir, current_session_id)
        buffer_ids = _session_buffer_ids(sessions)
        lines = [
            "Project Sessions",
            f"Project: {project_name}",
            f"Buffer Generated (UTC): {_format_utc(self.clock())}",
            "Order: Updated (UTC), newest first",
            "Current session omitted; RLM sessions excluded.",
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

    def render_session(self, state, buffer_id: str) -> ReadonlyBufferView | None:
        requested = str(buffer_id or "").strip().lower()
        if not re.fullmatch(r"ses_[0-9a-z]{8,}", requested):
            return None
        _, project_dir, current_session_id = self._runtime_context(state)
        sessions = _eligible_project_sessions(project_dir, current_session_id)
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
            return None
        session_id = str(session["session_id"])
        source = Path(str(session["source_transcript"]))
        text = _cached_compact_transcript(
            project_dir,
            session,
            source,
            generated_at=self.clock(),
        )
        return ReadonlyBufferView(
            id=requested,
            path=f"project-memory://sessions/{session_id}",
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
    sessions_enabled = sessions_requested and _session_namespaces_supported()
    if sessions_requested and not sessions_enabled:
        _warn_session_namespaces_unavailable()

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
