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
import sys
import tempfile
import threading
import time
import uuid
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
    """Project-scoped filesystem memory storage with local-first publication.

    The service-facing read path is deliberately cache-only.  Local CRUD writes
    a durable overlay and operation record first; a single daemon worker later
    refreshes the shared project store and publishes those operations with a
    per-memory revision check.  The worker never mutates pipeline state.
    """

    _SCHEMA_VERSION = 1
    _RECORD_NAMESPACE = "project_memory"
    _OVERLAY_NAMESPACE = "project_memory_overlay"
    _PENDING_NAMESPACE = "project_memory_pending"
    _SNAPSHOT_NAMESPACE = "project_memory_snapshot"
    _SNAPSHOT_META_KEY = "__meta__"
    _REFRESH_INTERVAL_SECONDS = 5.0
    _DEFAULT_LOCAL_RETRY_SECONDS = 1.0

    optional_reads = {"_agent_zoo_context"}

    def __init__(
        self,
        common_memories: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.common_memories = {
            str(key): dict(value)
            for key, value in (common_memories or {}).items()
            if isinstance(value, Mapping)
        }
        self._lock = threading.RLock()
        self._storage_lock = threading.RLock()
        self._initialization_lock = threading.RLock()
        self._local_store = None
        self._local_root: Path | None = None
        self._shared_root: Path | None = None
        self._project_name = ""
        self._runtime_key: tuple[str, str, str] | None = None
        self._generation = 0
        self._overlay: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._latest_operation: dict[str, str] = {}
        self._receipts: dict[str, dict[str, Any]] = {}
        self._shared_snapshot: dict[str, dict[str, Any]] = {}
        self._shared_ready = False
        self._shared_status = "not_ready"
        self._shared_error: str | None = None
        self._snapshot_generation = 0
        self._last_refresh_at: float | None = None
        self._refresh_requested = False
        self._last_refresh_request = 0.0
        self._retry_at = 0.0
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self._closed = False

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

    @staticmethod
    def _lexical_absolute(value: str | os.PathLike[str] | Path) -> Path:
        return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))

    @staticmethod
    def _default_local_root() -> Path:
        uid = getattr(os, "getuid", lambda: 0)()
        if sys.platform == "darwin":
            base = Path("/private/var/tmp")
        else:
            base = Path("/var/tmp") if os.name == "posix" else Path(tempfile.gettempdir())
        return base / f"agent-zoo-{uid}"

    def _runtime_paths(self, state) -> tuple[str, Path, Path, Path] | None:
        context = dict(getattr(state, "_agent_zoo_context", {}) or {})
        project_name = str(context.get("project_name") or "default").strip() or "default"
        project_dir_raw = str(context.get("project_dir") or "").strip()
        if not project_dir_raw:
            index_raw = str(context.get("project_session_index") or "").strip()
            if index_raw:
                project_dir_raw = str(Path(index_raw).parent.parent)
        if not project_dir_raw:
            return None
        project_dir = self._lexical_absolute(project_dir_raw)
        shared_root = project_dir / "plugin-data" / "project-memory"
        local_raw = os.environ.get("AGENT_ZOO_LOCAL_STATE_ROOT") or self._default_local_root()
        local_base = self._lexical_absolute(local_raw)
        digest = hashlib.sha256(str(shared_root).encode("utf-8")).hexdigest()
        local_root = local_base / "project-memory" / digest
        return project_name, project_dir, shared_root, local_root

    @staticmethod
    def _open_record_store(root: Path, *, create: bool):
        try:
            from tmux_pilot.fs_store import RecordStore
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise RuntimeError("filesystem memory storage requires tmux-pilot RecordStore") from exc
        return RecordStore(root, create=create)

    @staticmethod
    def _compile_trigger(pattern: str):
        return regex.compile(pattern)

    @classmethod
    def _validate_content(cls, content: str) -> str:
        text = str(content or "").strip()
        if not text:
            raise ValueError("memory content must not be empty")
        return text

    @classmethod
    def _validate_trigger(cls, trigger: str) -> str:
        pattern = str(trigger or "").strip()
        if not pattern:
            raise ValueError("memory trigger must not be empty")
        try:
            cls._compile_trigger(pattern)
        except Exception as exc:
            raise ValueError(f"invalid memory trigger regex: {exc}") from exc
        return pattern

    @classmethod
    def _normalize_record(
        cls,
        value: Mapping[str, Any],
        *,
        fallback_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("filesystem memory record must be an object")
        memory_id = str(value.get("id") or value.get("memory_id") or fallback_id or "").strip()
        if not memory_id:
            raise ValueError("filesystem memory record has no id")
        deleted = bool(value.get("deleted", False))
        content = str(value.get("content") or "")
        trigger = str(value.get("trigger") or "")
        if not deleted:
            if not content.strip():
                raise ValueError(f"filesystem memory {memory_id!r} has empty content")
            if not trigger.strip():
                raise ValueError(f"filesystem memory {memory_id!r} has empty trigger")
            cls._compile_trigger(trigger)
        try:
            revision = int(value.get("revision", 1))
            created_at = float(value.get("created_at"))
            updated_at = float(value.get("updated_at"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"filesystem memory {memory_id!r} has invalid timestamps or revision") from exc
        if revision < 1:
            raise ValueError(f"filesystem memory {memory_id!r} has invalid revision")
        operation_id = str(value.get("operation_id") or value.get("op_id") or "").strip()
        return {
            "schema_version": int(value.get("schema_version", cls._SCHEMA_VERSION)),
            "id": memory_id,
            "content": content,
            "trigger": trigger,
            "created_at": created_at,
            "updated_at": updated_at,
            "revision": revision,
            "deleted": deleted,
            "operation_id": operation_id,
        }

    @staticmethod
    def _public_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
        if bool(record.get("deleted", False)):
            return None
        return {
            "id": str(record["id"]),
            "content": str(record["content"]),
            "trigger": str(record["trigger"]),
            "created_at": float(record["created_at"]),
            "updated_at": float(record["updated_at"]),
        }

    @staticmethod
    def _record_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        fields = (
            "id",
            "content",
            "trigger",
            "created_at",
            "updated_at",
            "revision",
            "deleted",
            "operation_id",
        )
        return all(left.get(field) == right.get(field) for field in fields)

    @staticmethod
    def _bounded_error(exc: object) -> str:
        text = f"{type(exc).__name__}: {exc}".strip()
        return text[:400]

    def _persist_snapshot(self, snapshot: Mapping[str, Mapping[str, Any]]) -> None:
        with self._lock:
            local_store = self._local_store
            records = {str(key): dict(value) for key, value in snapshot.items()}
        if local_store is None:
            raise RuntimeError("project memory local storage is unavailable")
        with self._storage_lock:
            for memory_id, incoming in records.items():
                def merge(current, _incoming=incoming, _memory_id=memory_id):
                    if current is None:
                        return _incoming
                    if not isinstance(current, Mapping):
                        return current
                    try:
                        existing = self._normalize_record(current, fallback_id=_memory_id)
                    except (TypeError, ValueError):
                        return current
                    incoming_revision = int(_incoming.get("revision", 0) or 0)
                    existing_revision = int(existing.get("revision", 0) or 0)
                    if existing_revision > incoming_revision:
                        return existing
                    if existing_revision == incoming_revision and str(
                        existing.get("operation_id") or ""
                    ) != str(_incoming.get("operation_id") or ""):
                        return existing
                    return _incoming

                local_store.update(self._SNAPSHOT_NAMESPACE, memory_id, merge)
            local_store.put(
                self._SNAPSHOT_NAMESPACE,
                self._SNAPSHOT_META_KEY,
                {
                    "schema_version": self._SCHEMA_VERSION,
                    "kind": "last_good_snapshot",
                    "saved_at": time.time(),
                },
            )

    def _load_local_records(
        self,
        local_store,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        bool,
    ]:
        pending: dict[str, dict[str, Any]] = {}
        overlay: dict[str, dict[str, Any]] = {}
        snapshot: dict[str, dict[str, Any]] = {}
        snapshot_present = False
        for key, raw in local_store.items(self._SNAPSHOT_NAMESPACE):
            if key == self._SNAPSHOT_META_KEY:
                if isinstance(raw, Mapping) and raw.get("kind") == "last_good_snapshot":
                    snapshot_present = True
                continue
            # Accept the pre-marker shape only as an in-place upgrade aid for
            # local state created by this plugin before the per-record format.
            candidates = raw.get("records", ()) if isinstance(raw, Mapping) and isinstance(raw.get("records"), list) else (raw,)
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                try:
                    record = self._normalize_record(candidate, fallback_id=str(key))
                except (TypeError, ValueError):
                    log.warning("Ignoring malformed durable project-memory snapshot record")
                    continue
                snapshot_present = True
                snapshot[str(record["id"])] = record
        for key, raw in local_store.items(self._PENDING_NAMESPACE):
            if not isinstance(raw, Mapping):
                continue
            op_id = str(raw.get("operation_id") or key or "").strip()
            memory_id = str(raw.get("memory_id") or "").strip()
            desired = raw.get("record")
            if not op_id or not memory_id or not isinstance(desired, Mapping):
                continue
            try:
                record = self._normalize_record(desired, fallback_id=memory_id)
            except (TypeError, ValueError):
                log.warning("Ignoring malformed local project-memory operation %s", op_id)
                continue
            operation = dict(raw)
            operation["operation_id"] = op_id
            operation["memory_id"] = memory_id
            operation["record"] = record
            operation["status"] = str(raw.get("status") or "pending")
            pending[op_id] = operation
        for key, raw in local_store.items(self._OVERLAY_NAMESPACE):
            candidate = raw.get("record") if isinstance(raw, Mapping) else raw
            if not isinstance(candidate, Mapping):
                continue
            try:
                record = self._normalize_record(candidate, fallback_id=str(key))
            except (TypeError, ValueError):
                log.warning("Ignoring malformed local project-memory overlay %s", key)
                continue
            overlay[str(record["id"])] = record
        for operation in sorted(
            pending.values(),
            key=lambda item: (
                float(item.get("created_at", 0.0) or 0.0),
                str(item["operation_id"]),
            ),
        ):
            if operation.get("status") not in {"pending", "conflict"}:
                continue
            overlay[str(operation["memory_id"])] = dict(operation["record"])
        return pending, overlay, snapshot, snapshot_present

    def _set_unavailable(self, error: object) -> None:
        with self._lock:
            self._shared_status = "unavailable"
            self._shared_error = self._bounded_error(error)
            self._last_refresh_at = time.time()

    def _ensure_runtime(self, state) -> bool:
        paths = self._runtime_paths(state)
        if paths is None:
            self._set_unavailable(RuntimeError("Agent Zoo project context is unavailable"))
            return False
        project_name, _project_dir, shared_root, local_root = paths
        key = (project_name, str(shared_root), str(local_root))
        with self._lock:
            if self._runtime_key == key and self._local_store is not None:
                return True
        with self._initialization_lock:
            with self._lock:
                if self._runtime_key == key and self._local_store is not None:
                    return True
            try:
                local_store = self._open_record_store(local_root, create=True)
                pending, overlay, snapshot, snapshot_present = self._load_local_records(local_store)
            except Exception as exc:
                self._set_unavailable(exc)
                return False
            with self._lock:
                self._generation += 1
                self._runtime_key = key
                self._project_name = project_name
                self._shared_root = shared_root
                self._local_root = local_root
                self._local_store = local_store
                self._pending = pending
                self._overlay = overlay
                self._latest_operation = {}
                self._receipts = {}
                for operation in sorted(
                    pending.values(),
                    key=lambda item: (float(item.get("created_at", 0.0) or 0.0), str(item["operation_id"])),
                ):
                    memory_id = str(operation["memory_id"])
                    op_id = str(operation["operation_id"])
                    self._latest_operation[memory_id] = op_id
                    self._receipts[op_id] = self._receipt_for_operation(
                        operation,
                        status="conflict" if operation.get("status") == "conflict" else "local_queued",
                        error=operation.get("error"),
                        observed=operation.get("conflict_record"),
                    )
                self._shared_snapshot = snapshot
                self._shared_ready = bool(snapshot_present)
                self._shared_status = "cached" if snapshot_present else "not_ready"
                self._shared_error = None
                self._snapshot_generation = 1 if snapshot_present else 0
                self._last_refresh_at = None
                self._last_refresh_request = time.monotonic()
                self._refresh_requested = True
                self._wake.set()
                self._start_worker_locked(self._generation)
            return True

    def _start_worker_locked(self, generation: int) -> None:
        if self._closed:
            return
        if self._worker is not None and self._worker.is_alive():
            return
        worker = threading.Thread(
            target=self._worker_loop,
            args=(generation,),
            name="azo-project-memory-sync",
            daemon=True,
        )
        self._worker = worker
        worker.start()

    def _request_work(self, *, refresh: bool = False) -> None:
        with self._lock:
            if self._local_store is None or self._closed:
                return
            if refresh:
                now = time.monotonic()
                if (
                    not self._refresh_requested
                    and now - self._last_refresh_request >= self._REFRESH_INTERVAL_SECONDS
                ):
                    self._refresh_requested = True
                    self._last_refresh_request = now
            self._wake.set()
            self._start_worker_locked(self._generation)

    @property
    def worker_alive(self) -> bool:
        with self._lock:
            return bool(self._worker is not None and self._worker.is_alive())

    def close(self, timeout_s: float = 0.25) -> None:
        with self._lock:
            self._closed = True
            self._wake.set()
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, float(timeout_s)))

    def __call__(self, state):
        self._ensure_runtime(state)
        self._request_work(refresh=True)
        return state

    def snapshot_status(self, state=None) -> dict[str, Any]:
        if state is not None:
            self._ensure_runtime(state)
            self._request_work(refresh=True)
        with self._lock:
            conflicts = sorted(
                str(operation.get("memory_id") or "")
                for operation in self._pending.values()
                if operation.get("status") == "conflict"
            )
            return {
                "project": self._project_name,
                "ready": bool(self._shared_ready),
                "last_good": bool(self._shared_ready),
                "status": self._shared_status,
                "shared_status": self._shared_status,
                "shared_available": self._shared_status in {"ready", "empty"},
                "cache_only": True,
                "snapshot_generation": self._snapshot_generation,
                "last_refresh_at": self._last_refresh_at,
                "pending_operations": sum(
                    1 for operation in self._pending.values() if operation.get("status") == "pending"
                ),
                "conflicts": conflicts,
                "error": self._shared_error,
            }

    # Alias with a short name for recall/buffer integrations.
    status = snapshot_status

    def _worker_loop(self, generation: int) -> None:
        try:
            while True:
                with self._lock:
                    if generation != self._generation or self._closed:
                        return
                    refresh = self._refresh_requested
                    self._refresh_requested = False
                    operation_id = self._next_operation_locked()
                    pending_exists = any(
                        operation.get("status") == "pending"
                        for operation in self._pending.values()
                    )
                if refresh:
                    self._refresh_shared(generation)
                if operation_id is not None:
                    self._publish_operation(operation_id, generation)
                    continue
                with self._lock:
                    if generation != self._generation or self._closed:
                        return
                    pending_exists = any(
                        operation.get("status") == "pending"
                        for operation in self._pending.values()
                    )
                    refresh_pending = self._refresh_requested
                    retry_at = self._retry_at
                if not pending_exists and not refresh_pending:
                    return
                timeout = 0.5
                if retry_at > time.monotonic():
                    timeout = min(timeout, retry_at - time.monotonic())
                self._wake.wait(timeout=max(0.01, timeout))
                self._wake.clear()
        finally:
            with self._lock:
                if self._worker is threading.current_thread():
                    self._worker = None

    def _next_operation_locked(self) -> str | None:
        now = time.monotonic()
        if now < self._retry_at:
            return None
        ordered = sorted(
            self._pending.values(),
            key=lambda item: (float(item.get("created_at", 0.0) or 0.0), str(item["operation_id"])),
        )
        pending_ids = set(self._pending)
        for operation in ordered:
            if operation.get("status") != "pending":
                continue
            previous = str(operation.get("previous_operation_id") or "").strip()
            if previous and previous in pending_ids:
                continue
            return str(operation["operation_id"])
        return None

    def _retain_last_good_for_missing_shared(self, generation: int) -> bool:
        with self._lock:
            if generation != self._generation:
                return True
            if not self._shared_ready:
                return False
            self._shared_status = "unavailable"
            self._shared_error = "shared project-memory root is unavailable"
            self._last_refresh_at = time.time()
            return True

    def _refresh_shared(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._shared_root is None:
                return
            root = self._shared_root
        try:
            try:
                root.lstat()
            except FileNotFoundError:
                if self._retain_last_good_for_missing_shared(generation):
                    return
                snapshot: dict[str, dict[str, Any]] = {}
                shared_status = "empty"
            else:
                shared_store = self._open_record_store(root, create=False)
                snapshot = {}
                for key, raw in shared_store.items(self._RECORD_NAMESPACE):
                    record = self._normalize_record(raw, fallback_id=str(key))
                    snapshot[str(record["id"])] = record
                shared_status = "ready" if any(
                    not bool(record.get("deleted", False)) for record in snapshot.values()
                ) else "empty"
        except FileNotFoundError:
            if self._retain_last_good_for_missing_shared(generation):
                return
            snapshot = {}
            shared_status = "empty"
        except Exception as exc:
            with self._lock:
                if generation == self._generation:
                    self._shared_status = "unavailable"
                    self._shared_error = self._bounded_error(exc)
                    self._last_refresh_at = time.time()
            return

        with self._lock:
            if generation != self._generation:
                return
            prior_snapshot = {key: dict(value) for key, value in self._shared_snapshot.items()}
        merged_snapshot = dict(prior_snapshot)
        for memory_id, incoming in snapshot.items():
            existing = merged_snapshot.get(memory_id)
            if existing is None:
                merged_snapshot[memory_id] = dict(incoming)
                continue
            existing_revision = int(existing.get("revision", 0) or 0)
            incoming_revision = int(incoming.get("revision", 0) or 0)
            if existing_revision > incoming_revision:
                continue
            if existing_revision == incoming_revision and str(
                existing.get("operation_id") or ""
            ) != str(incoming.get("operation_id") or ""):
                continue
            merged_snapshot[memory_id] = dict(incoming)
        snapshot = merged_snapshot
        shared_status = "ready" if any(
            not bool(record.get("deleted", False)) for record in snapshot.values()
        ) else "empty"

        snapshot_error = None
        try:
            self._persist_snapshot(snapshot)
        except Exception as exc:
            snapshot_error = exc
            log.debug("could not persist project-memory last-good snapshot", exc_info=True)
        reconcile: list[str] = []
        with self._lock:
            if generation != self._generation:
                return
            self._shared_snapshot = snapshot
            self._shared_ready = True
            self._shared_status = shared_status
            self._shared_error = None if snapshot_error is None else self._bounded_error(snapshot_error)
            self._snapshot_generation += 1
            self._last_refresh_at = time.time()
            for op_id, operation in self._pending.items():
                remote = snapshot.get(str(operation.get("memory_id") or ""))
                desired = operation.get("record")
                if (
                    operation.get("status") in {"pending", "conflict"}
                    and isinstance(remote, Mapping)
                    and isinstance(desired, Mapping)
                    and self._record_equal(remote, desired)
                    and str(remote.get("operation_id") or "") == op_id
                ):
                    reconcile.append(op_id)
        for op_id in reconcile:
            self._finish_published(op_id, generation)

    def _receipt_for_operation(
        self,
        operation: Mapping[str, Any],
        *,
        status: str,
        error: object = None,
        observed: Mapping[str, Any] | None = None,
        shared_saved: bool = False,
    ) -> dict[str, Any]:
        return {
            "operation_id": str(operation.get("operation_id") or ""),
            "memory_id": str(operation.get("memory_id") or ""),
            "operation": str(operation.get("operation") or ""),
            "expected_revision": int(operation.get("expected_revision", 0) or 0),
            "expected_operation_id": str(operation.get("expected_operation_id") or ""),
            "revision": int((operation.get("record") or {}).get("revision", 0) or 0),
            "local_saved": True,
            "shared_saved": bool(shared_saved),
            "status": status,
            "observed_revision": (
                int(observed.get("revision", 0) or 0)
                if isinstance(observed, Mapping) and observed.get("revision") is not None
                else None
            ),
            "error": None if error is None else self._bounded_error(error),
        }

    def receipt(self, memory_id: str) -> dict[str, Any] | None:
        memory_id = str(memory_id or "").strip()
        with self._lock:
            op_id = self._latest_operation.get(memory_id)
            if op_id and op_id in self._receipts:
                return dict(self._receipts[op_id])
            candidates = [
                receipt
                for receipt in self._receipts.values()
                if receipt.get("memory_id") == memory_id
            ]
            if not candidates:
                return None
            return dict(candidates[-1])

    def receipt_text(self, memory_id: str) -> str:
        receipt = self.receipt(memory_id)
        if not receipt:
            return ""
        status = str(receipt.get("status") or "")
        if status == "shared":
            return "Persistence: shared."
        if status == "conflict":
            detail = receipt.get("error") or "shared revision changed; local edit retained"
            return f"Persistence: conflict; shared memory was not overwritten ({detail})."
        if status == "local_queued":
            detail = receipt.get("error")
            suffix = f" ({detail})" if detail else ""
            return f"Persistence: local queued; shared publication is pending{suffix}."
        if status == "unsaved":
            return f"Persistence: unsaved ({receipt.get('error') or 'local storage unavailable'})."
        return f"Persistence: {status or 'unknown'}."

    def _current_record_locked(self, memory_id: str) -> dict[str, Any] | None:
        overlay = self._overlay.get(memory_id)
        if overlay is not None:
            return dict(overlay)
        shared = self._shared_snapshot.get(memory_id)
        return None if shared is None else dict(shared)

    def _list_records_locked(self) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {
            key: dict(value) for key, value in self._shared_snapshot.items()
        }
        merged.update({key: dict(value) for key, value in self._overlay.items()})
        result = []
        for record in merged.values():
            public = self._public_record(record)
            if public is not None:
                result.append(public)
        result.sort(key=lambda item: (float(item.get("created_at", 0.0)), str(item["id"])))
        return result

    def get(self, state, memory_id: str) -> dict[str, Any] | None:
        memory_id = str(memory_id or "").strip()
        self._ensure_runtime(state)
        self._request_work(refresh=True)
        with self._lock:
            current = self._current_record_locked(memory_id)
            public = None if current is None else self._public_record(current)
        return public if public is not None else self.get_common(memory_id)

    def list_all(self, state) -> list[dict[str, Any]]:
        """Return the last-good shared snapshot plus local overlay, without shared I/O."""
        self._ensure_runtime(state)
        self._request_work(refresh=True)
        with self._lock:
            return self._list_records_locked()

    def _reserve_memory_id_locked(self) -> str:
        reserved_ids = {
            memory["id"] for memory in self.list_common()
        }
        reserved_ids.update(self._shared_snapshot)
        reserved_ids.update(self._overlay)
        for _attempt in range(64):
            memory_id = f"mem_{secrets.token_hex(3)}"
            if memory_id not in reserved_ids:
                return memory_id
        raise RuntimeError("could not allocate a unique project memory id")

    def _queue_operation(
        self,
        state,
        *,
        memory_id: str,
        operation_name: str,
        expected_revision: int,
        record: Mapping[str, Any],
        expected_operation_id: str = "",
        previous_operation_id: str = "",
        supersede_operation_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        if not self._ensure_runtime(state):
            raise RuntimeError("project memory local storage is unavailable")
        operation_id = f"op_{uuid.uuid4().hex}"
        desired = dict(record)
        desired["operation_id"] = operation_id
        desired = self._normalize_record(desired, fallback_id=memory_id)
        operation = {
            "schema_version": self._SCHEMA_VERSION,
            "operation_id": operation_id,
            "memory_id": memory_id,
            "operation": operation_name,
            "expected_revision": int(expected_revision),
            "expected_operation_id": str(expected_operation_id or ""),
            "previous_operation_id": str(previous_operation_id or ""),
            "created_at": time.time(),
            "status": "pending",
            "record": desired,
        }
        superseded = {str(item) for item in supersede_operation_ids if str(item)}
        with self._storage_lock:
            local_store = self._local_store
            if local_store is None:
                raise RuntimeError("project memory local storage is unavailable")
            local_store.put(self._PENDING_NAMESPACE, operation_id, operation)
            local_store.put(
                self._OVERLAY_NAMESPACE,
                memory_id,
                {"schema_version": self._SCHEMA_VERSION, "record": desired},
            )
            for old_id in superseded:
                old = self._pending.get(old_id)
                if old is not None:
                    old = dict(old)
                    old["status"] = "superseded"
                    old["superseded_by"] = operation_id
                    try:
                        local_store.put(self._PENDING_NAMESPACE, old_id, old)
                        local_store.delete(self._PENDING_NAMESPACE, old_id)
                    except Exception:
                        log.debug("could not clear superseded project-memory operation %s", old_id, exc_info=True)
        with self._lock:
            for old_id in superseded:
                old = self._pending.pop(old_id, None)
                if old is not None:
                    old_receipt = self._receipt_for_operation(old, status="superseded")
                    old_receipt["superseded_by"] = operation_id
                    self._receipts[old_id] = old_receipt
            self._pending[operation_id] = operation
            self._overlay[memory_id] = desired
            self._latest_operation[memory_id] = operation_id
            self._receipts[operation_id] = self._receipt_for_operation(
                operation, status="local_queued"
            )
            public = self._public_record(desired)
        self._request_work()
        self.invalidate_buffer(state)
        if public is None:
            return {"id": memory_id, "content": "", "trigger": "", "created_at": desired["created_at"], "updated_at": desired["updated_at"]}
        return public

    def create(self, state, content: str, trigger: str) -> dict[str, Any]:
        content = self._validate_content(content)
        trigger = self._validate_trigger(trigger)
        self._ensure_runtime(state)
        with self._lock:
            memory_id = self._reserve_memory_id_locked()
            now = time.time()
            record = {
                "id": memory_id,
                "content": content,
                "trigger": trigger,
                "created_at": now,
                "updated_at": now,
                "revision": 1,
                "deleted": False,
            }
        return self._queue_operation(
            state,
            memory_id=memory_id,
            operation_name="create",
            expected_revision=0,
            record=record,
        )

    def _conflict_resolution_base_locked(
        self,
        memory_id: str,
        current: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[str, ...]] | None:
        operation_id = str(current.get("operation_id") or "")
        operation = self._pending.get(operation_id)
        visited: set[str] = set()
        conflict_operation = None
        while operation is not None and operation_id not in visited:
            visited.add(operation_id)
            if operation.get("status") == "conflict":
                conflict_operation = operation
                break
            operation_id = str(operation.get("previous_operation_id") or "")
            if not operation_id:
                break
            operation = self._pending.get(operation_id)
        if conflict_operation is None:
            return None
        observed = conflict_operation.get("conflict_record")
        if not isinstance(observed, Mapping):
            raise RuntimeError(
                f"project memory {memory_id!r} has an unresolved conflict without observed shared state"
            )
        base = self._normalize_record(observed, fallback_id=memory_id)
        # A normal update/delete is the explicit resolution boundary.  Abandon
        # every local descendant for this memory, including a tombstone, before
        # publishing the new operation against the observed shared revision.
        supersede = tuple(
            str(item.get("operation_id") or "")
            for item in self._pending.values()
            if str(item.get("memory_id") or "") == memory_id
            and item.get("status") in {"pending", "conflict"}
            and str(item.get("operation_id") or "")
        )
        return base, supersede

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
        if content is None and trigger is None:
            raise ValueError("update_memory requires content and/or trigger")
        self._ensure_runtime(state)
        supersede: tuple[str, ...] = ()
        with self._lock:
            current = self._current_record_locked(memory_id)
            if current is None:
                raise KeyError(f"unknown project memory {memory_id!r}")
            resolution = self._conflict_resolution_base_locked(memory_id, current)
            if resolution is not None:
                current, supersede = resolution
            if bool(current.get("deleted", False)):
                raise KeyError(f"unknown project memory {memory_id!r}")
            next_content = current["content"] if content is None else self._validate_content(content)
            next_trigger = current["trigger"] if trigger is None else self._validate_trigger(trigger)
            now = time.time()
            expected = int(current.get("revision", 0) or 0)
            expected_operation_id = str(current.get("operation_id") or "")
            previous = ""
            if not supersede and expected_operation_id in self._pending:
                previous = expected_operation_id
            record = {
                "id": memory_id,
                "content": next_content,
                "trigger": next_trigger,
                "created_at": float(current["created_at"]),
                "updated_at": now,
                "revision": expected + 1,
                "deleted": False,
            }
        return self._queue_operation(
            state,
            memory_id=memory_id,
            operation_name="update",
            expected_revision=expected,
            record=record,
            expected_operation_id=expected_operation_id,
            previous_operation_id=previous,
            supersede_operation_ids=supersede,
        )

    def delete(self, state, memory_id: str) -> bool:
        memory_id = str(memory_id or "").strip()
        if self.get_common(memory_id) is not None:
            raise ValueError(
                f"configured common memory {memory_id!r} is read-only; edit the YAML registry"
            )
        self._ensure_runtime(state)
        supersede: tuple[str, ...] = ()
        with self._lock:
            current = self._current_record_locked(memory_id)
            if current is None:
                return False
            resolution = self._conflict_resolution_base_locked(memory_id, current)
            if resolution is not None:
                current, supersede = resolution
            if bool(current.get("deleted", False)):
                return False
            expected = int(current.get("revision", 0) or 0)
            expected_operation_id = str(current.get("operation_id") or "")
            previous = ""
            if not supersede and expected_operation_id in self._pending:
                previous = expected_operation_id
            record = {
                "id": memory_id,
                "content": str(current.get("content") or ""),
                "trigger": str(current.get("trigger") or ""),
                "created_at": float(current["created_at"]),
                "updated_at": time.time(),
                "revision": expected + 1,
                "deleted": True,
            }
        self._queue_operation(
            state,
            memory_id=memory_id,
            operation_name="delete",
            expected_revision=expected,
            record=record,
            expected_operation_id=expected_operation_id,
            previous_operation_id=previous,
            supersede_operation_ids=supersede,
        )
        return True

    def _publish_operation(self, operation_id: str, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            operation = self._pending.get(operation_id)
            root = self._shared_root
        if operation is None or root is None or operation.get("status") != "pending":
            return
        desired = dict(operation["record"])
        memory_id = str(operation["memory_id"])
        outcome: dict[str, Any] = {}

        def mutate(current):
            current_record = None
            if current is not None:
                current_record = self._normalize_record(current, fallback_id=memory_id)
            if current_record is not None and str(current_record.get("operation_id") or "") == operation_id:
                if self._record_equal(current_record, desired):
                    outcome["status"] = "published"
                else:
                    outcome["status"] = "conflict"
                    outcome["current"] = current_record
                return current_record
            expected = int(operation.get("expected_revision", 0) or 0)
            expected_operation_id = str(operation.get("expected_operation_id") or "")
            current_revision = 0 if current_record is None else int(current_record.get("revision", 0) or 0)
            current_operation_id = "" if current_record is None else str(current_record.get("operation_id") or "")
            if current_revision != expected or current_operation_id != expected_operation_id:
                outcome["status"] = "conflict"
                outcome["current"] = current_record
                return current_record
            outcome["status"] = "published"
            return desired

        try:
            shared_store = self._open_record_store(root, create=True)
            shared_store.update(self._RECORD_NAMESPACE, memory_id, mutate)
            if outcome.get("status") == "conflict":
                self._mark_conflict(operation_id, generation, outcome.get("current"))
            else:
                self._finish_published(operation_id, generation)
        except Exception as exc:
            verified, current = self._verify_shared_after_error(root, memory_id, desired)
            if verified:
                self._finish_published(operation_id, generation, cleanup_error=exc)
            elif isinstance(current, Mapping) and (
                int(current.get("revision", 0) or 0) != int(operation.get("expected_revision", 0) or 0)
                or str(current.get("operation_id") or "")
                != str(operation.get("expected_operation_id") or "")
            ):
                self._mark_conflict(operation_id, generation, current, error=exc)
            else:
                with self._lock:
                    self._retry_at = time.monotonic() + self._DEFAULT_LOCAL_RETRY_SECONDS
                self._mark_unavailable(operation_id, generation, exc)

    def _verify_shared_after_error(
        self,
        root: Path,
        memory_id: str,
        desired: Mapping[str, Any],
    ) -> tuple[bool, dict[str, Any] | None]:
        try:
            shared_store = self._open_record_store(root, create=False)
            current = shared_store.get(self._RECORD_NAMESPACE, memory_id, default=None)
            if not isinstance(current, Mapping):
                return False, None
            normalized = self._normalize_record(current, fallback_id=memory_id)
            return self._record_equal(normalized, desired), normalized
        except Exception:
            return False, None

    def _mark_unavailable(self, operation_id: str, generation: int, error: object) -> None:
        with self._lock:
            if generation != self._generation:
                return
            operation = self._pending.get(operation_id)
            if operation is None:
                return
            self._shared_status = "unavailable"
            self._shared_error = self._bounded_error(error)
            self._last_refresh_at = time.time()
            self._receipts[operation_id] = self._receipt_for_operation(
                operation,
                status="local_queued",
                error=error,
            )

    def _mark_conflict(
        self,
        operation_id: str,
        generation: int,
        current: Mapping[str, Any] | None,
        *,
        error: object = None,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                return
            operation = self._pending.get(operation_id)
            if operation is None:
                return
            operation = dict(operation)
            operation["status"] = "conflict"
            if current is not None:
                operation["conflict_record"] = dict(current)
            operation["error"] = self._bounded_error(error) if error else "shared revision changed"
            self._pending[operation_id] = operation
            self._shared_status = "ready" if self._shared_ready else self._shared_status
            self._shared_error = None if self._shared_ready else self._shared_error
            self._receipts[operation_id] = self._receipt_for_operation(
                operation,
                status="conflict",
                error=operation["error"],
                observed=current,
            )
        with self._storage_lock:
            local_store = self._local_store
            if local_store is not None:
                try:
                    local_store.put(self._PENDING_NAMESPACE, operation_id, operation)
                except Exception:
                    log.debug("could not persist project-memory conflict %s", operation_id, exc_info=True)
        snapshot = None
        with self._lock:
            if current is not None:
                self._shared_snapshot[str(operation["memory_id"])] = dict(current)
                snapshot = {key: dict(value) for key, value in self._shared_snapshot.items()}
        if snapshot is not None:
            try:
                self._persist_snapshot(snapshot)
            except Exception:
                log.debug("could not persist conflict last-good snapshot", exc_info=True)

    def _remove_overlay_if_operation(
        self,
        local_store,
        memory_id: str,
        operation_id: str,
    ) -> bool:
        removed = False

        def mutate(current):
            nonlocal removed
            if current is None:
                return None
            candidate = current.get("record") if isinstance(current, Mapping) else current
            if not isinstance(candidate, Mapping):
                return current
            current_operation_id = str(
                candidate.get("operation_id") or candidate.get("op_id") or ""
            )
            if current_operation_id != operation_id:
                return current
            removed = True
            return None

        local_store.update(self._OVERLAY_NAMESPACE, memory_id, mutate)
        return removed

    def _finish_published(
        self,
        operation_id: str,
        generation: int,
        *,
        cleanup_error: object = None,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                return
            operation = self._pending.get(operation_id)
            if operation is None:
                return
            memory_id = str(operation["memory_id"])
            desired = dict(operation["record"])
            snapshot = {key: dict(value) for key, value in self._shared_snapshot.items()}
            snapshot[memory_id] = desired
        try:
            self._persist_snapshot(snapshot)
        except Exception as exc:
            # The shared after-image is authoritative, but without a durable
            # last-good snapshot the pending operation must remain for restart
            # recovery and idempotent retry.
            with self._lock:
                if generation == self._generation:
                    self._shared_snapshot[memory_id] = desired
                    self._shared_ready = True
                    self._shared_status = "empty" if desired.get("deleted") else "ready"
                    self._shared_error = self._bounded_error(exc)
                    self._receipts[operation_id] = self._receipt_for_operation(
                        operation,
                        status="shared",
                        error=cleanup_error or exc,
                        shared_saved=True,
                    )
                    self._retry_at = time.monotonic() + self._DEFAULT_LOCAL_RETRY_SECONDS
            return

        pending_removed = False
        overlay_removed = False
        local_cleanup_error = cleanup_error
        with self._storage_lock:
            local_store = self._local_store
            if local_store is not None:
                try:
                    overlay_removed = self._remove_overlay_if_operation(
                        local_store,
                        memory_id,
                        operation_id,
                    )
                except Exception as exc:
                    local_cleanup_error = local_cleanup_error or exc
                    log.debug(
                        "could not CAS-clear project-memory overlay %s",
                        memory_id,
                        exc_info=True,
                    )
                try:
                    local_store.delete(self._PENDING_NAMESPACE, operation_id)
                    pending_removed = True
                except Exception as exc:
                    local_cleanup_error = local_cleanup_error or exc
                    log.debug(
                        "could not clear project-memory pending op %s",
                        operation_id,
                        exc_info=True,
                    )
        with self._lock:
            if generation != self._generation:
                return
            self._shared_snapshot = snapshot
            self._shared_ready = True
            self._shared_status = "empty" if not any(
                not bool(record.get("deleted", False))
                for record in snapshot.values()
            ) else "ready"
            self._shared_error = None if local_cleanup_error is None else self._bounded_error(local_cleanup_error)
            if pending_removed:
                self._pending.pop(operation_id, None)
            current_overlay = self._overlay.get(memory_id)
            if overlay_removed and isinstance(current_overlay, Mapping):
                if str(current_overlay.get("operation_id") or "") == operation_id:
                    self._overlay.pop(memory_id, None)
                    if self._latest_operation.get(memory_id) == operation_id:
                        self._latest_operation.pop(memory_id, None)
            self._receipts[operation_id] = self._receipt_for_operation(
                operation,
                status="shared",
                error=local_cleanup_error,
                shared_saved=True,
            )
            self._retry_at = (
                time.monotonic() + self._DEFAULT_LOCAL_RETRY_SECONDS
                if local_cleanup_error is not None or not pending_removed
                else 0.0
            )

    def invalidate_buffer(self, state) -> None:
        manager = getattr(state, "buffer_manager", None)
        cache = getattr(manager, "_special_cache", None)
        if isinstance(cache, dict):
            cache.pop("project_memory", None)


class ProjectMemoryBuffer:
    """Register the live readonly ``project_memory`` special buffer."""

    reads = {"buffer_manager"}
    optional_reads = {"_agent_zoo_context"}
    writes = {"buffer_manager"}

    def __init__(self, store: ProjectMemoryStore) -> None:
        self.store = store

    def __call__(self, state):
        manager = state.buffer_manager
        setattr(state, "_project_memory_render", functools.partial(self.render, state))
        _register_exact_special_buffer(
            manager,
            "project_memory",
            operator.methodcaller("_project_memory_render"),
        )
        return state

    def render(self, state) -> ReadonlyBufferView:
        memories = self.store.list_all(state)
        status = self.store.snapshot_status()
        lines = ["# Project memories", "", f"Count: {len(memories)}"]
        storage_status = str(status.get("status") or "not_ready")
        if storage_status == "unavailable":
            detail = status.get("error") or "shared storage is unavailable"
            lines.extend(["", f"Shared storage: unavailable ({detail})."])
            if status.get("last_good"):
                lines.append("Last-good shared snapshot retained; local queued edits remain visible.")
            else:
                lines.append("No shared snapshot is available; empty results are not authoritative.")
        elif storage_status == "not_ready":
            lines.extend(
                [
                    "",
                    "Shared storage: loading in background; no shared snapshot is ready yet.",
                    "Local queued edits remain visible.",
                ]
            )
        else:
            lines.extend(["", f"Shared storage: {storage_status}."])
        pending = int(status.get("pending_operations", 0) or 0)
        conflicts = list(status.get("conflicts") or ())
        if pending:
            lines.append(f"Local queued operations: {pending}")
        if conflicts:
            lines.append("Unresolved conflicts: " + ", ".join(conflicts))
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

        common_memories = self.store.list_common()
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
        return ReadonlyBufferView(
            id="project_memory",
            path="memory://project",
            text="\n".join(lines).rstrip() + "\n",
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
                return project_name, None, current_session_id
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
    optional_reads = {"_agent_zoo_context"}

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

    def _format(self, memory: Mapping[str, Any], verb: str) -> str:
        result = (
            f"{verb} {memory['id']}.\n"
            f"Trigger: {memory['trigger']}\n"
            f"Content: {memory['content']}"
        )
        receipt = self.store.receipt_text(str(memory["id"]))
        return result + (f"\n{receipt}" if receipt else "")


class RecordMemory(ProjectMemoryTool):
    """Save a project-scoped memory with a regular-expression trigger."""

    reads = {"entries", "step"}
    optional_reads = {"_agent_zoo_context", "context_compaction"}

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
            if truncated:
                return result + (
                    "\nBacktest: incomplete; the active context scan was truncated before "
                    "an exhaustive no-match conclusion."
                )
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

    optional_reads = {"_agent_zoo_context", PENDING_RECALLS_ATTR}
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

    optional_reads = {"_agent_zoo_context", PENDING_RECALLS_ATTR}
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

    optional_reads = {"_agent_zoo_context", SUPPRESSIONS_ATTR, PENDING_RECALLS_ATTR}
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
        receipt = self.store.receipt_text(memory_id)
        result = f"Deleted {memory_id}."
        return result + (f"\n{receipt}" if receipt else "")


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
