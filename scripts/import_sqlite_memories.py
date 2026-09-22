#!/usr/bin/env python3
"""Explicit one-time import of legacy SQLite project memories.

Runtime project-memory storage never opens SQLite.  This command is the only
SQLite compatibility path: it opens the source database read-only and writes
current filesystem records into an explicitly supplied RecordStore root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import regex
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

TABLE_NAME = "azo_project_memories"
NAMESPACE = "project_memory"
SCHEMA_VERSION = 1


def _compile_trigger(pattern: str):
    return regex.compile(pattern)


def _read_only_connection(source: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(source), safe='/')}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _rows(source: Path) -> list[dict[str, Any]]:
    with _read_only_connection(source) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({TABLE_NAME})")
        }
        required = {"id", "content", "trigger", "created_at", "updated_at"}
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(
                f"SQLite table {TABLE_NAME!r} is missing columns: {', '.join(missing)}"
            )
        result = []
        for row in connection.execute(
            f"SELECT id, content, trigger, created_at, updated_at "
            f"FROM {TABLE_NAME} ORDER BY created_at, id"
        ):
            memory_id, content, trigger, created_at, updated_at = row
            memory_id = str(memory_id or "").strip()
            content = str(content or "")
            trigger = str(trigger or "")
            if not memory_id:
                raise RuntimeError("SQLite memory row has an empty id")
            if not content.strip():
                raise RuntimeError(f"SQLite memory {memory_id!r} has empty content")
            if not trigger.strip():
                raise RuntimeError(f"SQLite memory {memory_id!r} has empty trigger")
            try:
                _compile_trigger(trigger)
                created = float(created_at)
                updated = float(updated_at)
            except Exception as exc:
                raise RuntimeError(
                    f"SQLite memory {memory_id!r} has invalid content or timestamps: {exc}"
                ) from exc
            result.append(
                {
                    "id": memory_id,
                    "content": content,
                    "trigger": trigger,
                    "created_at": created,
                    "updated_at": updated,
                }
            )
        return result


def _operation_id(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        [
            row["id"],
            row["content"],
            row["trigger"],
            row["created_at"],
            row["updated_at"],
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "import_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": str(row["id"]),
        "content": str(row["content"]),
        "trigger": str(row["trigger"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "revision": 1,
        "deleted": False,
        "operation_id": _operation_id(row),
    }


def _same_logical_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(field) == right.get(field)
        for field in ("id", "content", "trigger", "created_at", "updated_at", "deleted")
    )


def import_sqlite_memories(
    source: str | Path,
    destination: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import rows without modifying the SQLite source or overwriting conflicts."""
    source_path = Path(source).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise RuntimeError(f"SQLite source is not a regular file: {source_path}")
    rows = _rows(source_path)

    try:
        from tmux_pilot.fs_store import RecordStore
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise RuntimeError("current tmux-pilot RecordStore is unavailable") from exc

    destination_path = Path(destination).expanduser().resolve()
    store = None
    if not dry_run:
        store = RecordStore(destination_path, create=True)
    elif destination_path.exists():
        store = RecordStore(destination_path, create=False)

    imported = 0
    existing = 0
    conflicts = []
    for row in rows:
        candidate = _candidate(row)
        memory_id = candidate["id"]
        if dry_run:
            if store is None:
                imported += 1
            else:
                current = store.get(NAMESPACE, memory_id, default=None)
                if current is None:
                    imported += 1
                elif _same_logical_record(current, candidate):
                    existing += 1
                else:
                    conflicts.append(memory_id)
            continue
        if store is None:
            imported += 1
            continue
        outcome: dict[str, Any] = {}

        def mutate(current):
            if current is None:
                outcome["status"] = "imported"
                return candidate
            if not isinstance(current, Mapping):
                raise RuntimeError(f"filesystem memory {memory_id!r} is malformed")
            if _same_logical_record(current, candidate):
                outcome["status"] = "existing"
                return dict(current)
            outcome["status"] = "conflict"
            outcome["current"] = dict(current)
            return dict(current)

        store.update(NAMESPACE, memory_id, mutate)
        status = outcome.get("status")
        if status == "imported":
            imported += 1
        elif status == "existing":
            existing += 1
        else:
            conflicts.append(memory_id)

    return {
        "source": str(source_path),
        "destination": str(destination_path),
        "rows": len(rows),
        "imported": imported,
        "existing": existing,
        "conflicts": conflicts,
        "dry_run": bool(dry_run),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicitly import azo_project_memories from a read-only SQLite source."
    )
    parser.add_argument("source", help="Legacy SQLite database file.")
    parser.add_argument("destination", help="Current filesystem RecordStore root.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate the source without writing the destination.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON result instead of a human-readable summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = import_sqlite_memories(
            args.source,
            args.destination,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"import_sqlite_memories: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        action = "Validated" if args.dry_run else "Imported"
        print(
            f"{action} {result['rows']} SQLite memory row(s): "
            f"{result['imported']} new, {result['existing']} unchanged, "
            f"{len(result['conflicts'])} conflict(s)."
        )
        if result["conflicts"]:
            print("Conflicts preserved: " + ", ".join(result["conflicts"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
