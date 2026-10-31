"""
Demo services registry module.
Tracks demo services created via the /ui operator console in an atomic JSON file.
Thread-safe with asyncio.Lock, atomic replacement on write, safe fallback on corruption.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

logger = logging.getLogger("siglip.demo_registry")

_REGISTRY_LOCK = asyncio.Lock()


def _resolve_path(registry_path: str) -> Path:
    p = Path(registry_path)
    if not p.is_absolute():
        # Relative to project root
        p = Path(__file__).resolve().parent.parent / p
    return p


def _read_registry_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                # Validate item structure
                clean_entries = []
                for item in data:
                    if isinstance(item, dict) and "id" in item and "title" in item:
                        clean_entries.append({
                            "id": int(item["id"]),
                            "title": str(item["title"]),
                            "created_at": str(item.get("created_at") or datetime.now(timezone.utc).isoformat()),
                        })
                return clean_entries
            logger.warning("Demo registry at %s is not a JSON list; treating as empty", path)
            return []
    except Exception as exc:
        logger.warning("Failed to read demo registry at %s: %s; treating as empty", path, exc)
        return []


def _write_registry_file(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            prefix="demo_reg_",
            suffix=".tmp",
        ) as f:
            temp_file = f.name
            json.dump(entries, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_file, str(path))
        temp_file = None
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


async def load_demo_registry(registry_path: str) -> list[dict[str, Any]]:
    """Loads all demo service entries from the JSON registry."""
    path = _resolve_path(registry_path)
    async with _REGISTRY_LOCK:
        return _read_registry_file(path)


async def get_demo_entry(registry_path: str, service_id: int) -> dict[str, Any] | None:
    """Finds a single demo entry by service ID."""
    path = _resolve_path(registry_path)
    async with _REGISTRY_LOCK:
        entries = _read_registry_file(path)
        for e in entries:
            if e["id"] == service_id:
                return e
        return None


async def add_demo_service(registry_path: str, service_id: int, title: str) -> dict[str, Any]:
    """Adds or updates a demo service entry in the registry."""
    path = _resolve_path(registry_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    entry = {
        "id": int(service_id),
        "title": title.strip(),
        "created_at": now_iso,
    }
    async with _REGISTRY_LOCK:
        entries = _read_registry_file(path)
        # Remove any existing with same id
        entries = [e for e in entries if e["id"] != service_id]
        entries.append(entry)
        _write_registry_file(path, entries)
    logger.info("Registered demo service id=%d ('%s')", service_id, title)
    return entry


async def remove_demo_service(registry_path: str, service_id: int) -> bool:
    """Removes a demo service entry from the registry. Returns True if removed."""
    path = _resolve_path(registry_path)
    async with _REGISTRY_LOCK:
        entries = _read_registry_file(path)
        initial_len = len(entries)
        entries = [e for e in entries if e["id"] != service_id]
        if len(entries) != initial_len:
            _write_registry_file(path, entries)
            logger.info("Unregistered demo service id=%d", service_id)
            return True
        return False


async def clear_demo_registry(registry_path: str) -> int:
    """Clears all entries from the demo registry. Returns count of cleared entries."""
    path = _resolve_path(registry_path)
    async with _REGISTRY_LOCK:
        entries = _read_registry_file(path)
        count = len(entries)
        _write_registry_file(path, [])
        logger.info("Cleared %d entries from demo registry", count)
        return count


async def prune_demo_registry(registry_path: str, valid_ids: set[int]) -> list[dict[str, Any]]:
    """Prunes stale demo entries whose IDs are not in valid_ids."""
    path = _resolve_path(registry_path)
    async with _REGISTRY_LOCK:
        entries = _read_registry_file(path)
        retained = [e for e in entries if e["id"] in valid_ids]
        if len(retained) != len(entries):
            _write_registry_file(path, retained)
            logger.info("Pruned %d stale entries from demo registry", len(entries) - len(retained))
        return retained
