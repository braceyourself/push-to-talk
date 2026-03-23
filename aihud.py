#!/usr/bin/env python3
"""
AIHUD — Shared state file for front-of-mind context in live voice sessions.

Any process (learner, task manager, custodian, live_session) can write to aihud.json
using atomic operations. The live session reads it to prepend context to every user message.

Schema:
{
  "notifications": [
    {"id": "n1", "type": "learning", "summary": "...", "ts": 1708012345.67},
    {"id": "n2", "type": "task_complete", "summary": "...", "ts": 1708012400.0}
  ],
  "tasks": [
    {"name": "fix tests", "status": "running", "elapsed": "45s"},
    {"name": "auth refactor", "status": "completed", "elapsed": "2m 10s"}
  ],
  "learnings": [
    "prefers Bun over npm",
    "dog named Biscuit"
  ]
}
"""

import json
import time
import tempfile
import os
from pathlib import Path
from uuid import uuid4


AIHUD_PATH = Path(__file__).parent / "aihud.json"


def _get_hud_path() -> Path:
    """Get the HUD file path."""
    return AIHUD_PATH


def _ensure_hud_exists():
    """Create a blank HUD file if it doesn't exist."""
    path = _get_hud_path()
    if not path.exists():
        default_hud = {
            "notifications": [],
            "tasks": [],
            "learnings": []
        }
        _atomic_write_json(path, default_hud)


def _atomic_write_json(path: Path, data: dict):
    """Write JSON to file atomically using tempfile + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, str(path))
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def read_hud() -> dict:
    """Read the current HUD state. Returns empty default if file doesn't exist."""
    _ensure_hud_exists()
    path = _get_hud_path()
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {
            "notifications": [],
            "tasks": [],
            "learnings": []
        }


def write_hud(data: dict):
    """Write HUD state atomically."""
    _atomic_write_json(_get_hud_path(), data)


def add_notification(notif_type: str, summary: str) -> str:
    """Add a notification to the HUD. Returns the notification ID."""
    hud = read_hud()
    notif_id = f"n{int(time.time() * 1000)}{str(uuid4())[:8]}"
    hud["notifications"].append({
        "id": notif_id,
        "type": notif_type,
        "summary": summary,
        "ts": time.time()
    })
    write_hud(hud)
    return notif_id


def dismiss_notification(notif_id: str):
    """Remove a notification from the HUD by ID."""
    hud = read_hud()
    hud["notifications"] = [
        n for n in hud["notifications"]
        if n.get("id") != notif_id
    ]
    write_hud(hud)


def set_tasks(task_list: list[dict]):
    """Update the task list in the HUD. Each task should have 'name', 'status', 'elapsed'."""
    hud = read_hud()
    hud["tasks"] = task_list
    write_hud(hud)


def add_learning(text: str):
    """Add a learning to the HUD."""
    hud = read_hud()
    if text not in hud["learnings"]:
        hud["learnings"].append(text)
        write_hud(hud)


def clear_learnings():
    """Clear all learnings from the HUD (for end-of-session cleanup)."""
    hud = read_hud()
    hud["learnings"] = []
    write_hud(hud)


def get_notifications_count() -> int:
    """Get the count of unaddressed notifications."""
    hud = read_hud()
    return len(hud.get("notifications", []))


def get_running_tasks() -> list[dict]:
    """Get only running tasks from the HUD."""
    hud = read_hud()
    return [
        t for t in hud.get("tasks", [])
        if t.get("status") == "running"
    ]


def get_learnings() -> list[str]:
    """Get recent learnings from the HUD."""
    hud = read_hud()
    return hud.get("learnings", [])
