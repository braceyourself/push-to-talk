#!/usr/bin/env python3
"""
Background custodian daemon for live voice sessions.

Periodically reviews the HUD and memory state, uses Claude to decide what to prune,
and removes stale/contradictory information.

Usage: python custodian.py <path-to-session-dir>
       python custodian.py <path-to-events.jsonl>    (legacy compat)
"""

import sys
import os
import json
import time
import shutil
import subprocess
from pathlib import Path

import aihud

# Review intervals
FIRST_REVIEW_DELAY = 30   # seconds after startup
REVIEW_INTERVAL = 300      # 5 minutes between reviews

MEMORIES_DIR = Path(__file__).parent / "personality" / "memories"

CLAUDE_CLI = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")

REVIEW_PROMPT = """\
You are a memory custodian for a voice assistant. Your job is to review the HUD and memory files, \
identify what is stale, contradictory, or no longer useful, and decide what to remove.

## Current HUD State
{hud_json}

## Existing Memories
{existing_memories}

## Your Task
1. Review notifications — remove any that are:
   - Older than 24 hours and type "learning" (user already knows)
   - Orphaned (task no longer in system)
2. Review learnings — remove any that are:
   - Contradicted by newer facts
   - One-off observations not reinforced
   - Low-confidence or speculative
3. Review memory files — identify:
   - Duplicate entries across files
   - Contradictions (keep newest, remove old)
   - Information confirmed outdated
4. Output a JSON plan specifying what to remove/update.

Output ONLY valid JSON:
{{
  "remove_notifications": ["n1", "n2"],
  "remove_learnings": [0, 1],
  "remove_memory_lines": [
    {{"file": "personal.md", "line": "line text to remove"}}
  ]
}}

If nothing to remove, output:
{{
  "remove_notifications": [],
  "remove_learnings": [],
  "remove_memory_lines": []
}}
"""


def read_existing_memories() -> str:
    """Read all existing memory files for review."""
    if not MEMORIES_DIR.exists():
        return "(none)"
    parts = []
    for md_file in sorted(MEMORIES_DIR.glob("*.md")):
        content = md_file.read_text().strip()
        if content:
            parts.append(f"### {md_file.name}\n{content}")
    return "\n\n".join(parts) if parts else "(none)"


def run_review() -> dict | None:
    """Run Claude review to decide what to prune. Returns pruning plan or None."""
    hud = aihud.read_hud()
    existing = read_existing_memories()

    prompt = REVIEW_PROMPT.format(
        hud_json=json.dumps(hud, indent=2),
        existing_memories=existing,
    )

    env = {**os.environ}
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.pop("CLAUDECODE", None)

    try:
        result = subprocess.run(
            [CLAUDE_CLI, "-p", prompt,
             "--no-session-persistence",
             "--permission-mode", "bypassPermissions",
             "--output-format", "text",
             "--model", "claude-sonnet-4-5-20250929"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        if result.returncode != 0:
            print(f"Custodian: claude -p failed (exit {result.returncode}): {result.stderr[:200]}", flush=True)
            return None

        output = result.stdout.strip()
        return parse_review_output(output)

    except subprocess.TimeoutExpired:
        print("Custodian: claude -p timed out", flush=True)
        return None
    except Exception as e:
        print(f"Custodian: review error: {e}", flush=True)
        return None


def parse_review_output(output: str) -> dict | None:
    """Parse the JSON output from claude review."""
    json_str = output
    if "```json" in json_str:
        json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in json_str:
        json_str = json_str.split("```", 1)[1].split("```", 1)[0]

    try:
        plan = json.loads(json_str.strip())
        return plan
    except json.JSONDecodeError:
        print(f"Custodian: Failed to parse review output: {output[:200]}", flush=True)
        return None


def apply_pruning_plan(plan: dict):
    """Apply the pruning plan to HUD and memory files."""
    if not plan:
        return

    # Remove notifications
    remove_notif_ids = plan.get("remove_notifications", [])
    if remove_notif_ids:
        for notif_id in remove_notif_ids:
            aihud.dismiss_notification(notif_id)
        print(f"Custodian: Removed {len(remove_notif_ids)} notifications", flush=True)

    # Remove learnings
    remove_learning_indices = plan.get("remove_learnings", [])
    if remove_learning_indices:
        hud = aihud.read_hud()
        learnings = hud.get("learnings", [])
        # Remove in reverse order to avoid index shifting
        for idx in sorted(remove_learning_indices, reverse=True):
            if 0 <= idx < len(learnings):
                learnings.pop(idx)
        hud["learnings"] = learnings
        aihud.write_hud(hud)
        print(f"Custodian: Removed {len(remove_learning_indices)} learnings", flush=True)

    # Remove memory lines
    remove_lines = plan.get("remove_memory_lines", [])
    if remove_lines:
        for entry in remove_lines:
            filename = entry.get("file", "")
            line_text = entry.get("line", "").strip()
            if not filename or not line_text:
                continue
            filepath = MEMORIES_DIR / filename
            if not filepath.exists():
                continue
            try:
                content = filepath.read_text()
                # Remove the line (exact match)
                lines = content.split('\n')
                new_lines = [l for l in lines if l.strip() != line_text]
                filepath.write_text('\n'.join(new_lines))
                print(f"Custodian: Removed line from {filename}", flush=True)
            except Exception as e:
                print(f"Custodian: Error removing line from {filename}: {e}", flush=True)


def _resolve_bus_path(arg: str) -> Path:
    """Resolve argument to events.jsonl path."""
    p = Path(arg)
    if p.is_dir():
        return p / "events.jsonl"
    if p.name == "events.jsonl":
        return p
    if p.name == "conversation.jsonl":
        events_path = p.parent / "events.jsonl"
        if events_path.exists():
            return events_path
    return p


def tail_and_review(bus_path: Path):
    """Main loop: tail the event bus and periodically review/prune."""
    print(f"Custodian: Watching {bus_path}", flush=True)

    last_event_time = time.time()
    last_review_time = 0
    session_ended = False

    # Wait for the file to appear
    while not bus_path.exists():
        time.sleep(0.5)

    with open(bus_path, "r") as f:
        while True:
            line = f.readline()

            if line:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                last_event_time = time.time()
                event_type = event.get("type", "")

                if event_type == "session_end":
                    session_ended = True

            # Check review timing
            now = time.time()
            should_review = False

            if last_review_time == 0 and (now - last_event_time) >= FIRST_REVIEW_DELAY:
                # First review after startup
                should_review = True
            elif last_review_time > 0 and (now - last_review_time) >= REVIEW_INTERVAL:
                # Periodic review
                should_review = True

            if should_review:
                print("Custodian: Running review", flush=True)
                plan = run_review()
                if plan:
                    apply_pruning_plan(plan)
                last_review_time = now

            if session_ended:
                print("Custodian: Session ended, exiting", flush=True)
                break

            if not line:
                # No new data — sleep briefly before polling again
                time.sleep(1.0)


def main():
    if len(sys.argv) < 2:
        print("Usage: custodian.py <path-to-session-dir-or-events.jsonl>", file=sys.stderr)
        sys.exit(1)

    bus_path = _resolve_bus_path(sys.argv[1])
    try:
        tail_and_review(bus_path)
    except KeyboardInterrupt:
        print("Custodian: Interrupted", flush=True)
    except Exception as e:
        print(f"Custodian: Fatal error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
