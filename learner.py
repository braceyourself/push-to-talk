#!/usr/bin/env python3
"""
Background learner daemon for live voice sessions.

Tails the event bus JSONL log, periodically evaluates accumulated turns
via `claude -p`, writes discoveries to personality/memories/*.md, and
signals the live session via bus event.

Usage: python learner.py <path-to-session-dir>
       python learner.py <path-to-events.jsonl>    (legacy compat)
"""

import sys
import os
import json
import time
import shutil
import subprocess
from pathlib import Path

from event_bus import EventBusWriter

# Evaluation thresholds
IDLE_SECONDS = 10       # seconds of no new events before evaluating
MIN_USER_TURNS = 3      # minimum user turns since last eval to trigger
MIN_SESSION_TURNS = 2   # minimum total user turns to evaluate at all

MEMORIES_DIR = Path(__file__).parent / "personality" / "memories"

CLAUDE_CLI = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")

EXTRACTION_PROMPT = """\
You are a memory extraction system for a voice assistant. Below is a recent conversation \
between the user (Ethan) and the voice assistant. Your job is to extract durable facts worth \
remembering across sessions.

## What to Extract
- Personal facts: name, pets, family, location, job, hobbies
- Preferences: tools, languages, workflows, communication style
- Corrections: things the user corrected the assistant on
- Projects and topics: what the user is working on, cares about
- Behavioral feedback: if the user said "don't do X" or "I prefer Y"

## What to Skip
- Small talk, greetings, one-off questions with no lasting value
- Things already known (listed below under EXISTING MEMORIES)
- Speculation or things that might not be true
- Session-specific context (current task status, temporary state)

## EXISTING MEMORIES
{existing_memories}

## RECENT CONVERSATION
{conversation}

## Instructions
1. Read the conversation carefully.
2. Identify any new durable facts not already in EXISTING MEMORIES.
3. For each fact, write it to the appropriate file below by outputting a JSON block.
4. Output a single-line voice notification summary at the end.

Output format — output ONLY valid JSON, nothing else:
{{
  "writes": [
    {{
      "file": "personal.md",
      "content": "- Fact here <!-- learned {date} -->"
    }}
  ],
  "notification": "Short natural summary of what was learned, or empty string if nothing new"
}}

Valid files: personal.md, preferences.md, projects.md, corrections.md

If nothing worth remembering was found, output:
{{
  "writes": [],
  "notification": ""
}}
"""


def read_existing_memories() -> str:
    """Read all existing memory files for deduplication."""
    if not MEMORIES_DIR.exists():
        return "(none)"
    parts = []
    for md_file in sorted(MEMORIES_DIR.glob("*.md")):
        content = md_file.read_text().strip()
        if content:
            parts.append(f"### {md_file.name}\n{content}")
    return "\n\n".join(parts) if parts else "(none)"


def build_transcript(turns: list[dict]) -> str:
    """Format turns into a readable transcript."""
    lines = []
    for turn in turns:
        role = turn.get("type", "unknown")
        text = turn.get("text", "")
        if role == "user":
            lines.append(f"User: {text}")
        elif role == "assistant":
            lines.append(f"Assistant: {text}")
    return "\n".join(lines)


def run_evaluation(turns: list[dict]) -> str | None:
    """Run claude -p to extract learnings. Returns notification summary or None."""
    transcript = build_transcript(turns)
    if not transcript.strip():
        return None

    existing = read_existing_memories()
    date = time.strftime("%Y-%m-%d")

    prompt = EXTRACTION_PROMPT.format(
        existing_memories=existing,
        conversation=transcript,
        date=date,
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
            print(f"Learner: claude -p failed (exit {result.returncode}): {result.stderr[:200]}", flush=True)
            return None

        output = result.stdout.strip()
        return process_extraction_output(output)

    except subprocess.TimeoutExpired:
        print("Learner: claude -p timed out", flush=True)
        return None
    except Exception as e:
        print(f"Learner: evaluation error: {e}", flush=True)
        return None


def process_extraction_output(output: str) -> str | None:
    """Parse the JSON output from claude, write memories, return notification."""
    # Try to extract JSON from the output (may have markdown fences)
    json_str = output
    if "```json" in json_str:
        json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in json_str:
        json_str = json_str.split("```", 1)[1].split("```", 1)[0]

    try:
        data = json.loads(json_str.strip())
    except json.JSONDecodeError:
        print(f"Learner: Failed to parse extraction output: {output[:200]}", flush=True)
        return None

    writes = data.get("writes", [])
    notification = data.get("notification", "")

    if not writes:
        print("Learner: No learnings extracted", flush=True)
        return None

    # Write memories
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    for write in writes:
        filename = write.get("file", "")
        content = write.get("content", "").strip()
        if not filename or not content:
            continue
        # Sanitize filename
        if filename not in ("personal.md", "preferences.md", "projects.md", "corrections.md"):
            print(f"Learner: Skipping unknown memory file: {filename}", flush=True)
            continue
        filepath = MEMORIES_DIR / filename
        # Append to existing file or create with header
        if filepath.exists():
            existing = filepath.read_text()
            if content in existing:
                print(f"Learner: Skipping duplicate in {filename}", flush=True)
                continue
            with open(filepath, "a") as f:
                f.write("\n" + content + "\n")
        else:
            header = f"# {filename.replace('.md', '').title()}\n\n"
            filepath.write_text(header + content + "\n")
        print(f"Learner: Wrote to {filename}: {content[:60]}...", flush=True)

    return notification if notification else None


def _resolve_bus_path(arg: str) -> Path:
    """Resolve argument to events.jsonl path.

    Accepts either:
    - Path to session directory (contains events.jsonl)
    - Direct path to events.jsonl (or legacy conversation.jsonl)
    """
    p = Path(arg)
    if p.is_dir():
        return p / "events.jsonl"
    if p.name == "events.jsonl":
        return p
    # Legacy: conversation.jsonl -> try events.jsonl in same dir
    if p.name == "conversation.jsonl":
        events_path = p.parent / "events.jsonl"
        if events_path.exists():
            return events_path
    return p  # Fall through — will be whatever was passed


def _extract_session_id(bus_path: Path) -> str:
    """Extract session ID from bus path (parent dir name)."""
    return bus_path.parent.name


def tail_and_evaluate(bus_path: Path):
    """Main loop: tail the event bus JSONL and evaluate periodically."""
    print(f"Learner: Watching {bus_path}", flush=True)

    session_id = _extract_session_id(bus_path)
    bus_writer = EventBusWriter(bus_path, "learner", session_id)

    all_turns = []           # All user/assistant turns in session
    pending_turns = []       # Turns since last evaluation
    user_turns_since_eval = 0
    last_event_time = time.time()
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
                    # Fall through to final evaluation below

                elif event_type in ("user", "assistant"):
                    all_turns.append(event)
                    pending_turns.append(event)
                    if event_type == "user":
                        user_turns_since_eval += 1

            # Check evaluation conditions
            idle_gap = time.time() - last_event_time

            should_evaluate = False
            if session_ended and pending_turns:
                # Final evaluation on session end
                total_user = sum(1 for t in all_turns if t.get("type") == "user")
                if total_user >= MIN_SESSION_TURNS:
                    should_evaluate = True
            elif idle_gap >= IDLE_SECONDS and user_turns_since_eval >= MIN_USER_TURNS:
                should_evaluate = True

            if should_evaluate:
                print(f"Learner: Evaluating ({len(pending_turns)} turns, {user_turns_since_eval} user)", flush=True)
                notification = run_evaluation(pending_turns)
                pending_turns = []
                user_turns_since_eval = 0

                if notification:
                    try:
                        bus_writer.emit("learner_notify", summary=notification)
                        print(f"Learner: Notification emitted: {notification[:60]}...", flush=True)
                    except Exception as e:
                        print(f"Learner: Failed to emit notification: {e}", flush=True)

            if session_ended:
                print("Learner: Session ended, exiting", flush=True)
                break

            if not line:
                # No new data — sleep briefly before polling again
                time.sleep(1.0)


def main():
    if len(sys.argv) < 2:
        print("Usage: learner.py <path-to-session-dir-or-events.jsonl>", file=sys.stderr)
        sys.exit(1)

    bus_path = _resolve_bus_path(sys.argv[1])
    try:
        tail_and_evaluate(bus_path)
    except KeyboardInterrupt:
        print("Learner: Interrupted", flush=True)
    except Exception as e:
        print(f"Learner: Fatal error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
