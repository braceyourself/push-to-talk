---
name: ptt-deploy
description: Deploy push-to-talk changes from repo to the running service. Syncs all source files, personality, config, and restarts the systemd unit.
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Grep
---

## When to Use

**Always run this after modifying any push-to-talk source file.** This is the ONLY way changes reach the running service.

The repo lives at `/home/ethan/code/push-to-talk/`. The service runs from `/home/ethan/.local/share/push-to-talk/`. Editing repo files does nothing until deployed.

## Deployable Files

All of these must be synced (repo → deploy dir):

**Python source:**
- `push-to-talk.py`
- `indicator.py`
- `live_session.py`
- `input_classifier.py`
- `learner.py`
- `task_manager.py`
- `task_tools_mcp.py`
- `pipeline_frames.py`
- `clip_factory.py`
- `stream_composer.py`
- `response_library.py`
- `openai_realtime.py`

**Data/config:**
- `category_exemplars.json`
- `vocabulary.txt`
- `requirements.txt`

**Personality (recursive):**
- `personality/` directory (core.md, context.md, voice-style.md, memories/*.md)

## Steps

1. **Diff check** — Show which files differ between repo and deploy:
   ```bash
   REPO=/home/ethan/code/push-to-talk
   DEPLOY=/home/ethan/.local/share/push-to-talk
   changed=()
   for f in push-to-talk.py indicator.py live_session.py input_classifier.py learner.py task_manager.py task_tools_mcp.py pipeline_frames.py clip_factory.py stream_composer.py response_library.py openai_realtime.py category_exemplars.json vocabulary.txt requirements.txt; do
     if [ -f "$REPO/$f" ] && ! diff -q "$REPO/$f" "$DEPLOY/$f" > /dev/null 2>&1; then
       changed+=("$f")
     fi
   done
   # Also check personality dir
   if ! diff -rq "$REPO/personality/" "$DEPLOY/personality/" > /dev/null 2>&1; then
     changed+=("personality/")
   fi
   echo "Changed: ${changed[*]:-nothing}"
   ```

2. **Copy changed files** — Sync each changed file individually:
   ```bash
   for f in "${changed[@]}"; do
     if [ "$f" = "personality/" ]; then
       cp -r "$REPO/personality/" "$DEPLOY/personality/"
     else
       cp "$REPO/$f" "$DEPLOY/$f"
     fi
   done
   ```

3. **Restart service:**
   ```bash
   systemctl --user restart push-to-talk.service
   ```

4. **Verify startup** — Tail journal briefly to confirm clean start:
   ```bash
   sleep 2 && journalctl --user -u push-to-talk -n 5 --no-pager
   ```

5. **Report** — List what was deployed and confirm service is running.

## Important

- NEVER skip any file category. Past bugs were caused by deploying only some files.
- If `requirements.txt` changed, also run: `~/.local/share/push-to-talk/venv/bin/pip install -r $REPO/requirements.txt`
- The service MUST be restarted for changes to take effect. GTK indicator changes require restart too.
