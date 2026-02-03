#!/bin/bash
# Update push-to-talk from git
set -e

INSTALL_DIR="$HOME/.local/share/push-to-talk"
cd "$INSTALL_DIR"

# Pull latest
git pull --ff-only origin main 2>/dev/null || git pull --ff-only

# Reinstall any new pip dependencies
if [ -f requirements.txt ] && [ -f venv/bin/pip ]; then
    venv/bin/pip install -q -r requirements.txt 2>/dev/null
fi

# Restart service
systemctl --user restart push-to-talk.service
echo "[$(date)] Push-to-Talk updated"
