#!/bin/bash
# Watch for changes to push-to-talk files and auto-sync

INSTALL_DIR="$HOME/.local/share/push-to-talk"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Debounce: wait for changes to settle before syncing
DEBOUNCE_SEC=5
last_sync=0

sync_if_needed() {
    now=$(date +%s)
    if (( now - last_sync >= DEBOUNCE_SEC )); then
        echo "[$(date)] Changes detected, syncing..."
        "$SCRIPT_DIR/sync.sh"
        last_sync=$(date +%s)
    fi
}

echo "Watching for changes in $INSTALL_DIR..."

inotifywait -m -e modify,create,delete \
    "$INSTALL_DIR/push-to-talk.py" \
    "$INSTALL_DIR/indicator.py" \
    "$INSTALL_DIR/vocabulary.txt" \
    2>/dev/null | while read -r; do
    sync_if_needed
done
