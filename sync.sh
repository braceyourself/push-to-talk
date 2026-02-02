#!/bin/bash
# Sync local changes to git and update remote machines

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.local/share/push-to-talk"
REMOTES="laptop"  # Space-separated list of remote hosts

cd "$SCRIPT_DIR"

# Copy installed files back to repo if they're newer
for file in push-to-talk.py indicator.py vocabulary.txt; do
    if [ -f "$INSTALL_DIR/$file" ]; then
        if [ "$INSTALL_DIR/$file" -nt "$SCRIPT_DIR/$file" ] 2>/dev/null; then
            cp "$INSTALL_DIR/$file" "$SCRIPT_DIR/$file"
            echo "Updated $file from install dir"
        fi
    fi
done

# Check for changes
if git diff --quiet && git diff --cached --quiet; then
    echo "No changes to commit"
    exit 0
fi

# Commit and push
git add -A
git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M')"
git push

# Update remotes
for remote in $REMOTES; do
    echo "Updating $remote..."
    ssh "$remote" "cd ~/projects/push-to-talk && git pull --ff-only && ./update.sh" &
done

wait
echo "Sync complete"
