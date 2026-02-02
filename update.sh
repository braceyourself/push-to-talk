#!/bin/bash
# Update push-to-talk from git and reinstall

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$HOME/.local/share/push-to-talk"

cd "$SCRIPT_DIR"

# Pull latest changes
echo "Pulling latest changes..."
git pull --ff-only

# Copy updated files
echo "Updating installed files..."
cp push-to-talk.py "$INSTALL_DIR/"
cp indicator.py "$INSTALL_DIR/"

# Restart service
echo "Restarting service..."
systemctl --user restart push-to-talk.service

echo "Update complete!"
