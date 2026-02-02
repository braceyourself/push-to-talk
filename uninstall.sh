#!/bin/bash
set -e

echo "Uninstalling Push-to-Talk Dictation Service..."

# Stop and disable service
systemctl --user stop push-to-talk.service 2>/dev/null || true
systemctl --user disable push-to-talk.service 2>/dev/null || true

# Remove service file
rm -f "$HOME/.config/systemd/user/push-to-talk.service"
systemctl --user daemon-reload

# Remove installation directory
INSTALL_DIR="$HOME/.local/share/push-to-talk"
if [ -d "$INSTALL_DIR" ]; then
    read -p "Remove $INSTALL_DIR (includes vocabulary)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$INSTALL_DIR"
        echo "Removed $INSTALL_DIR"
    else
        echo "Kept $INSTALL_DIR (you can remove it manually)"
    fi
fi

echo "Uninstall complete."
