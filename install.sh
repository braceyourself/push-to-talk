#!/bin/bash
set -e

echo "Installing Push-to-Talk Dictation Service..."

# Check dependencies
command -v python3 >/dev/null || { echo "Error: python3 required"; exit 1; }
command -v xdotool >/dev/null || { echo "Error: xdotool required (apt install xdotool)"; exit 1; }
command -v ffmpeg >/dev/null || { echo "Error: ffmpeg required (apt install ffmpeg)"; exit 1; }
command -v pw-record >/dev/null || { echo "Error: pw-record required (PipeWire)"; exit 1; }

# Create directories
INSTALL_DIR="$HOME/.local/share/push-to-talk"
mkdir -p "$INSTALL_DIR"
mkdir -p "$HOME/.config/systemd/user"

# Copy files
cp push-to-talk.py "$INSTALL_DIR/"
cp indicator.py "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/push-to-talk.py"
chmod +x "$INSTALL_DIR/indicator.py"

# Create vocabulary if it doesn't exist
if [ ! -f "$INSTALL_DIR/vocabulary.txt" ]; then
    cp vocabulary.txt.example "$INSTALL_DIR/vocabulary.txt"
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "$INSTALL_DIR/venv"

# Install dependencies
echo "Installing Python dependencies (this may take a while)..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install openai-whisper pynput

# Install systemd service
cp push-to-talk.service "$HOME/.config/systemd/user/"

# Reload and enable service
systemctl --user daemon-reload
systemctl --user enable push-to-talk.service
systemctl --user start push-to-talk.service

echo ""
echo "Installation complete!"
echo ""
echo "The service is now running. Hold Right Ctrl to dictate."
echo "A status indicator dot should appear at the top center of your screen."
echo ""
echo "Commands:"
echo "  systemctl --user status push-to-talk   # check status"
echo "  systemctl --user restart push-to-talk  # restart"
echo "  journalctl --user -u push-to-talk -f   # view logs"
