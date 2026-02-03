# Push-to-Talk Dictation Service

A Linux push-to-talk dictation service using OpenAI Whisper for speech recognition. Hold a key to record, release to transcribe and type into the focused input.

## Features

- **Push-to-talk recording** - Hold Right Ctrl to record, release to transcribe
- **Local speech recognition** - Uses OpenAI Whisper (no cloud required)
- **Status indicator** - Visual dot showing recording/processing state
- **Hover menu** - View status, recent transcriptions, manage service
- **Custom vocabulary** - Teach it unusual words for better accuracy
- **Auto-start** - Runs as a systemd user service

## Requirements

- Linux with PipeWire audio
- Python 3.10+
- GTK 3 (for indicator)
- xdotool (for typing)
- ffmpeg (for audio conversion)

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd push-to-talk

# Run the installer
./install.sh
```

## Usage

### Basic Dictation
1. Focus a text input (browser, editor, terminal, etc.)
2. Hold **Right Ctrl**
3. Speak your text
4. Release **Right Ctrl**
5. Text appears in the focused input

### AI Assistant Mode
1. Hold **Right Ctrl + Right Shift**
2. Ask your question
3. Release to send to Claude AI
4. Listen to the spoken response
5. Auto-listen kicks in for follow-up questions
6. Press **Escape** to interrupt the assistant at any time

### Status Indicator
- **Gray dot** - Idle, ready
- **Red dot** - Recording
- **Yellow dot** - Processing/transcribing
- **Green dot** - Success (briefly)

Hover over the dot to see:
- Current status
- Vocabulary word count
- Recent transcriptions
- Buttons: Restart, Vocabulary, Logs

### Teaching Vocabulary
Say these commands (they won't be typed):
- `"add word: Kubernetes"`
- `"correction: PostgreSQL"`
- `"remember: MyCompanyName"`

Or edit `~/.local/share/push-to-talk/vocabulary.txt` directly.

## Configuration

Edit `~/.local/share/push-to-talk/push-to-talk.py` to change:
- `PTT_KEY` - The push-to-talk key (default: Right Ctrl)
- `WHISPER_MODEL` - Model size: tiny, base, small, medium, large

## Service Management

```bash
# Check status
systemctl --user status push-to-talk

# Restart
systemctl --user restart push-to-talk

# Stop
systemctl --user stop push-to-talk

# View logs
journalctl --user -u push-to-talk -f

# Disable auto-start
systemctl --user disable push-to-talk
```

## Files

- `~/.local/share/push-to-talk/` - Runtime files
  - `push-to-talk.py` - Main service
  - `indicator.py` - Status indicator
  - `vocabulary.txt` - Custom words
  - `venv/` - Python virtual environment
- `~/.config/systemd/user/push-to-talk.service` - Service definition

## Uninstall

```bash
./uninstall.sh
```

## License

MIT
