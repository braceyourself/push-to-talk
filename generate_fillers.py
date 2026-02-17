#!/usr/bin/env python3
"""
One-time script to generate filler WAV clips using OpenAI TTS.
Produces PCM 24kHz 16-bit mono WAVs for conversational presence.

Usage: python generate_fillers.py [--voice ash]
"""

import sys
import wave
import argparse
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required. pip install openai")
    sys.exit(1)


FILLERS = {
    "acknowledge": [
        ("mhm_1", "Mhm."),
        ("mhm_2", "Mm-hmm."),
        ("right_1", "Right."),
        ("got_it_1", "Got it."),
        ("yeah_1", "Yeah."),
        ("okay_1", "Okay."),
    ],
    "thinking": [
        ("hmm_1", "Hmm."),
        ("let_me_think_1", "Let me think."),
        ("so_1", "So..."),
        ("hmm_2", "Hmm, let's see."),
    ],
    "tool_use": [
        ("let_me_check_1", "Let me check."),
        ("one_moment_1", "One moment."),
        ("on_it_1", "On it."),
        ("working_on_that_1", "Working on that."),
    ],
}

SAMPLE_RATE = 24000


def generate_fillers(voice="ash", output_dir=None):
    """Generate all filler WAV files."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "audio" / "fillers"

    output_dir = Path(output_dir)

    # Check for API key
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        key_file = Path.home() / ".config" / "openai" / "api_key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        print("Error: Set OPENAI_API_KEY or create ~/.config/openai/api_key")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    total = sum(len(clips) for clips in FILLERS.values())
    generated = 0

    for category, clips in FILLERS.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for filename, text in clips:
            wav_path = cat_dir / f"{filename}.wav"

            if wav_path.exists():
                print(f"  Skip (exists): {wav_path}")
                generated += 1
                continue

            print(f"  Generating: {wav_path} -- \"{text}\"")

            try:
                response = client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                    response_format="pcm"
                )

                # Collect all PCM bytes
                pcm_data = b""
                for chunk in response.iter_bytes(chunk_size=4096):
                    pcm_data += chunk

                # Write as WAV
                with wave.open(str(wav_path), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(pcm_data)

                generated += 1
                print(f"    OK ({len(pcm_data)} bytes)")

            except Exception as e:
                print(f"    Error: {e}")

    print(f"\nDone: {generated}/{total} fillers generated in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate filler WAV clips via OpenAI TTS")
    parser.add_argument("--voice", default="ash", help="OpenAI TTS voice (default: ash)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    generate_fillers(voice=args.voice, output_dir=args.output_dir)
