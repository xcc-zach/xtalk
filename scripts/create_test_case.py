# scripts/create_test_case.py
"""
Create test cases for offline_client.py from transcription text files.

This script reads a transcription file with lines in the format:
    <timestamp/ai_end>:<text_to_transcribe>

It uses DashScope TTS API to generate audio files and creates an output
directory compatible with scripts/offline_client.py.

Dependencies:
    pip install requests

Usage:
    python scripts/create_test_case.py --input transcription.txt --output test_case_dir

Environment:
    DASHSCOPE_API_KEY: Required. Your DashScope API key.

Example transcription.txt:
    0:Hello, how are you?
    ai_end:I have another question.
    5.0:This will be sent at 5 seconds.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import requests


@dataclass
class TranscriptionEntry:
    """A transcription entry with timing and text."""

    timing: Union[
        float, str
    ]  # float = seconds, "ai_end" = wait for response
    text: str
    audio_filename: str = ""


def parse_transcription_file(input_path: str) -> List[TranscriptionEntry]:
    """Parse transcription file with format: <timestamp>:<text>"""
    entries = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Find the first colon to split timing from text
            colon_idx = line.find(":")
            if colon_idx == -1:
                raise ValueError(
                    f"Line {line_num}: Invalid format, expected '<timing>:<text>'"
                )

            timing_str = line[:colon_idx].strip()
            text = line[colon_idx + 1 :].strip()

            if not text:
                raise ValueError(f"Line {line_num}: Empty text content")

            # Parse timing
            if timing_str == "ai_end":
                timing = "ai_end"
            else:
                try:
                    timing = float(timing_str)
                except ValueError:
                    raise ValueError(
                        f"Line {line_num}: Invalid timing '{timing_str}', "
                        "expected number or 'ai_end'"
                    )

            entries.append(TranscriptionEntry(timing=timing, text=text))

    return entries


def generate_audio_dashscope(
    text: str,
    api_key: str,
    voice: str = "Cherry",
    language_type: str = "Auto",
) -> bytes:
    """Generate audio using DashScope TTS API."""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "qwen3-tts-flash",
        "input": {
            "text": text,
            "voice": voice,
            "language_type": language_type,
        },
    }

    print(f"  Requesting TTS for: {text[:50]}...")

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()

    if "output" not in result or "audio" not in result["output"]:
        raise ValueError(f"Unexpected API response: {result}")

    audio_url = result["output"]["audio"].get("url")
    if not audio_url:
        raise ValueError(f"No audio URL in response: {result}")

    print(f"  Downloading audio from: {audio_url[:80]}...")

    # Download the audio file
    audio_response = requests.get(audio_url, timeout=60)
    audio_response.raise_for_status()

    return audio_response.content


def create_test_case(
    entries: List[TranscriptionEntry],
    output_dir: str,
    api_key: str,
    voice: str = "Cherry",
    language_type: str = "Audo",
) -> None:
    """Create test case directory with audio files and timestamp.txt."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp_lines = []

    for i, entry in enumerate(entries):
        # Generate audio filename
        audio_filename = f"audio_{i:03d}.wav"
        audio_path = output_path / audio_filename

        print(f"\n[{i + 1}/{len(entries)}] Processing entry:")
        print(f"  Timing: {entry.timing}")
        print(f"  Text: {entry.text}")

        # Generate audio
        try:
            audio_data = generate_audio_dashscope(
                text=entry.text,
                api_key=api_key,
                voice=voice,
                language_type=language_type,
            )

            # Save audio file
            with open(audio_path, "wb") as f:
                f.write(audio_data)

            print(f"  Saved: {audio_path} ({len(audio_data)} bytes)")

            # Add to timestamp file
            timestamp_lines.append(f"{audio_filename}:{entry.timing}")

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR: Failed to generate audio: {e}")
            raise

    # Write timestamp.txt
    timestamp_path = output_path / "timestamp.txt"
    with open(timestamp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(timestamp_lines) + "\n")

    print(f"\n[Done] Created test case at: {output_path}")
    print(f"  Audio files: {len(entries)}")
    print(f"  Timestamp file: {timestamp_path}")
    print(f"\nRun with:")
    print(f"  python scripts/offline_client.py --input {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create test cases for offline_client.py from transcription files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --input transcription.txt --output test_case_dir
  python %(prog)s --input transcription.txt --output test_case_dir --voice Cherry
  python %(prog)s --input transcription.txt --output test_case_dir --language English

Transcription file format (one entry per line):
  <timing>:<text_to_transcribe>

Where <timing> is:
  - A float number: seconds from start (e.g., 0, 5.0, 10.5)
  - "ai_end": wait for previous response to finish

Example transcription.txt:
  0:Hello, how are you?
  ai_end:I have another question.
  5.0:This will be sent at 5 seconds.
        """,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to transcription file with format '<timing>:<text>' per line",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for test case (audio files + timestamp.txt)",
    )
    parser.add_argument(
        "--voice",
        default="Cherry",
        help="TTS voice name (default: Cherry)",
    )
    parser.add_argument(
        "--language",
        default="Auto",
        choices=[
            "Chinese",
            "English",
            "German",
            "Italian",
            "Portuguese",
            "Spanish",
            "Japanese",
            "Korean",
            "French",
            "Russian",
            "Auto",
        ],
        help="Language type for TTS (default: Auto)",
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY environment variable is required")
        sys.exit(1)

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Parse transcription file
    print(f"Parsing transcription file: {args.input}")
    entries = parse_transcription_file(args.input)
    print(f"Found {len(entries)} entries")

    # Create test case
    create_test_case(
        entries=entries,
        output_dir=args.output,
        api_key=api_key,
        voice=args.voice,
        language_type=args.language,
    )


if __name__ == "__main__":
    main()
