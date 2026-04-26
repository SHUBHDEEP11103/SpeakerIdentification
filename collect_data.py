"""
collect_data.py – Record voice samples from a new student.

Usage:
    python collect_data.py --name "StudentName" --samples 8

Each run records `--samples` WAV files into recording2.0/<StudentName>/.
After collecting all students, re-run main.ipynb to retrain the model.
"""

import argparse
import os
import time

SAMPLE_RATE = 16000   # must match training
RECORD_SECS = 3       # seconds per recording
DATA_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording2.0")

PROMPTS = [
    "Say your name naturally (e.g. 'My name is ...')",
    "Say 'I am present' clearly",
    "Count slowly from one to five",
    "Read this sentence: 'The quick brown fox jumps over the lazy dog'",
    "Say today's date",
    "Say your roll number or student ID",
    "Speak freely for 3 seconds (any topic)",
    "Say 'Present sir' or 'Present ma'am'",
]


def record_sample(save_path: str, duration: int = RECORD_SECS) -> None:
    """Record `duration` seconds of audio from the default microphone and save as WAV."""
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write
    except ImportError:
        raise SystemExit(
            "❌  Missing dependencies. Run:  pip install sounddevice scipy"
        )

    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    wav_write(save_path, SAMPLE_RATE, recording)


def collect(name: str, n_samples: int) -> None:
    speaker_dir = os.path.join(DATA_ROOT, name)
    os.makedirs(speaker_dir, exist_ok=True)

    # Find next available index so we never overwrite existing files
    existing = [
        f for f in os.listdir(speaker_dir) if f.startswith("voice_") and f.endswith(".wav")
    ]
    start_idx = len(existing) + 1

    print(f"\n🎤  Recording {n_samples} samples for '{name}'")
    print(f"    Files will be saved to: {speaker_dir}\n")

    for i in range(n_samples):
        idx      = start_idx + i
        filename = f"voice_{idx:03d}.wav"
        filepath = os.path.join(speaker_dir, filename)

        prompt = PROMPTS[i % len(PROMPTS)]
        print(f"  Sample {i + 1}/{n_samples} – {prompt}")
        input("  Press Enter when ready, then speak …")

        print(f"  🔴  Recording for {RECORD_SECS}s …")
        record_sample(filepath)
        print(f"  ✅  Saved  →  {filename}\n")
        time.sleep(0.5)   # brief pause between recordings

    print(f"✅  Done! {n_samples} samples saved for '{name}'.")
    print("   Re-run main.ipynb (or retrain.py) to update the model.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record voice samples for the Speaker Identification attendance system."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Student name (creates recording2.0/<name>/ folder)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of voice samples to record (default: 8)",
    )
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("❌  --samples must be at least 1")

    collect(args.name, args.samples)


if __name__ == "__main__":
    main()
