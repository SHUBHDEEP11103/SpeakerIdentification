"""
attendance.py – Real-time voice-based attendance system.

Usage:
    python attendance.py                     # interactive session
    python attendance.py --file test.wav     # identify from existing file
    python attendance.py --report            # print today's attendance log

Prerequisites:
    1. Run main.ipynb to train and save models/svm_model.pkl
    2. pip install sounddevice scipy librosa scikit-learn joblib
"""

import argparse
import csv
import os
from datetime import datetime

import joblib
import librosa
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "models", "svm_model.pkl")
SCALER_PATH     = os.path.join(BASE_DIR, "models", "svm_scaler.pkl")
ATTENDANCE_LOG  = os.path.join(BASE_DIR, "attendance_log.csv")

# ─── Audio settings (must match training) ─────────────────────────────────────
SAMPLE_RATE        = 16000
RECORD_SECS        = 3
N_MFCC             = 13
CONFIDENCE_THRESH  = 0.50   # predictions below this are rejected as "Unknown"


# ─── Feature extraction (identical to main.ipynb) ─────────────────────────────
def extract_features(file_path: str, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Return a (4 × n_mfcc,) feature vector for one WAV file."""
    audio, _ = librosa.load(file_path, sr=sr)
    audio, _ = librosa.effects.trim(audio)

    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.hstack([
        np.mean(mfcc,   axis=1),
        np.std(mfcc,    axis=1),
        np.mean(delta,  axis=1),
        np.mean(delta2, axis=1),
    ])


# ─── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌  Model not found at {MODEL_PATH}\n"
            "   Please run main.ipynb first to train and save the model."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ─── Prediction ────────────────────────────────────────────────────────────────
def predict_speaker(file_path: str, model, scaler, threshold: float = CONFIDENCE_THRESH):
    """
    Return (speaker_name, confidence).
    Returns ('Unknown', confidence) when confidence < threshold.
    """
    feat        = extract_features(file_path).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    speaker     = model.predict(feat_scaled)[0]
    confidence  = model.predict_proba(feat_scaled).max()

    if confidence < threshold:
        return "Unknown", float(confidence)
    return speaker, float(confidence)


# ─── Recording ─────────────────────────────────────────────────────────────────
def record_voice(save_path: str = "temp_recording.wav", duration: int = RECORD_SECS) -> str:
    """Record `duration` seconds from the microphone and save to `save_path`."""
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write
    except ImportError:
        raise SystemExit("❌  Run:  pip install sounddevice scipy")

    print(f"🔴  Recording for {duration}s … speak now!")
    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    wav_write(save_path, SAMPLE_RATE, recording)
    print("✅  Recording complete.")
    return save_path


# ─── Attendance log ────────────────────────────────────────────────────────────
def _init_log() -> None:
    if not os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "date", "time", "speaker", "confidence"])


def mark_attendance(speaker: str, confidence: float) -> None:
    """Append one attendance record to the CSV log."""
    _init_log()
    now = datetime.now()
    with open(ATTENDANCE_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            now.isoformat(timespec="seconds"),
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            speaker,
            f"{confidence:.4f}",
        ])


def print_report(date_str: str = None) -> None:
    """Print attendance records for a given date (default: today)."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    _init_log()

    print(f"\n{'─' * 48}")
    print(f"  Attendance Report  –  {date_str}")
    print(f"{'─' * 48}")

    found = False
    with open(ATTENDANCE_LOG, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["date"] == date_str and row["speaker"] != "Unknown":
                print(f"  ✅  {row['time']}  –  {row['speaker']}  "
                      f"(conf: {float(row['confidence']):.1%})")
                found = True

    if not found:
        print("  No attendance records found for this date.")
    print(f"{'─' * 48}\n")


# ─── Interactive session ───────────────────────────────────────────────────────
def run_interactive(model, scaler) -> None:
    print("\n🎓  Speaker Identification – Attendance System")
    print("   Press  Ctrl+C  to exit.\n")

    temp_wav = os.path.join(BASE_DIR, "temp_recording.wav")
    marked   = set()   # avoid duplicate entries in the same session

    while True:
        try:
            input("Press Enter to record a student's voice (Ctrl+C to quit) …")
        except KeyboardInterrupt:
            break

        try:
            record_voice(temp_wav)
            speaker, conf = predict_speaker(temp_wav, model, scaler)
        except Exception as exc:
            print(f"⚠️  Error: {exc}\n")
            continue

        if speaker == "Unknown":
            print(f"❓  Speaker not recognised  (confidence: {conf:.1%})\n")
            continue

        print(f"🎤  Identified: {speaker}  (confidence: {conf:.1%})")

        if speaker in marked:
            print(f"ℹ️   {speaker} already marked present today.\n")
        else:
            mark_attendance(speaker, conf)
            marked.add(speaker)
            print(f"✅  Attendance marked for {speaker}\n")

    # Cleanup temp file
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    print_report()


# ─── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice-based attendance system for Speaker Identification project."
    )
    parser.add_argument(
        "--file",
        metavar="WAV",
        help="Identify speaker from an existing WAV file instead of recording live.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print today's attendance report and exit.",
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Date for --report (default: today).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESH,
        help=f"Confidence threshold for accepting a prediction (default: {CONFIDENCE_THRESH}).",
    )
    args = parser.parse_args()

    if args.report:
        print_report(args.date)
        return

    model, scaler = load_model()

    if args.file:
        if not os.path.exists(args.file):
            raise SystemExit(f"❌  File not found: {args.file}")
        speaker, conf = predict_speaker(args.file, model, scaler, args.threshold)
        print(f"🎤  Identified: {speaker}  (confidence: {conf:.1%})")
        if speaker != "Unknown":
            mark_attendance(speaker, conf)
            print(f"✅  Attendance marked for {speaker}")
        return

    run_interactive(model, scaler)


if __name__ == "__main__":
    main()
