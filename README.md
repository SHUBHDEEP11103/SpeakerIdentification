# 🎓 JUET Voice-Based Smart Attendance System

A speaker-identification system that uses deep learning to mark student attendance through voice recognition. Built for **Jaypee University of Engineering and Technology (JUET)**, it extracts a unique "voiceprint" from each student and matches live recordings against enrolled profiles using cosine similarity.

---

## 📖 Overview

Traditional roll-call and card-based attendance systems are slow and prone to proxy attendance. This project replaces them with a voice-based pipeline:

1. **Enroll** – Record short voice samples for each student and store a compact 192-dimensional embedding (voiceprint).
2. **Identify** – Record a new 4-second clip, extract its embedding, and compare it against all enrolled voiceprints using cosine similarity.
3. **Log** – Automatically append the identified student's name, date, time, and confidence score to a CSV file.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔒 Deep Voiceprint | ECAPA-TDNN model pre-trained on VoxCeleb (192-dim embeddings) |
| 🎙️ Live Recording | 4-second microphone capture via `sounddevice` |
| 📊 Cosine Similarity | Threshold-based speaker matching (default: 50 %) |
| 📋 Attendance Log | Auto-saved CSV with name, date, day, time, and confidence |
| ➕ Self-Service Enrolment | Register a new student by recording 3 voice samples in-app |
| 🖥️ Streamlit Web UI | Full-featured dashboard with navigation, stats cards, and live results |
| 📂 Student Directory | Browse all enrolled students and manage profiles |

---

## 🗂️ Repository Structure

```
SpeakerIdentification/
├── app.py                   # Streamlit web application (main entry point)
├── enroll_students.py       # Batch-enrol students from audio files (CLI)
├── take_attendance.py       # Record live audio and identify speaker (CLI)
├── student_embeddings.pt    # Saved voiceprints (generated at enrolment time)
├── attendance_log.csv       # Running attendance log
├── live_attendance.wav      # Temporary file for live microphone recordings
├── recording2.0/            # Voice samples dataset (one sub-folder per student)
│   ├── Shubh/
│   ├── Aditya/
│   └── ...                  # 27 enrolled students
└── pretrained_models/
    └── spkrec-ecapa-voxceleb/  # Cached SpeechBrain model weights
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A working microphone (for live attendance / registration)
- The packages listed below

### Installation

```bash
# Clone the repository
git clone https://github.com/SHUBHDEEP11103/SpeakerIdentification.git
cd SpeakerIdentification

# Install dependencies
pip install torch torchaudio librosa speechbrain sounddevice scipy pandas streamlit
```

> **Note:** `speechbrain` will automatically download the ECAPA-TDNN model weights (~80 MB) from HuggingFace on first run and cache them in `pretrained_models/`.

---

## 🖥️ Usage

### Option A — Streamlit Web App (recommended)

```bash
streamlit run app.py
```

Open the URL printed in your terminal (usually `http://localhost:8501`). The sidebar provides four pages:

| Page | Description |
|---|---|
| **Take Attendance** | Click *Start Recording* → speak → see the result instantly |
| **Attendance Log** | View / download the full attendance CSV |
| **Register New Student** | Enter a name and record 3 × 5-second samples |
| **Student Directory** | Browse all enrolled student profiles |

---

### Option B — Command Line

#### 1. Enrol students (batch)

Place each student's `.wav` files in `recording2.0/<StudentName>/`, then run:

```bash
python enroll_students.py
```

This reads every `.wav` file, extracts embeddings, averages them per student, and saves `student_embeddings.pt`.

#### 2. Take attendance (live)

```bash
python take_attendance.py
```

The script records 4 seconds of audio, computes the voiceprint, and prints the best-matching student together with their cosine-similarity score.

---

## ⚙️ Configuration

All key parameters are defined as constants at the top of each file:

| Constant | Default | Description |
|---|---|---|
| `MODEL_NAME` | `speechbrain/spkrec-ecapa-voxceleb` | Pre-trained speaker model |
| `DATASET_PATH` | `recording2.0` | Root folder for voice samples |
| `EMBEDDING_FILE` | `student_embeddings.pt` | Serialised voiceprints |
| `LOG_FILE` | `attendance_log.csv` | Attendance output CSV |
| `FS` | `16000` | Sample rate (Hz) |
| `SECONDS` | `4` | Recording duration for attendance |
| `SIMILARITY_THRESHOLD` | `0.50` | Minimum cosine similarity to accept a match |

---

## 🧠 How It Works

```
Enrolment                              Identification
──────────────────────────────────     ──────────────────────────────────
Student speaks  →  .wav files          Mic records 4 s  →  live.wav
       ↓                                      ↓
  librosa loads audio (16 kHz)         librosa loads audio (16 kHz)
       ↓                                      ↓
  ECAPA-TDNN encoder                   ECAPA-TDNN encoder
       ↓                                      ↓
  192-dim embedding per file           192-dim live embedding
       ↓                                      ↓
  Average all files for student        Cosine similarity vs every
       ↓                               enrolled embedding
  Save to student_embeddings.pt              ↓
                                       Best score ≥ threshold?
                                         Yes → mark attendance ✅
                                         No  → unrecognised ❌
```

---

## 📦 Key Dependencies

| Library | Purpose |
|---|---|
| [SpeechBrain](https://speechbrain.github.io/) | ECAPA-TDNN speaker encoder |
| [PyTorch](https://pytorch.org/) | Tensor operations & cosine similarity |
| [librosa](https://librosa.org/) | Audio loading and resampling |
| [sounddevice](https://python-sounddevice.readthedocs.io/) | Microphone recording |
| [Streamlit](https://streamlit.io/) | Web UI framework |
| pandas | CSV attendance logging |

---

## 📄 Attendance Log Format

`attendance_log.csv` has the following columns:

| Name | Date | Day | Time | Confidence Score |
|---|---|---|---|---|
| SHUBH | 2025-06-01 | Sunday | 10:23:45 | 78.34% |

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add my feature"`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Open a Pull Request.
