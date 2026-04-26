# Speaker Identification – Voice-Based Attendance System

A voice-based attendance system that uses **MFCC feature extraction** and a **Support Vector Machine (SVM)** classifier to identify students from short voice recordings and automatically mark their attendance.

---

## Project Structure

```
SpeakerIdentification/
├── main.ipynb           # Training notebook (feature extraction + KNN + SVM)
├── collect_data.py      # Record voice samples for new students
├── attendance.py        # Live attendance marking system
├── requirements.txt     # Python dependencies
├── recording2.0/        # Voice dataset (one sub-folder per speaker)
│   ├── Abhishek/
│   ├── Adamya/
│   └── ...
├── models/              # Saved models (created after running main.ipynb)
│   ├── svm_model.pkl
│   ├── svm_scaler.pkl
│   ├── knn_model.pkl
│   └── knn_scaler.pkl
└── attendance_log.csv   # Auto-generated attendance records
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Open and run **`main.ipynb`** from top to bottom.  
The notebook will:
- Extract MFCC features from all recordings in `recording2.0/`
- Train both a KNN and an SVM model
- Print accuracy and a classification report
- Save models to `models/`

### 3. Add new students (optional)

```bash
python collect_data.py --name "StudentName" --samples 8
```

Re-run `main.ipynb` after adding new students to retrain the model.

### 4. Run the attendance system

```bash
# Interactive mode (microphone required)
python attendance.py

# Identify from an existing WAV file
python attendance.py --file test.wav

# Print today's attendance report
python attendance.py --report
```

---

## How It Works

```
Student speaks
      ↓
Record 3-second WAV (16 kHz)
      ↓
Extract MFCC features (13 coefficients + deltas → 52-dim vector)
      ↓
Normalise with StandardScaler
      ↓
SVM (RBF kernel) predicts speaker + confidence score
      ↓
If confidence ≥ 50% → mark attendance in CSV log
```

### Why MFCC?

**Mel-Frequency Cepstral Coefficients** convert raw audio into a compact "voice fingerprint":
- Captures unique vocal-tract characteristics of each person
- Robust to small volume and speed changes
- Used in all modern speech/speaker recognition systems

### Why SVM?

- Works well with small datasets (7–8 samples per speaker)
- High-dimensional MFCC feature space is ideal for SVM's hyperplane separation
- `probability=True` gives calibrated confidence scores to reject unknown speakers

---

## Dataset

The `recording2.0/` folder contains recordings from **14 students**:

| Speaker | Samples |
|---------|---------|
| Abhishek | 6 |
| Adamya | 8 |
| Aditya | 8 |
| Danish | 8 |
| Madhur | 8 |
| Sarvagya | 8 |
| Saurabh | 8 |
| ShubhamB | 7 |
| ShubhamJ | 8 |
| Swayam | 8 |
| Tanish | 8 |
| Utkarsh | 8 |
| Vaibhav | 8 |
| Vivek | 8 |

Each recording is a `.wav` file sampled at 16 kHz.

---

## Configuration

| Parameter | Default | Location |
|-----------|---------|----------|
| Sample rate | 16 000 Hz | `main.ipynb`, `attendance.py` |
| MFCC coefficients | 13 | `main.ipynb`, `attendance.py` |
| Recording duration | 3 s | `collect_data.py`, `attendance.py` |
| Confidence threshold | 50 % | `attendance.py` (`--threshold`) |

---

## Workflow: Adding a New Batch of Students

```
1. python collect_data.py --name "Alice" --samples 8
2. python collect_data.py --name "Bob"   --samples 8
   … (repeat for each student)
3. Open main.ipynb → Run All
4. python attendance.py
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `sounddevice` not found | `pip install sounddevice scipy` |
| Model not found | Run main.ipynb to train and save models |
| Low accuracy | Collect more samples per student (aim for 8–10) |
| Wrong speaker identified | Record samples in the same environment as deployment |
| `Unknown` returned | Speak for the full 3 seconds; adjust `--threshold` |
