import os
import torch
import librosa
import torchaudio
import sounddevice as sd
from scipy.io.wavfile import write
from speechbrain.inference.classifiers import EncoderClassifier

MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_FILE = "student_embeddings.pt"
TEST_FILE = "live_attendance.wav"
FS = 16000
SECONDS = 4
SIMILARITY_THRESHOLD = 0.50 # You can adjust this. 0.5 is usually a good threshold for cosine similarity.

def take_attendance():
    if not os.path.exists(EMBEDDING_FILE):
        print(f"❌ Error: {EMBEDDING_FILE} not found. Please run enroll_students.py first.")
        return

    print("Loading pre-trained model and student voiceprints...")
    classifier = EncoderClassifier.from_hparams(source=MODEL_NAME, savedir="pretrained_models/spkrec-ecapa-voxceleb")
    
    # Load the dictionary of enrolled embeddings
    student_embeddings = torch.load(EMBEDDING_FILE)
    print(f"Loaded {len(student_embeddings)} enrolled students.")

    # 1. Record live audio
    print("\n" + "="*40)
    print(f"🎙️ PLEASE SPEAK YOUR NAME FOR ATTENDANCE ({SECONDS} SECONDS)...")
    print("="*40)
    
    recording = sd.rec(int(SECONDS * FS), samplerate=FS, channels=1)
    sd.wait()
    write(TEST_FILE, FS, recording)
    
    print("\n✅ Recording saved. Analyzing Voiceprint...")

    # 2. Extract embedding for live audio
    audio, fs = librosa.load(TEST_FILE, sr=16000)
    signal = torch.from_numpy(audio).unsqueeze(0)
    
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
        
    live_embedding = classifier.encode_batch(signal)
    live_embedding = live_embedding.squeeze(0).squeeze(0)

    # 3. Compare with all enrolled students using Cosine Similarity
    best_match = None
    best_score = -1.0
    
    print("\n📊 Similarity Scores:")
    for name, enrolled_emb in student_embeddings.items():
        # Cosine similarity measures the angle between the two high-dimensional vectors
        # Score is between -1 and 1. Higher is better.
        similarity = torch.nn.functional.cosine_similarity(live_embedding.unsqueeze(0), enrolled_emb.unsqueeze(0))
        score = similarity.item()
        
        print(f"  {name}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_match = name

    # 4. Final Decision
    print("\n" + "="*40)
    if best_score >= SIMILARITY_THRESHOLD:
        print(f"🎯 ATTENDANCE MARKED FOR: {best_match.upper()}")
        print(f"   Match Confidence: {best_score:.2%}")
    else:
        print(f"❌ UNRECOGNIZED SPEAKER. Best match was {best_match} ({best_score:.2%}), but it was below the threshold ({SIMILARITY_THRESHOLD:.2%}).")
        print("   Please try again or speak more clearly.")
    print("="*40)

if __name__ == "__main__":
    take_attendance()
