import os
import torch
import librosa
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

# We use the state-of-the-art ECAPA-TDNN model pre-trained on VoxCeleb.
# VoxCeleb contains thousands of hours of speech from thousands of people in real-world environments.
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
DATASET_PATH = "recording2.0"
EMBEDDING_FILE = "student_embeddings.pt"

print(f"Loading pre-trained deep learning model: {MODEL_NAME}...")
# This will download the model weights the first time it is run (approx 80MB)
classifier = EncoderClassifier.from_hparams(source=MODEL_NAME, savedir="pretrained_models/spkrec-ecapa-voxceleb")

student_embeddings = {}

print("\nExtracting robust Voiceprints (Embeddings)...")
for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)
    if os.path.isdir(person_path):
        person_embs = []
        
        for file in os.listdir(person_path):
            if file.endswith(".wav"):
                file_path = os.path.join(person_path, file)
                try:
                    # Load audio using librosa to avoid torchcodec issues
                    audio, fs = librosa.load(file_path, sr=16000)
                    
                    # Convert to PyTorch tensor (1, num_samples)
                    signal = torch.from_numpy(audio).unsqueeze(0)
                    
                    if signal.shape[0] > 1:
                        signal = signal.mean(dim=0, keepdim=True)
                        
                    # Extract the Voiceprint (a 192-dimensional vector)
                    embeddings = classifier.encode_batch(signal)
                    
                    # Squeeze out the batch dimensions
                    embeddings = embeddings.squeeze(0).squeeze(0)
                    
                    person_embs.append(embeddings)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        if len(person_embs) > 0:
            # Average the embeddings of all recordings to get a single, highly robust Voiceprint
            # We use torch.stack to combine the list of tensors, then calculate the mean
            avg_emb = torch.mean(torch.stack(person_embs), dim=0)
            student_embeddings[person] = avg_emb
            print(f"✅ Enrolled {person} (used {len(person_embs)} files)")

# Save the dictionary of voiceprints to disk
torch.save(student_embeddings, EMBEDDING_FILE)
print(f"\n🎉 Successfully enrolled {len(student_embeddings)} students! Saved to {EMBEDDING_FILE}")
