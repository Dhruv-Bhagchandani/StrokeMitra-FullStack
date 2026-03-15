#!/usr/bin/env python3
"""Test the trained HuBERT model on real audio."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import librosa
import numpy as np
from training.train_hubert_fast import SimplifiedHuBERTClassifier


def test_trained_model():
    """Test the trained model on a real dysarthric sample."""

    # Load checkpoint
    checkpoint_path = Path("models/hubert_fast_best.pt")
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"🔧 Using device: {device}")

    # Load model
    print("📥 Loading trained model...")
    model = SimplifiedHuBERTClassifier(freeze_base=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"✓ Best validation AUC: {checkpoint['val_auc']:.4f}")
    print()

    # Test on a few samples (one dysarthric, one healthy)
    test_files = [
        "data/raw/kaggle_dysarthria/Dysarthria and Non Dysarthria/Dataset/Male_dysarthria/M04/Session2/Wav/0298.wav",
        "data/raw/kaggle_dysarthria/Dysarthria and Non Dysarthria/Dataset/Female_Non_Dysarthria/FC03/Session3/Wav/0164.wav",
    ]

    for audio_path in test_files:
        path = Path(audio_path)
        if not path.exists():
            print(f"⚠️  File not found: {path}")
            continue

        # Load audio
        waveform, sr = librosa.load(path, sr=16000)

        # Pad or truncate to 10 seconds
        max_length = 16000 * 10
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        else:
            waveform = np.pad(waveform, (0, max_length - len(waveform)))

        # Convert to tensor
        waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logits = model(waveform_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_class].item()

        # True label from path
        path_lower = str(path).lower()
        # Check for non-dysarthric first (avoid false positives)
        if "non_dysarthria" in path_lower or "non-dysarthria" in path_lower:
            true_label = 0  # Healthy
        elif "dysarthria" in path_lower or "_dysarthria" in path_lower:
            true_label = 1  # Dysarthric
        else:
            true_label = 0  # Default to healthy
        true_label_str = "Dysarthric" if true_label == 1 else "Healthy"
        pred_label_str = "Dysarthric" if pred_class == 1 else "Healthy"

        correct = "✅" if pred_class == true_label else "❌"

        print(f"{'='*80}")
        print(f"Audio: {path.name}")
        print(f"Duration: {len(waveform)/16000:.1f}s")
        print(f"True Label: {true_label_str}")
        print(f"Prediction: {pred_label_str} (confidence: {confidence:.2%}) {correct}")
        print(f"Probabilities: Healthy={probs[0,0]:.2%}, Dysarthric={probs[0,1]:.2%}")
        print()


if __name__ == "__main__":
    test_trained_model()
