#!/usr/bin/env python3
"""Evaluate the trained HuBERT model on the test set."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from training.train_hubert_fast import SimplifiedHuBERTClassifier, FastDysarthriaDataset
from torch.utils.data import DataLoader


def evaluate_on_test_set():
    """Evaluate trained model on test set."""

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"🔧 Using device: {device}\n")

    # Load test dataset
    test_manifest = Path("data/manifests/test.csv")
    if not test_manifest.exists():
        print(f"❌ Test manifest not found: {test_manifest}")
        return

    test_dataset = FastDysarthriaDataset(test_manifest, max_duration=10.0)
    print(f"📊 Test set: {len(test_dataset)} samples\n")

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # Load trained model
    checkpoint_path = Path("models/hubert_fast_best.pt")
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print("📥 Loading trained model...")
    model = SimplifiedHuBERTClassifier(freeze_base=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded (epoch {checkpoint['epoch']}, val AUC: {checkpoint['val_auc']:.4f})")
    print()

    # Evaluate
    print("🧪 Evaluating on test set...")
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            waveform = batch["waveform"].to(device)
            labels = batch["label"].to(device)

            logits = model(waveform)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*80)
    print("  🎯 TEST SET RESULTS")
    print("="*80)
    print(f"\n📈 Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")
    print()

    print("📊 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              Healthy  Dysarthric")
    print(f"Actual Healthy     {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"       Dysarthric  {cm[1,0]:3d}      {cm[1,1]:3d}")
    print()

    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for dysarthric
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for healthy
    precision_dys = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_healthy = tn / (tn + fn) if (tn + fn) > 0 else 0

    print("🔍 Per-Class Metrics:")
    print(f"  Sensitivity (Dysarthric recall): {sensitivity:.2%}")
    print(f"  Specificity (Healthy recall):    {specificity:.2%}")
    print(f"  Precision (Dysarthric):          {precision_dys:.2%}")
    print(f"  Precision (Healthy):             {precision_healthy:.2%}")
    print()

    print("="*80)
    print("\n✅ Evaluation complete!")
    print(f"\nModel checkpoint: {checkpoint_path}")
    print(f"Test set size: {len(test_dataset)} samples")
    print(f"Final Test AUC: {auc:.4f}")
    print()


if __name__ == "__main__":
    evaluate_on_test_set()
