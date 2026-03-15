#!/usr/bin/env python3
"""
Fast fine-tuning script for HuBERT-SALR model.

Optimizations:
- Reduced dataset size (500-1000 samples)
- Fewer epochs (5 instead of 20)
- Simplified model architecture
- Uses MPS/GPU acceleration
- Faster feature extraction

Usage:
    python training/train_hubert_fast.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import HubertModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Simplified HuBERT Model
# ══════════════════════════════════════════════════════════════════════════════

class SimplifiedHuBERTClassifier(nn.Module):
    """Simplified HuBERT for faster training."""

    def __init__(self, freeze_base=True):
        super().__init__()

        # Load pre-trained HuBERT (smaller version for speed)
        logger.info("Loading HuBERT-base model...")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        # Freeze base model for faster training
        if freeze_base:
            for param in self.hubert.parameters():
                param.requires_grad = False
            logger.info("✓ HuBERT base frozen (only training classifier)")

        # Simple classifier head
        hidden_size = self.hubert.config.hidden_size  # 768 for base
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),  # Binary: healthy vs dysarthric
        )

    def forward(self, input_values):
        # Extract features
        with torch.no_grad() if self.training else torch.enable_grad():
            outputs = self.hubert(input_values)

        # Pool: mean across time dimension
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden)
        pooled = hidden_states.mean(dim=1)  # (batch, hidden)

        # Classify
        logits = self.classifier(pooled)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# Fast Dataset (No Heavy Feature Extraction)
# ══════════════════════════════════════════════════════════════════════════════

class FastDysarthriaDataset(torch.utils.data.Dataset):
    """Simplified dataset for fast training."""

    def __init__(self, manifest_path, max_duration=10.0, sample_rate=16000):
        self.manifest = pd.read_csv(manifest_path)
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)

        # Filter valid files
        self.manifest = self.manifest[
            (self.manifest['duration'] >= 5.0) &  # Min duration
            (self.manifest['duration'] <= max_duration)  # Max duration
        ].reset_index(drop=True)

        logger.info(f"Dataset: {len(self.manifest)} samples (filtered for 5-10s duration)")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        # Load audio
        import librosa
        waveform, sr = librosa.load(row['file_path'], sr=self.sample_rate)

        # Pad or truncate to fixed length
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))

        return {
            'waveform': torch.FloatTensor(waveform),
            'label': int(row['label']),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Training Functions
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(waveform)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, accuracy, f1


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            waveform = batch["waveform"].to(device)
            labels = batch["label"].to(device)

            logits = model(waveform)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, f1, auc


# ══════════════════════════════════════════════════════════════════════════════
# Main Training
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"🚀 Using device: {device}")

    # Load datasets
    train_manifest = Path("data/manifests/train.csv")
    val_manifest = Path("data/manifests/val.csv")

    train_dataset = FastDysarthriaDataset(train_manifest, max_duration=10.0)
    val_dataset = FastDysarthriaDataset(val_manifest, max_duration=10.0)

    # Use subset for faster training
    MAX_TRAIN_SAMPLES = 500  # Reduced from 3000
    MAX_VAL_SAMPLES = 100    # Reduced from 647

    if len(train_dataset) > MAX_TRAIN_SAMPLES:
        indices = np.random.choice(len(train_dataset), MAX_TRAIN_SAMPLES, replace=False)
        train_dataset = Subset(train_dataset, indices)
        logger.info(f"✂️  Using subset: {MAX_TRAIN_SAMPLES} training samples")

    if len(val_dataset) > MAX_VAL_SAMPLES:
        indices = np.random.choice(len(val_dataset), MAX_VAL_SAMPLES, replace=False)
        val_dataset = Subset(val_dataset, indices)
        logger.info(f"✂️  Using subset: {MAX_VAL_SAMPLES} validation samples")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Small batch for speed
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # Model
    model = SimplifiedHuBERTClassifier(freeze_base=True).to(device)
    logger.info(f"✓ Model loaded on {device}")

    # Optimizer and loss
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)  # Higher LR for frozen base
    criterion = nn.CrossEntropyLoss()

    # Training loop
    NUM_EPOCHS = 5  # Reduced from 20
    best_val_auc = 0
    best_model_path = Path("models/hubert_fast_best.pt")
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"  FAST TRAINING - {NUM_EPOCHS} epochs")
    logger.info(f"{'='*80}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        logger.info("-" * 40)

        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc, val_f1, val_auc = validate(
            model, val_loader, criterion, device
        )

        # Log
        logger.info(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")
        logger.info(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, best_model_path)
            logger.info(f"✓ New best model saved (AUC: {val_auc:.4f})")

    logger.info(f"\n{'='*80}")
    logger.info(f"  ✓ TRAINING COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")
    logger.info(f"Model saved to: {best_model_path}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Test the model on test set")
    logger.info(f"  2. Update model_registry.py to use this checkpoint")
    logger.info(f"  3. Run inference on new audio files")


if __name__ == "__main__":
    main()
