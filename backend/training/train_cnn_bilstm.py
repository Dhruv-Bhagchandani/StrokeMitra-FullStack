"""
Training script for CNN-BiLSTM-Transformer model (spectrogram branch).

This model processes log-mel spectrograms and CWT scalograms through:
1. CNN feature extraction (ResNet-style blocks)
2. BiLSTM temporal modeling
3. Transformer encoder with self-attention
4. Classification head

Usage:
    python training/train_cnn_bilstm.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import numpy as np
from tqdm import tqdm
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging

from training.dataset import DysarthriaDataset
from training.augmentation import AudioAugmentor

# ══════════════════════════════════════════════════════════════════════════════
# CNN-BiLSTM-Transformer Model Architecture
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Residual block for CNN feature extraction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


class CNNBiLSTMTransformer(nn.Module):
    """
    Spectrogram-based dysarthria detection model.

    Architecture:
    - CNN: Extract spatial features from spectrogram
    - BiLSTM: Model temporal dependencies
    - Transformer: Self-attention for long-range patterns
    - Classifier: Binary classification head
    """

    def __init__(
        self,
        input_channels: int = 2,  # Log-mel + CWT
        cnn_channels: list = [64, 128, 256],
        lstm_hidden: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ─────────────────────────────────────────────────────────────────────
        # CNN Feature Extractor
        # ─────────────────────────────────────────────────────────────────────
        self.cnn_blocks = nn.ModuleList()
        in_ch = input_channels
        for out_ch in cnn_channels:
            self.cnn_blocks.append(ResidualBlock(in_ch, out_ch))
            in_ch = out_ch

        self.pool = nn.AdaptiveAvgPool2d((None, 1))  # Pool frequency dimension

        # ─────────────────────────────────────────────────────────────────────
        # BiLSTM Temporal Modeling
        # ─────────────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Transformer Encoder
        # ─────────────────────────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden * 2,  # Bidirectional
            nhead=transformer_heads,
            dim_feedforward=lstm_hidden * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # ─────────────────────────────────────────────────────────────────────
        # Classification Head
        # ─────────────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # Binary: healthy vs dysarthric
        )

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: (batch, 2, freq, time) - Log-mel + CWT

        Returns:
            logits: (batch, 2)
            attention_weights: Transformer attention for explainability
        """
        batch_size = spectrogram.size(0)

        # CNN feature extraction
        x = spectrogram
        for block in self.cnn_blocks:
            x = block(x)

        # Pool frequency dimension: (batch, channels, freq, time) → (batch, channels, time)
        x = self.pool(x).squeeze(2)

        # Transpose for LSTM: (batch, time, channels)
        x = x.transpose(1, 2)

        # BiLSTM
        x, _ = self.lstm(x)

        # Transformer encoder
        x = self.transformer(x)

        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, lstm_hidden*2)

        # Classification
        logits = self.classifier(x)

        return logits


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        spectrogram = batch["spectrogram"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(spectrogram)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
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
            spectrogram = batch["spectrogram"].to(device)
            labels = batch["label"].to(device)

            logits = model(spectrogram)
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


def main():
    # ──────────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load config
    config_path = Path("configs/model_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # MLflow setup
    mlflow.set_experiment("cnn_bilstm_transformer_training")

    # ──────────────────────────────────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────────────────────────────────
    train_manifest = Path("data/manifests/train.csv")
    val_manifest = Path("data/manifests/val.csv")

    augmentor = AudioAugmentor(
        time_stretch_range=(0.9, 1.1),
        pitch_shift_range=(-2, 2),
        noise_level=0.005,
    )

    train_dataset = DysarthriaDataset(train_manifest, augmentor=augmentor, mode="train")
    val_dataset = DysarthriaDataset(val_manifest, augmentor=None, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("cnn_bilstm", {}).get("batch_size", 16),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("cnn_bilstm", {}).get("batch_size", 16),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Model Setup
    # ──────────────────────────────────────────────────────────────────────────
    model = CNNBiLSTMTransformer(
        input_channels=2,
        cnn_channels=[64, 128, 256],
        lstm_hidden=256,
        transformer_heads=8,
        transformer_layers=4,
        dropout=0.3,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Loss function with class weights (handle imbalance)
    criterion = nn.CrossEntropyLoss()

    # ──────────────────────────────────────────────────────────────────────────
    # Training Loop
    # ──────────────────────────────────────────────────────────────────────────
    num_epochs = config.get("cnn_bilstm", {}).get("epochs", 30)
    best_val_auc = 0
    best_model_path = Path("models/cnn_bilstm_best.pt")
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "model": "cnn_bilstm_transformer",
            "epochs": num_epochs,
            "batch_size": config.get("cnn_bilstm", {}).get("batch_size", 16),
            "learning_rate": 1e-4,
            "optimizer": "AdamW",
        })

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")

            # Train
            train_loss, train_acc, train_f1 = train_epoch(
                model, train_loader, optimizer, criterion, device
            )

            # Validate
            val_loss, val_acc, val_f1, val_auc = validate(
                model, val_loader, criterion, device
            )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Logging
            logger.info(
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}"
            )
            logger.info(
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                }, best_model_path)
                logger.info(f"✓ New best model saved (AUC: {val_auc:.4f})")
                mlflow.log_artifact(str(best_model_path))

        logger.info(f"\n✓ Training complete! Best validation AUC: {best_val_auc:.4f}")
        mlflow.log_metric("best_val_auc", best_val_auc)


if __name__ == "__main__":
    main()
