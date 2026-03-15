#!/usr/bin/env python3
"""Train HuBERT-SALR model for dysarthria detection."""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow
import yaml

from training.dataset import DysarthriaDataset, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuBERTSALRModel(nn.Module):
    """HuBERT with SALR head for dysarthria detection."""

    def __init__(self, hubert_checkpoint="facebook/hubert-large-ll60k"):
        super().__init__()

        from transformers import HubertModel

        # Load pretrained HuBERT
        self.hubert = HubertModel.from_pretrained(hubert_checkpoint)

        # Freeze feature extractor (optional)
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False

        # Layer-weighted pooling (learnable weights for 24 layers)
        self.layer_weights = nn.Parameter(torch.ones(24) / 24)

        # SALR head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),  # Binary classification
        )

        self.embedder = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Embedding for triplet loss
        )

    def forward(self, waveform):
        """Forward pass."""
        # HuBERT encoding
        outputs = self.hubert(waveform, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # (batch, seq_len, hidden_size) × 24 layers

        # Layer-weighted pooling
        weighted_hidden = torch.stack(
            [self.layer_weights[i] * hidden_states[i] for i in range(24)],
            dim=0
        ).sum(dim=0)  # (batch, seq_len, 1024)

        # Global average pooling
        pooled = weighted_hidden.mean(dim=1)  # (batch, 1024)

        # Classification logits
        logits = self.classifier(pooled)

        # Embeddings for triplet loss
        embeddings = self.embedder(pooled)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return logits, embeddings


def train_hubert_salr(
    train_manifest="data/manifests/train_manifest.csv",
    val_manifest="data/manifests/val_manifest.csv",
    batch_size=8,
    num_epochs=50,
    learning_rate=1e-4,
    device="cuda",
):
    """
    Train HuBERT-SALR model.

    Args:
        train_manifest: Path to training manifest
        val_manifest: Path to validation manifest
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device (cuda/cpu)
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize MLflow
    mlflow.set_experiment("dysarthria_hubert_salr")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model": "HuBERT-SALR",
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
        })

        # Create datasets
        train_dataset = DysarthriaDataset(train_manifest, augment=True)
        val_dataset = DysarthriaDataset(val_manifest, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Disabled for compatibility
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Disabled for compatibility
            collate_fn=collate_fn,
        )

        # Initialize model
        model = HuBERTSALRModel().to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Losses
        ce_loss_fn = nn.CrossEntropyLoss()
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)

        # Training loop
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                waveform = batch["waveform"].to(device)
                labels = batch["label"].squeeze(1).to(device)

                optimizer.zero_grad()

                # Forward pass
                logits, embeddings = model(waveform)

                # Classification loss
                ce_loss = ce_loss_fn(logits, labels)

                # Triplet loss (simplified: use random triplets)
                # In full implementation, use hard negative mining
                triplet_loss = torch.tensor(0.0).to(device)  # Placeholder

                # Combined loss
                loss = ce_loss + 0.5 * triplet_loss

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    waveform = batch["waveform"].to(device)
                    labels = batch["label"].squeeze(1).to(device)

                    logits, _ = model(waveform)
                    loss = ce_loss_fn(logits, labels)

                    val_loss += loss.item()

                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = correct / total

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = Path("models/checkpoints/hubert_salr_best.pt")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(str(checkpoint_path))

        logger.info("Training complete!")


if __name__ == "__main__":
    train_hubert_salr()
