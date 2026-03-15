#!/usr/bin/env bash
# Complete training pipeline

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════"
echo "  Speech Slurring Detection - Training Pipeline"
echo "════════════════════════════════════════════════════════════════════════════"

# Step 1: Download dataset
echo ""
echo "Step 1: Downloading Kaggle dataset..."
python scripts/download_dataset.py

# Step 2: Build manifests
echo ""
echo "Step 2: Building train/val/test manifests..."
python scripts/build_manifests.py

# Step 3: Train HuBERT-SALR
echo ""
echo "Step 3: Training HuBERT-SALR model..."
python training/train_hubert_salr.py

# Step 4: Train CNN-BiLSTM (if implemented)
# echo ""
# echo "Step 4: Training CNN-BiLSTM model..."
# python training/train_cnn_bilstm.py

# Step 5: Optimize ensemble weights (if implemented)
# echo ""
# echo "Step 5: Optimizing ensemble weights..."
# python training/train_ensemble_weights.py

# Step 6: Calibrate (if implemented)
# echo ""
# echo "Step 6: Calibrating model outputs..."
# python training/calibrate.py

# Step 7: Evaluate
# echo ""
# echo "Step 7: Evaluating trained models..."
# python training/evaluate.py

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  ✓ Training pipeline complete!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Check MLflow UI: http://localhost:5000"
echo "  2. Update model_registry.py to use trained models"
echo "  3. Test the API with new models"
