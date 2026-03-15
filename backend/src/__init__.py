"""
Speech Slurring Detection System
=================================

A clinical-grade dysarthria detection system for brain stroke early detection.

Modules:
    ingestion: Audio loading, preprocessing, and quality validation
    features: Acoustic feature extraction (MFCC, prosody, formants, eGeMAPS)
    models: Deep learning models (HuBERT-SALR, CNN-BiLSTM-Transformer, ensemble)
    scoring: Slurring score and severity classification
    explainability: Grad-CAM, attention rollout, segment localization
    risk: Stroke risk score computation
    reporting: PDF/JSON report generation
    streaming: Real-time streaming analysis (v2)
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Package-level exports
from src.pipeline import SlurringDetectionPipeline

__all__ = ["SlurringDetectionPipeline"]
