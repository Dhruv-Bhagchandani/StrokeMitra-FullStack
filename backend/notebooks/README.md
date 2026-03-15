# Analysis Notebooks

This directory contains Jupyter notebooks for exploring data, analyzing features, and understanding model behavior.

## 📓 Notebooks

### [01_data_exploration.ipynb](01_data_exploration.ipynb)
**Data Exploration and Dataset Statistics**

- Dataset statistics (train/val/test splits)
- Class distribution analysis
- Audio duration analysis
- Sample audio playback
- Waveform and spectrogram visualization
- Summary statistics

**Usage:**
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

### [02_feature_analysis.ipynb](02_feature_analysis.ipynb)
**Acoustic Feature Analysis**

- MFCC distributions
- Prosodic features (F0, energy, speaking rate, pauses)
- Formant features (F1, F2, vowel space)
- Statistical significance tests (t-tests)
- Feature correlation analysis
- Feature separability visualization

**Usage:**
```bash
jupyter notebook notebooks/02_feature_analysis.ipynb
```

---

### [03_explainability_analysis.ipynb](03_explainability_analysis.ipynb)
**Model Interpretability and Explainability**

- Grad-CAM heatmap visualization
- Segment-level prediction localization
- Attention weight analysis
- Feature importance scores
- Model decision visualization

**Usage:**
```bash
jupyter notebook notebooks/03_explainability_analysis.ipynb
```

---

## 🚀 Getting Started

### Prerequisites

Install Jupyter and required dependencies:

```bash
poetry install
poetry run pip install jupyter ipykernel
```

### Launch Jupyter Lab

```bash
cd notebooks/
poetry run jupyter lab
```

Or with classic Jupyter Notebook:

```bash
poetry run jupyter notebook
```

---

## 📊 Recommended Workflow

1. **Start with Data Exploration** (`01_data_exploration.ipynb`)
   - Understand dataset composition
   - Verify data quality
   - Identify any class imbalance

2. **Analyze Features** (`02_feature_analysis.ipynb`)
   - Examine feature distributions
   - Test statistical significance
   - Identify discriminative features

3. **Understand Model Decisions** (`03_explainability_analysis.ipynb`)
   - Visualize model attention
   - Interpret predictions
   - Validate clinical relevance

---

## 💡 Tips

- **Large Datasets**: If you have many audio files, notebooks sample a subset to avoid long processing times
- **GPU Acceleration**: Notebooks detect available GPUs automatically
- **Interactive Widgets**: Some notebooks use interactive plots (Plotly) for better exploration
- **Reproducibility**: Set random seeds at the beginning of each notebook for reproducible results

---

## 🔧 Troubleshooting

**Kernel Not Found:**
```bash
poetry run python -m ipykernel install --user --name=slurring_detection
```

**Module Import Errors:**
Ensure you're running Jupyter from the project root:
```bash
cd /path/to/slurry_speech
poetry run jupyter lab
```

**Memory Issues:**
Reduce sample sizes in notebook code cells (look for `sample(n=...)` calls)

---

## 📝 Adding New Notebooks

When creating new notebooks:

1. Use the project root as the base path:
   ```python
   import sys
   sys.path.insert(0, '..')
   ```

2. Load configs from the `configs/` directory
3. Save outputs to `reports/` or `notebooks/outputs/`
4. Document your analysis with markdown cells
5. Update this README with a description of the new notebook

---

## 📚 Additional Resources

- [Feature Extraction Documentation](../src/features/README.md)
- [Model Architecture Documentation](../src/models/README.md)
- [API Documentation](../api/README.md)
- [Training Pipeline](../training/README.md)

---

**Questions?** Check the main [project README](../README.md) or open an issue on GitHub.
