# Electrofacies

ML system for automated lithofacies classification in the Delaware Mountain Group (Permian Basin) from well-log data.

Upload a LAS file, get back facies predictions with confidence scores and QC flags.

## Facies Classes

| Facies | Description |
|--------|-------------|
| Massive Sandstone | Structureless, gravity-flow deposits |
| Structured Sandstone | Laminated, tractional structures |
| Sandy Siltstone | Transitional, mixed grain size |
| Siltstone | Fine-grained, low-energy |
| Calciturbidite | Calcareous gravity-flow deposits |
| Clast-Supported Conglomerate | High-energy channel deposits |

## Model Architecture

The system uses a **4-tier routing** approach that automatically selects the best model based on which logs are available in the uploaded well:

| Tier | Required Logs | Features |
|------|--------------|----------|
| Tier 1 — Full Suite | GR, RESD, RHOB, NPHI, DTC | 27 |
| Tier 2 — No Sonic | GR, RESD, RHOB, NPHI | 22 |
| Tier 3 — Triple Combo | GR, RESD, RHOB | 16 |
| Tier 4 — Minimal | GR, RESD | 11 |

Each tier trains both **Random Forest** and **XGBoost** classifiers with:
- SMOTE oversampling for class imbalance
- Depth-blocked train/test splits (respects spatial autocorrelation)
- GroupKFold cross-validation with randomized hyperparameter search

### Feature Engineering

- Z-scores, rolling mean/std (window=5), first-order differences
- Log ratios: NPHI-RHOB separation, GR/RESD, DTC*RHOB (acoustic impedance), log10(RESD)
- Relative depth normalized to [0, 1]

### Mnemonic Standardization

Automatically maps 144+ vendor-specific log mnemonics to 8 canonical names (GR, RESD, RHOB, NPHI, DTC, PE, CALI, SP).

## Quick Start

### Install

```bash
pip install -e .
```

### Train Models

```bash
python scripts/train_pipeline.py
```

### Run the Web App

```bash
pip install -e ".[app]"
streamlit run app.py
```

### CLI

```bash
# Single well prediction
electrofacies predict --well path/to/well.las --model artifacts/ --output outputs/

# Batch processing
electrofacies batch --inbox data/wells/inbox/ --model artifacts/ --output outputs/

# Model info
electrofacies info --model artifacts/
```

## Project Structure

```
electrofacies/
├── app.py                          # Streamlit web application
├── scripts/train_pipeline.py       # Training orchestration
├── configs/
│   ├── default.yaml                # Master pipeline config
│   ├── facies_schema.yaml          # 6 lithofacies with aliases and colors
│   ├── model_tiers.yaml            # Tier routing hierarchy
│   ├── mnemonic_aliases.yaml       # 144+ vendor mnemonic mappings
│   └── physical_ranges.yaml        # Valid log ranges for QC
├── src/electrofacies/
│   ├── io/                         # LAS/CSV/Excel readers and writers
│   ├── preprocessing/              # Standardization, validation, feature engineering
│   ├── training/                   # RF/XGBoost training, evaluation, artifacts
│   ├── inference/                  # Prediction, batch processing, tier routing
│   ├── qc/                         # Confidence scoring, OOD detection, QC reports
│   └── visualization/              # Log displays, confusion matrices, figures
├── data/training/
│   └── PDB03_training.xlsx         # Training well (Delaware Mountain Group)
└── artifacts/                      # Trained model bundles (generated)
```

## Training Data

Single well (PDB03) from the Delaware Mountain Group, Permian Basin:
- 4,928 samples, 5 log curves (GR, RESD, RHOB, NPHI, DTC)
- 6 lithofacies classes
- Depth-blocked evaluation (bottom 25% held out)

## Requirements

- Python >= 3.9
- scikit-learn, xgboost, imbalanced-learn, lasio, pandas, numpy, matplotlib, scipy

## License

MIT
