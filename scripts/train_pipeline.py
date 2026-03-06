"""End-to-end training pipeline script.

Loads PDB03 training data, trains RF and XGBoost models for all viable tiers,
saves model bundles to artifacts/, and runs a quick self-test prediction.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pipeline")

# ── Configuration ──────────────────────────────────────────────────────
CONFIGS_DIR = PROJECT_ROOT / "configs"
with open(CONFIGS_DIR / "default.yaml") as f:
    CONFIG = yaml.safe_load(f)
with open(CONFIGS_DIR / "model_tiers.yaml") as f:
    TIERS_CONFIG = yaml.safe_load(f)
with open(CONFIGS_DIR / "facies_schema.yaml") as f:
    FACIES_CONFIG = yaml.safe_load(f)

# ── Step 1: Read training data ─────────────────────────────────────────
logger.info("=" * 60)
logger.info("STEP 1: Reading training data")
logger.info("=" * 60)

from electrofacies.io.readers import read_training_data

training_path = PROJECT_ROOT / CONFIG["paths"]["training_data"]
raw_df = read_training_data(training_path)
logger.info("Raw data shape: %s", raw_df.shape)

# Filter to DMG formation
train_cfg = CONFIG["training"]
formation_col = train_cfg["formation_col"]
formation_filter = train_cfg["formation_filter"]
if formation_col in raw_df.columns:
    raw_df = raw_df[raw_df[formation_col] == formation_filter].copy()
    logger.info("After formation filter ('%s'): %d rows", formation_filter, len(raw_df))

# ── Step 2: Canonicalize facies labels ──────────────────────────────────
logger.info("=" * 60)
logger.info("STEP 2: Canonicalizing facies labels")
logger.info("=" * 60)

from electrofacies.preprocessing.standardize import canonicalize_facies_labels

target_col = train_cfg["target_col"]
raw_df[target_col] = canonicalize_facies_labels(raw_df[target_col], FACIES_CONFIG)

# Remove excluded facies
excluded = FACIES_CONFIG.get("excluded_labels", [])
before = len(raw_df)
raw_df = raw_df[~raw_df[target_col].isin(excluded)].copy()
logger.info("Removed excluded facies %s: %d -> %d rows", excluded, before, len(raw_df))

logger.info("Final facies distribution:")
for facies, count in raw_df[target_col].value_counts().items():
    logger.info("  %s: %d (%.1f%%)", facies, count, 100 * count / len(raw_df))

# ── Step 3: Map raw columns to canonical names ─────────────────────────
logger.info("=" * 60)
logger.info("STEP 3: Mapping columns to canonical names")
logger.info("=" * 60)

raw_to_canonical = train_cfg["raw_to_canonical"]
rename_map = {raw: canon for raw, canon in raw_to_canonical.items() if raw in raw_df.columns}
df = raw_df.rename(columns=rename_map).copy()

depth_col = train_cfg["depth_col"]
if depth_col in df.columns:
    df = df.rename(columns={depth_col: "DEPTH"})
    depth_col = "DEPTH"

canonical_logs = list(raw_to_canonical.values())
available = [c for c in canonical_logs if c in df.columns]
logger.info("Canonical columns available: %s", available)

# ── Step 4: Train models per tier ──────────────────────────────────────
logger.info("=" * 60)
logger.info("STEP 4: Training models per tier")
logger.info("=" * 60)

from electrofacies.preprocessing.transform import FaciesTransformer
from electrofacies.training.split import create_depth_blocked_split, make_depth_groups
from electrofacies.training.train import train_random_forest, train_xgboost
from electrofacies.training.evaluate import evaluate_model
from electrofacies.training.artifacts import save_model_bundle
from electrofacies.qc.ood import OODDetector

artifacts_dir = PROJECT_ROOT / CONFIG["paths"].get("artifacts_dir", CONFIG["paths"]["artifacts"])
artifacts_dir.mkdir(parents=True, exist_ok=True)

fe_config = CONFIG.get("feature_engineering", {})
tiers = TIERS_CONFIG["tiers"]

for tier_name, tier_info in sorted(tiers.items(), key=lambda x: x[1]["priority"]):
    required_logs = tier_info["required_logs"]
    tier_available = [c for c in required_logs if c in df.columns]

    if set(required_logs) != set(tier_available):
        logger.warning(
            "Tier '%s' requires %s but only %s available — SKIPPING.",
            tier_name, required_logs, tier_available,
        )
        continue

    logger.info("─" * 50)
    logger.info("Training tier '%s' (%s) — logs: %s",
                tier_name, tier_info["name"], required_logs)
    logger.info("─" * 50)

    # Subset to required logs + depth + target
    cols_needed = required_logs + [target_col]
    if depth_col in df.columns:
        cols_needed = [depth_col] + cols_needed
    tier_df = df[cols_needed].dropna(subset=required_logs + [target_col]).copy()
    logger.info("Tier '%s' data: %d rows after dropna", tier_name, len(tier_df))

    # Feature engineering
    transformer = FaciesTransformer(config=fe_config)
    features_df, feature_names = transformer.fit_transform(tier_df, required_logs)
    logger.info("Engineered %d features: %s...", len(feature_names), feature_names[:5])

    X = features_df[feature_names].values
    y = tier_df[target_col].values

    # Depth-blocked split
    if depth_col in tier_df.columns:
        depths = tier_df[depth_col].values
        train_idx, test_idx = create_depth_blocked_split(
            X, y, depths,
            test_fraction=train_cfg.get("test_size", 0.25),
        )
    else:
        from electrofacies.training.split import create_stratified_split
        train_idx, test_idx = create_stratified_split(
            X, y, test_size=train_cfg.get("test_size", 0.25),
        )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    logger.info("Split: %d train, %d test", len(X_train), len(X_test))

    # Create depth groups for GroupKFold in training set
    if depth_col in tier_df.columns:
        depth_train = depths[train_idx]
        groups = make_depth_groups(
            pd.Series(depth_train),
            group_size=train_cfg.get("depth_group_size", 50),
        )
    else:
        groups = None

    # Class names
    class_names = sorted(set(y))

    # OOD detector (fit on training features)
    logger.info("Fitting OOD detector for tier '%s'...", tier_name)
    try:
        ood = OODDetector(method="mahalanobis", percentile_threshold=95)
        ood.fit(X_train)
        logger.info("OOD detector fitted.")
    except Exception:
        logger.exception("OOD detector fitting failed for tier '%s'", tier_name)
        ood = None

    # ── Train Random Forest ──
    logger.info("Training Random Forest for tier '%s'...", tier_name)
    rf_config = CONFIG["models"]["random_forest"].copy()
    rf_config["smote_k_neighbors"] = train_cfg.get("smote_k_neighbors", 3)
    try:
        rf_model, rf_params = train_random_forest(
            X_train, y_train, groups, rf_config,
            use_smote=train_cfg.get("use_smote", True),
        )

        rf_metrics = evaluate_model(rf_model, X_test, y_test, class_names=class_names)
        logger.info("RF accuracy: %.3f, balanced_accuracy: %.3f, kappa: %.3f",
                     rf_metrics["accuracy"],
                     rf_metrics["balanced_accuracy"],
                     rf_metrics["cohen_kappa"])

        rf_bundle_dir = save_model_bundle(
            model=rf_model,
            transformer=transformer,
            ood_detector=ood,
            config=CONFIG,
            metrics=rf_metrics,
            output_dir=str(artifacts_dir),
            tier_name=tier_name,
            algorithm="random_forest",
            feature_names=feature_names,
            class_names=class_names,
        )
        logger.info("RF bundle saved: %s", rf_bundle_dir)

    except Exception:
        logger.exception("Random Forest training failed for tier '%s'", tier_name)

    # ── Train XGBoost ──
    logger.info("Training XGBoost for tier '%s'...", tier_name)
    xgb_config = CONFIG["models"]["xgboost"].copy()
    xgb_config["smote_k_neighbors"] = train_cfg.get("smote_k_neighbors", 3)
    try:
        xgb_model, xgb_params = train_xgboost(
            X_train, y_train, groups, xgb_config,
            use_smote=train_cfg.get("use_smote", True),
        )

        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, class_names=class_names)
        logger.info("XGB accuracy: %.3f, balanced_accuracy: %.3f, kappa: %.3f",
                     xgb_metrics["accuracy"],
                     xgb_metrics["balanced_accuracy"],
                     xgb_metrics["cohen_kappa"])

        xgb_bundle_dir = save_model_bundle(
            model=xgb_model,
            transformer=transformer,
            ood_detector=ood,
            config=CONFIG,
            metrics=xgb_metrics,
            output_dir=str(artifacts_dir),
            tier_name=tier_name,
            algorithm="xgboost",
            feature_names=feature_names,
            class_names=class_names,
        )
        logger.info("XGB bundle saved: %s", xgb_bundle_dir)

    except Exception:
        logger.exception("XGBoost training failed for tier '%s'", tier_name)

logger.info("=" * 60)
logger.info("TRAINING COMPLETE")
logger.info("=" * 60)

# ── Step 5: Quick self-test prediction ─────────────────────────────────
logger.info("=" * 60)
logger.info("STEP 5: Self-test prediction on training data")
logger.info("=" * 60)

from electrofacies.inference.tier_router import load_tier_models
from electrofacies.inference.predict import predict_single_well

try:
    test_models = load_tier_models(str(artifacts_dir), "tier_1")
    if test_models:
        algo_name, bundle = next(iter(test_models.items()))
        # Predict on 50 samples from training data
        test_slice = df.head(50).copy()

        # Load transformer from bundle
        test_transformer = bundle.get("transformer")
        if test_transformer is None:
            # Fallback: create a new one
            test_transformer = FaciesTransformer(config=fe_config)
            test_transformer.fit(test_slice, ["GR", "RESD", "RHOB", "NPHI", "DTC"])

        preds = predict_single_well(test_slice, bundle, test_transformer, CONFIG)
        logger.info("Self-test prediction shape: %s", preds.shape)
        logger.info("Predicted facies:\n%s", preds["PREDICTED_FACIES"].value_counts().to_string())
        logger.info("Mean confidence: %.3f", preds["CONFIDENCE_SCORE"].mean())
        logger.info("SELF-TEST PASSED!")
    else:
        logger.warning("No models loaded for self-test.")
except Exception:
    logger.exception("Self-test failed (non-critical)")

logger.info("=" * 60)
logger.info("ALL DONE. Artifacts saved to: %s", artifacts_dir)
logger.info("=" * 60)
