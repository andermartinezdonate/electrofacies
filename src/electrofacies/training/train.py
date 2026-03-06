"""Model training routines for the electrofacies prediction pipeline.

Trains Random Forest and (optionally) XGBoost classifiers using SMOTE
oversampling and randomised hyperparameter search with depth-blocked
grouped cross-validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
)

from electrofacies.training.split import (
    create_depth_blocked_split,
    make_depth_groups,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional XGBoost import
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False
    logger.warning(
        "XGBoost is not installed.  XGBoost models will be skipped.  "
        "Install with: pip install xgboost"
    )

# ---------------------------------------------------------------------------
# Conditional imbalanced-learn import
# ---------------------------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    _HAS_IMBLEARN = True
except ImportError:  # pragma: no cover
    _HAS_IMBLEARN = False
    logger.warning(
        "imbalanced-learn is not installed.  SMOTE will be disabled.  "
        "Install with: pip install imbalanced-learn"
    )


# ===================================================================
# Feature engineering (lightweight, training-side)
# ===================================================================

def _engineer_features(
    df: pd.DataFrame,
    base_features: List[str],
    rolling_window: int = 5,
    diff_lag: int = 1,
) -> pd.DataFrame:
    """Create derived features from base log curves.

    Produces z-scores, rolling mean/std, first-differences, and a normalised
    relative-depth column.  NaN rows introduced by rolling operations are
    dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least *base_features* and a ``"DEPTH"``
        column (or equivalent index).
    base_features : list[str]
        Canonical log names to engineer from (e.g. ``["GR", "RESD", ...]``).
    rolling_window : int, optional
        Window size for rolling statistics (default 5).
    diff_lag : int, optional
        Lag for first-difference features (default 1).

    Returns
    -------
    pd.DataFrame
        Wide dataframe with base + engineered features and ``"rel_depth_01"``.
    """
    out = df[base_features].copy()

    for col in base_features:
        series = df[col]
        # Z-score
        mean = series.mean()
        std = series.std()
        out[f"{col}_z"] = (series - mean) / std if std > 0 else 0.0
        # Rolling mean
        out[f"{col}_roll{rolling_window}"] = series.rolling(
            rolling_window, min_periods=1, center=True
        ).mean()
        # Rolling std
        out[f"{col}_rollstd{rolling_window}"] = series.rolling(
            rolling_window, min_periods=1, center=True
        ).std()
        # First difference
        out[f"{col}_diff{diff_lag}"] = series.diff(diff_lag)

    # Relative depth normalised to [0, 1]
    if "DEPTH" in df.columns:
        depth = df["DEPTH"]
    elif df.index.name == "DEPTH":
        depth = df.index.to_series()
    else:
        depth = pd.Series(np.arange(len(df)), index=df.index)
    dmin, dmax = depth.min(), depth.max()
    if dmax > dmin:
        out["rel_depth_01"] = (depth.values - dmin) / (dmax - dmin)
    else:
        out["rel_depth_01"] = 0.5

    # Drop rows that are fully NaN from diff/rolling (edges)
    out = out.dropna()
    return out


# ===================================================================
# Individual model trainers
# ===================================================================

def train_random_forest(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    groups: Optional[np.ndarray],
    config: Dict[str, Any],
    use_smote: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a Random Forest classifier with optional SMOTE and hyperparameter search.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training features.
    y_train : array-like of shape (n_samples,)
        Training labels (integer-encoded).
    groups : np.ndarray or None
        Group labels for GroupKFold.  If ``None``, falls back to
        StratifiedKFold.
    config : dict
        Model config section (``config['models']['random_forest']``).
    use_smote : bool, optional
        Whether to apply SMOTE oversampling (default ``True``).

    Returns
    -------
    tuple[Pipeline, dict]
        ``(best_pipeline, best_params)`` where *best_pipeline* is the refit
        imblearn Pipeline and *best_params* are the hyperparameters chosen by
        RandomizedSearchCV.
    """
    logger.info("Training Random Forest (SMOTE=%s)", use_smote)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Build pipeline steps
    steps: list = []
    if use_smote and _HAS_IMBLEARN:
        smote = SMOTE(
            k_neighbors=config.get("smote_k_neighbors", 3),
            random_state=42,
        )
        steps.append(("smote", smote))
    elif use_smote and not _HAS_IMBLEARN:
        logger.warning("SMOTE requested but imbalanced-learn not installed; skipping.")
    steps.append(("clf", rf))

    if _HAS_IMBLEARN:
        pipeline = ImbPipeline(steps)
    else:
        from sklearn.pipeline import Pipeline as SkPipeline
        pipeline = SkPipeline(steps)

    # Hyperparameter distributions — prefix with "clf__" for pipeline
    param_distributions = {
        "clf__n_estimators": config.get("n_estimators", [200, 300, 500]),
        "clf__max_depth": config.get("max_depth", [None, 10, 15, 20]),
        "clf__min_samples_split": config.get("min_samples_split", [2, 5, 10]),
        "clf__min_samples_leaf": config.get("min_samples_leaf", [1, 2, 3]),
        "clf__max_features": config.get("max_features", ["sqrt", "log2"]),
        "clf__class_weight": config.get(
            "class_weight", ["balanced", "balanced_subsample"]
        ),
    }
    n_iter = config.get("n_iter_search", 40)

    # CV strategy
    if groups is not None:
        n_unique = len(np.unique(groups))
        n_splits = min(5, n_unique)
        cv = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups))
        logger.info("Using GroupKFold with %d splits (%d groups)", n_splits, n_unique)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        logger.info("No groups provided — using StratifiedKFold (5 splits)")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=min(n_iter, _max_combinations(param_distributions)),
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=0,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    logger.info(
        "RF best balanced_accuracy: %.4f | params: %s",
        search.best_score_,
        search.best_params_,
    )
    return search.best_estimator_, search.best_params_


def train_xgboost(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    groups: Optional[np.ndarray],
    config: Dict[str, Any],
    use_smote: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an XGBoost classifier with optional SMOTE and hyperparameter search.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training features.
    y_train : array-like of shape (n_samples,)
        Training labels (integer-encoded).
    groups : np.ndarray or None
        Group labels for GroupKFold.  If ``None``, falls back to
        StratifiedKFold.
    config : dict
        Model config section (``config['models']['xgboost']``).
    use_smote : bool, optional
        Whether to apply SMOTE oversampling (default ``True``).

    Returns
    -------
    tuple[Pipeline, dict]
        ``(best_pipeline, best_params)``.

    Raises
    ------
    ImportError
        If XGBoost is not installed.
    """
    if not _HAS_XGBOOST:
        raise ImportError(
            "XGBoost is not installed.  Install with: pip install xgboost"
        )

    logger.info("Training XGBoost (SMOTE=%s)", use_smote)

    # XGBoost requires integer-encoded labels; encode if strings are provided
    from sklearn.preprocessing import LabelEncoder
    _xgb_le = LabelEncoder()
    y_train_enc = _xgb_le.fit_transform(y_train)

    xgb = XGBClassifier(
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # Build pipeline steps
    steps: list = []
    if use_smote and _HAS_IMBLEARN:
        smote = SMOTE(
            k_neighbors=config.get("smote_k_neighbors", 3),
            random_state=42,
        )
        steps.append(("smote", smote))
    elif use_smote and not _HAS_IMBLEARN:
        logger.warning("SMOTE requested but imbalanced-learn not installed; skipping.")
    steps.append(("clf", xgb))

    if _HAS_IMBLEARN:
        pipeline = ImbPipeline(steps)
    else:
        from sklearn.pipeline import Pipeline as SkPipeline
        pipeline = SkPipeline(steps)

    # Hyperparameter distributions
    param_distributions = {
        "clf__n_estimators": config.get("n_estimators", [200, 300, 500]),
        "clf__max_depth": config.get("max_depth", [4, 6, 8, 10]),
        "clf__learning_rate": config.get("learning_rate", [0.01, 0.05, 0.1]),
        "clf__subsample": config.get("subsample", [0.7, 0.8, 0.9, 1.0]),
        "clf__colsample_bytree": config.get(
            "colsample_bytree", [0.7, 0.8, 0.9, 1.0]
        ),
        "clf__min_child_weight": config.get("min_child_weight", [1, 3, 5]),
        "clf__gamma": config.get("gamma", [0, 0.1, 0.3]),
    }
    n_iter = config.get("n_iter_search", 50)

    # CV strategy (use encoded labels for XGBoost)
    if groups is not None:
        n_unique = len(np.unique(groups))
        n_splits = min(5, n_unique)
        cv = list(GroupKFold(n_splits=n_splits).split(X_train, y_train_enc, groups))
        logger.info("Using GroupKFold with %d splits (%d groups)", n_splits, n_unique)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        logger.info("No groups provided — using StratifiedKFold (5 splits)")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=min(n_iter, _max_combinations(param_distributions)),
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=0,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train_enc)

    logger.info(
        "XGB best balanced_accuracy: %.4f | params: %s",
        search.best_score_,
        search.best_params_,
    )

    # Wrap the pipeline so .predict() returns original string labels
    wrapped = _LabelEncodedPipelineWrapper(search.best_estimator_, _xgb_le)

    return wrapped, search.best_params_


# ===================================================================
# Multi-tier orchestrator
# ===================================================================

def train_all_tiers(
    training_df: pd.DataFrame,
    tiers_config: Dict[str, Any],
    training_config: Dict[str, Any],
    facies_config: Dict[str, Any],
) -> Dict[Tuple[str, str], Tuple[Any, Dict, Dict]]:
    """Train models for every (tier, algorithm) combination.

    Parameters
    ----------
    training_df : pd.DataFrame
        Full training dataframe with canonical column names (``GR``, ``RESD``,
        etc.), a ``DEPTH`` column, and a ``Lithofacies`` target column.
    tiers_config : dict
        Parsed ``model_tiers.yaml`` — must have ``tiers`` and ``algorithms``.
    training_config : dict
        The ``training`` section of ``default.yaml``.
    facies_config : dict
        Parsed ``facies_schema.yaml`` — defines valid facies classes and
        exclusions.

    Returns
    -------
    dict
        Mapping of ``(tier_name, algorithm_name)`` to
        ``(fitted_pipeline, best_params, validation_metrics)``.
    """
    from electrofacies.training.evaluate import evaluate_model

    results: Dict[Tuple[str, str], Tuple[Any, Dict, Dict]] = {}

    # Resolve valid facies
    excluded = set(facies_config.get("excluded_labels", []))
    valid_facies = [
        f for f in facies_config.get("display_order", []) if f not in excluded
    ]
    class_names = valid_facies
    facies_to_code = {
        name: info["code"]
        for name, info in facies_config.get("facies", {}).items()
        if name in valid_facies
    }

    # Resolve algorithms to train
    algorithms = tiers_config.get("algorithms", ["random_forest"])

    # Feature-engineering params
    fe_config = tiers_config.get("feature_engineering", {})
    rolling_window = fe_config.get("rolling_window", training_config.get("rolling_window", 5))
    diff_lag = fe_config.get("diff_lag", training_config.get("diff_lag", 1))

    # Training knobs
    depth_group_size = training_config.get("depth_group_size", 50)
    test_fraction = training_config.get("test_size", 0.25)
    use_smote = training_config.get("use_smote", True)
    smote_k = training_config.get("smote_k_neighbors", 3)

    # Target column
    target_col = training_config.get("target_col", "Lithofacies")

    # Iterate tiers
    tiers = tiers_config.get("tiers", {})
    for tier_name, tier_def in sorted(tiers.items(), key=lambda t: t[1].get("priority", 99)):
        required_logs: List[str] = tier_def["required_logs"]
        logger.info(
            "=== Tier: %s (%s) — logs: %s ===",
            tier_name,
            tier_def.get("name", ""),
            required_logs,
        )

        # Filter to rows with all required logs non-null
        cols_needed = required_logs + ["DEPTH", target_col]
        available = [c for c in cols_needed if c in training_df.columns]
        if len(available) < len(cols_needed):
            missing = set(cols_needed) - set(available)
            logger.warning(
                "Tier %s: missing columns %s in training data — skipping.",
                tier_name,
                missing,
            )
            continue

        tier_df = training_df[cols_needed].dropna(subset=required_logs).copy()

        # Exclude unwanted facies
        tier_df = tier_df[~tier_df[target_col].isin(excluded)]
        # Map target to integer codes
        tier_df = tier_df[tier_df[target_col].isin(facies_to_code)]
        tier_df["target_code"] = tier_df[target_col].map(facies_to_code)

        if tier_df.empty:
            logger.warning("Tier %s: no valid samples after filtering — skipping.", tier_name)
            continue

        logger.info("Tier %s: %d samples after filtering", tier_name, len(tier_df))

        # Engineer features
        X_full = _engineer_features(
            tier_df,
            base_features=required_logs,
            rolling_window=rolling_window,
            diff_lag=diff_lag,
        )

        # Align target and depth after feature engineering dropped rows
        common_idx = X_full.index
        y_full = tier_df.loc[common_idx, "target_code"].values
        depth_full = tier_df.loc[common_idx, "DEPTH"].values

        # Depth-blocked train/test split
        train_idx, test_idx = create_depth_blocked_split(
            X_full.values, y_full, depth_full, test_fraction=test_fraction
        )
        X_train = X_full.iloc[train_idx]
        y_train = y_full[train_idx]
        X_test = X_full.iloc[test_idx]
        y_test = y_full[test_idx]
        depth_train = depth_full[train_idx]

        # Groups for CV within training set
        groups = make_depth_groups(
            pd.Series(depth_train), group_size=depth_group_size
        )

        # Train each algorithm
        for algo in algorithms:
            logger.info("--- Training %s for %s ---", algo, tier_name)

            if algo == "random_forest":
                model_config = training_config if "n_estimators" in training_config else {}
                # Prefer top-level models config passed via training_config
                model_config = {**model_config, "smote_k_neighbors": smote_k}
                try:
                    pipeline, params = train_random_forest(
                        X_train, y_train, groups, model_config, use_smote=use_smote
                    )
                except Exception:
                    logger.exception("Failed to train RF for %s", tier_name)
                    continue

            elif algo == "xgboost":
                if not _HAS_XGBOOST:
                    logger.warning("XGBoost not installed — skipping %s/%s", tier_name, algo)
                    continue
                model_config = {"smote_k_neighbors": smote_k}
                try:
                    pipeline, params = train_xgboost(
                        X_train, y_train, groups, model_config, use_smote=use_smote
                    )
                except Exception:
                    logger.exception("Failed to train XGBoost for %s", tier_name)
                    continue
            else:
                logger.warning("Unknown algorithm '%s' — skipping.", algo)
                continue

            # Evaluate on held-out test set
            metrics = evaluate_model(pipeline, X_test, y_test, class_names)
            logger.info(
                "%s/%s — balanced_accuracy=%.4f, kappa=%.4f",
                tier_name,
                algo,
                metrics["balanced_accuracy"],
                metrics["cohen_kappa"],
            )

            results[(tier_name, algo)] = (pipeline, params, metrics)

    logger.info("Training complete. %d (tier, algorithm) bundles produced.", len(results))
    return results


# ===================================================================
# XGBoost label-encoding wrapper
# ===================================================================

class _LabelEncodedPipelineWrapper:
    """Wraps an XGBoost pipeline + LabelEncoder so predict() returns original string labels.

    XGBoost requires integer-encoded targets, but the rest of the pipeline
    (evaluate, inference) expects string labels matching the RF models.
    This wrapper transparently inverse-transforms predictions.
    """

    def __init__(self, pipeline: Any, label_encoder: Any) -> None:
        self._pipeline = pipeline
        self._label_encoder = label_encoder

    def predict(self, X: Any) -> np.ndarray:
        encoded_preds = self._pipeline.predict(X)
        return self._label_encoder.inverse_transform(encoded_preds)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._pipeline.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self._label_encoder.classes_

    def __getattr__(self, name: str) -> Any:
        # Guard against infinite recursion during unpickling —
        # pickle calls __getattr__ before __dict__ is populated.
        if name in ("_pipeline", "_label_encoder"):
            raise AttributeError(name)
        return getattr(self._pipeline, name)


# ===================================================================
# Helpers
# ===================================================================

def _max_combinations(param_dist: Dict[str, list]) -> int:
    """Return the total number of parameter combinations.

    Used to cap ``n_iter`` in ``RandomizedSearchCV`` so it does not exceed
    the exhaustive search space.
    """
    total = 1
    for values in param_dist.values():
        total *= len(values) if isinstance(values, list) else 1
    return max(total, 1)
