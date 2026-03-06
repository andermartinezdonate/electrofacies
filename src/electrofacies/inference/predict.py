"""Core prediction logic for single-well electrofacies inference.

Provides the predict-single-well pipeline: feature transformation, model
invocation, probability extraction, and algorithm selection.  All functions
operate on a single well at a time and return DataFrames suitable for
downstream post-processing and output writing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from electrofacies.io.schemas import DEPTH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-model prediction
# ---------------------------------------------------------------------------

def predict_single_well(
    well_data: pd.DataFrame,
    model_bundle: Dict[str, Any],
    transformer: Any,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Run facies prediction on a single well using one model bundle.

    Parameters
    ----------
    well_data : pd.DataFrame
        Preprocessed well-log DataFrame.  Must contain a ``DEPTH`` column
        (or depth as the index) and the canonical log columns required by
        the model tier.
    model_bundle : dict
        A loaded model bundle as returned by
        :func:`electrofacies.training.artifacts.load_model_bundle`.
        Expected keys:

        - ``"model"`` : fitted sklearn-compatible estimator with
          ``predict_proba``.
        - ``"feature_columns"`` : list of feature names the model was
          trained on.
        - ``"class_names"`` : list of target class labels in the order
          used by ``predict_proba``.
        - ``"tier"`` : tier name string.
        - ``"algorithm"`` : algorithm name string.
        - ``"training_means"`` (optional) : dict of per-feature means for
          NaN imputation.

    transformer : object
        A fitted ``FaciesTransformer`` (or compatible) instance with a
        ``.transform(df)`` method that engineers features from the raw
        canonical columns.
    config : dict
        Master configuration dict (parsed ``default.yaml``).

    Returns
    -------
    pd.DataFrame
        Prediction results with columns:
        ``DEPTH``, ``PREDICTED_FACIES``, ``PROB_<class>`` for each class,
        ``CONFIDENCE_SCORE``, ``MODEL_TIER``, ``ALGORITHM``.
    """
    metadata = model_bundle.get("metadata", {})
    tier_name = metadata.get("tier", model_bundle.get("tier", "unknown"))
    algorithm = metadata.get("algorithm", model_bundle.get("algorithm", "unknown"))
    model = model_bundle["model"]
    feature_columns: List[str] = metadata.get(
        "feature_names", model_bundle.get("feature_columns", [])
    )
    class_names: List[str] = metadata.get(
        "class_names", model_bundle.get("class_names", [])
    )
    training_means: Dict[str, float] = model_bundle.get("training_means", {})

    logger.info(
        "Predicting well: tier=%s, algorithm=%s, %d samples, "
        "%d expected features.",
        tier_name,
        algorithm,
        len(well_data),
        len(feature_columns),
    )

    # ---- Preserve depth information ----------------------------------------
    if DEPTH in well_data.columns:
        depth_values = well_data[DEPTH].values.copy()
        work_df = well_data.drop(columns=[DEPTH]).copy()
    elif well_data.index.name == DEPTH:
        depth_values = well_data.index.values.copy()
        work_df = well_data.reset_index(drop=True).copy()
    else:
        # Fallback: treat the index as depth
        depth_values = well_data.index.values.copy()
        work_df = well_data.reset_index(drop=True).copy()
        logger.warning(
            "No explicit DEPTH column or index found; using DataFrame "
            "index as depth."
        )

    # ---- Feature engineering via transformer --------------------------------
    try:
        transform_result = transformer.transform(work_df)
        # transformer.transform() returns (DataFrame, list[str]) tuple
        if isinstance(transform_result, tuple):
            features_df, _feat_names = transform_result
        else:
            features_df = transform_result
    except Exception:
        logger.exception(
            "Transformer.transform() failed for tier=%s, algorithm=%s.",
            tier_name,
            algorithm,
        )
        raise

    # ---- Align columns to model's expected feature order --------------------
    missing_features = [c for c in feature_columns if c not in features_df.columns]
    if missing_features:
        logger.warning(
            "%d feature column(s) missing after transform; "
            "filling with training means or zeros: %s",
            len(missing_features),
            missing_features,
        )
        for col in missing_features:
            features_df[col] = training_means.get(col, 0.0)

    extra_features = [c for c in features_df.columns if c not in feature_columns]
    if extra_features:
        logger.debug(
            "Dropping %d extra feature column(s) not expected by model: %s",
            len(extra_features),
            extra_features[:10],
        )

    features_df = features_df[feature_columns]

    # ---- Impute residual NaNs with column means ----------------------------
    nan_counts = features_df.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if not cols_with_nans.empty:
        logger.info(
            "Filling residual NaNs in %d feature column(s) with "
            "training means / column means.",
            len(cols_with_nans),
        )
        for col in cols_with_nans.index:
            fill_value = training_means.get(col, features_df[col].mean())
            if pd.isna(fill_value):
                fill_value = 0.0
            features_df[col] = features_df[col].fillna(fill_value)

    # ---- Run prediction -----------------------------------------------------
    X = features_df.values

    try:
        probas = model.predict_proba(X)
    except Exception:
        logger.exception(
            "model.predict_proba() failed for tier=%s, algorithm=%s.",
            tier_name,
            algorithm,
        )
        raise

    predicted_indices = np.argmax(probas, axis=1)
    predicted_facies = [class_names[i] for i in predicted_indices]

    # ---- Confidence scores (max probability per sample) ---------------------
    confidence_scores = np.max(probas, axis=1)

    # ---- Assemble output DataFrame -----------------------------------------
    result = pd.DataFrame({DEPTH: depth_values})
    result["PREDICTED_FACIES"] = predicted_facies

    for idx, cls_name in enumerate(class_names):
        result[f"PROB_{cls_name}"] = probas[:, idx]

    result["CONFIDENCE_SCORE"] = confidence_scores
    result["MODEL_TIER"] = tier_name
    result["ALGORITHM"] = algorithm

    logger.info(
        "Prediction complete: %d samples, %d classes, "
        "mean confidence %.3f, tier=%s, algorithm=%s.",
        len(result),
        len(class_names),
        float(confidence_scores.mean()),
        tier_name,
        algorithm,
    )

    return result


# ---------------------------------------------------------------------------
# Multi-algorithm prediction
# ---------------------------------------------------------------------------

def predict_all_algorithms(
    well_data: pd.DataFrame,
    tier_models: Dict[str, Dict[str, Any]],
    transformer: Any,
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Run prediction with every algorithm available for the tier.

    Parameters
    ----------
    well_data : pd.DataFrame
        Preprocessed well-log data (canonical columns + DEPTH).
    tier_models : dict
        ``{algorithm_name: model_bundle}`` as returned by
        :func:`~electrofacies.inference.tier_router.load_tier_models`.
    transformer : object
        Fitted ``FaciesTransformer`` instance.
    config : dict
        Master configuration.

    Returns
    -------
    dict
        ``{algorithm_name: predictions_df}`` for each algorithm that
        succeeded.  Algorithms that fail are logged but omitted from the
        output.
    """
    all_predictions: Dict[str, pd.DataFrame] = {}

    for algo_name, bundle in tier_models.items():
        logger.info("Running predictions with algorithm '%s'.", algo_name)
        try:
            preds = predict_single_well(well_data, bundle, transformer, config)
            all_predictions[algo_name] = preds
        except Exception:
            logger.exception(
                "Prediction failed for algorithm '%s'; skipping.",
                algo_name,
            )

    if not all_predictions:
        logger.error(
            "All %d algorithm(s) failed during prediction.",
            len(tier_models),
        )

    return all_predictions


# ---------------------------------------------------------------------------
# Algorithm selection
# ---------------------------------------------------------------------------

def select_best_prediction(
    all_predictions: Dict[str, pd.DataFrame],
    strategy: str = "highest_mean_confidence",
) -> Tuple[str, pd.DataFrame]:
    """Select the best algorithm's predictions according to the given strategy.

    Parameters
    ----------
    all_predictions : dict
        ``{algorithm_name: predictions_df}`` as returned by
        :func:`predict_all_algorithms`.
    strategy : str, optional
        Selection strategy.  Currently supported:

        - ``"highest_mean_confidence"`` (default): pick the algorithm whose
          predictions have the highest average ``CONFIDENCE_SCORE``.
        - ``"highest_min_confidence"``: pick the algorithm whose worst-case
          (minimum) ``CONFIDENCE_SCORE`` is highest.

    Returns
    -------
    tuple of (str, pd.DataFrame)
        ``(algorithm_name, best_predictions_df)``.

    Raises
    ------
    ValueError
        If *all_predictions* is empty.
    """
    if not all_predictions:
        raise ValueError(
            "No predictions available to select from.  All algorithms "
            "may have failed."
        )

    if len(all_predictions) == 1:
        algo_name = next(iter(all_predictions))
        logger.info(
            "Only one algorithm available ('%s'); selecting by default.",
            algo_name,
        )
        return algo_name, all_predictions[algo_name]

    best_algo: Optional[str] = None
    best_score: float = -1.0

    for algo_name, preds_df in all_predictions.items():
        if "CONFIDENCE_SCORE" not in preds_df.columns:
            logger.warning(
                "Algorithm '%s' predictions missing CONFIDENCE_SCORE; "
                "cannot score.",
                algo_name,
            )
            continue

        if strategy == "highest_mean_confidence":
            score = float(preds_df["CONFIDENCE_SCORE"].mean())
        elif strategy == "highest_min_confidence":
            score = float(preds_df["CONFIDENCE_SCORE"].min())
        else:
            logger.warning(
                "Unknown selection strategy '%s'; falling back to "
                "'highest_mean_confidence'.",
                strategy,
            )
            score = float(preds_df["CONFIDENCE_SCORE"].mean())

        logger.info(
            "Algorithm '%s' selection score (strategy=%s): %.4f",
            algo_name,
            strategy,
            score,
        )

        if score > best_score:
            best_score = score
            best_algo = algo_name

    if best_algo is None:
        # Fallback: just pick the first algorithm.
        best_algo = next(iter(all_predictions))
        logger.warning(
            "Could not score any algorithm; falling back to '%s'.",
            best_algo,
        )

    logger.info(
        "Selected best algorithm: '%s' with score %.4f (strategy=%s).",
        best_algo,
        best_score,
        strategy,
    )
    return best_algo, all_predictions[best_algo]
