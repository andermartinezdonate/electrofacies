"""Post-processing for electrofacies predictions.

Applies smoothing filters, confidence flags, out-of-distribution flags, and
QC status labelling to raw model output.  Also provides per-well summary
statistics used in batch reporting.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def modal_filter(facies_series: pd.Series, window: int = 3) -> pd.Series:
    """Apply a modal (majority-vote) smoothing filter to a facies series.

    Within each sliding window, the most frequent facies label replaces the
    centre sample.  This removes isolated single-sample spikes while
    preserving genuine facies boundaries.

    Parameters
    ----------
    facies_series : pd.Series
        Series of predicted facies labels (strings or integers).
    window : int, optional
        Window size for the rolling vote.  Must be an odd positive integer.
        Defaults to ``3``.

    Returns
    -------
    pd.Series
        Smoothed facies series with the same index as the input.
    """
    if window < 1:
        logger.warning(
            "modal_filter called with window=%d; returning input unchanged.",
            window,
        )
        return facies_series.copy()

    if window % 2 == 0:
        window += 1
        logger.warning(
            "modal_filter window must be odd; adjusted to %d.", window
        )

    if len(facies_series) <= window:
        logger.debug(
            "Series length (%d) <= window (%d); returning input unchanged.",
            len(facies_series),
            window,
        )
        return facies_series.copy()

    values = facies_series.values
    half = window // 2
    smoothed = values.copy()

    for i in range(half, len(values) - half):
        local_window = values[i - half : i + half + 1]
        counts = Counter(local_window)
        # Pick the most common label; on ties keep the original centre value.
        most_common_label, most_common_count = counts.most_common(1)[0]
        # Only replace if the mode is strictly dominant (more than 1 vote if
        # the centre disagrees), otherwise keep the centre value to be
        # conservative on ties.
        centre_val = values[i]
        centre_count = counts.get(centre_val, 0)
        if most_common_count > centre_count:
            smoothed[i] = most_common_label

    n_changed = int(np.sum(smoothed != values))
    logger.info(
        "Modal filter (window=%d): %d / %d samples changed (%.1f%%).",
        window,
        n_changed,
        len(values),
        100.0 * n_changed / max(len(values), 1),
    )

    return pd.Series(smoothed, index=facies_series.index, name=facies_series.name)


# ---------------------------------------------------------------------------
# Flag assignment
# ---------------------------------------------------------------------------

def assign_confidence_flags(
    predictions_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Add a ``LOW_CONFIDENCE`` boolean column to *predictions_df*.

    Samples whose ``CONFIDENCE_SCORE`` falls below *threshold* are flagged
    ``True``.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain a ``CONFIDENCE_SCORE`` column.
    threshold : float, optional
        Confidence threshold below which a sample is flagged.
        Defaults to ``0.5``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``LOW_CONFIDENCE`` column.
    """
    df = predictions_df.copy()

    if "CONFIDENCE_SCORE" not in df.columns:
        logger.warning(
            "CONFIDENCE_SCORE column not found; setting LOW_CONFIDENCE=False "
            "for all rows."
        )
        df["LOW_CONFIDENCE"] = False
        return df

    df["LOW_CONFIDENCE"] = df["CONFIDENCE_SCORE"] < threshold

    n_flagged = int(df["LOW_CONFIDENCE"].sum())
    logger.info(
        "Confidence flags: %d / %d samples flagged LOW_CONFIDENCE "
        "(threshold=%.2f).",
        n_flagged,
        len(df),
        threshold,
    )
    return df


def assign_ood_flags(
    predictions_df: pd.DataFrame,
    well_features: pd.DataFrame,
    ood_detector: Any,
) -> pd.DataFrame:
    """Add an ``OOD_FLAG`` boolean column to *predictions_df*.

    Uses the provided ``OODDetector`` instance to score the well's feature
    vectors against the training-data distribution.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions DataFrame (one row per depth sample).
    well_features : pd.DataFrame
        Feature matrix for the well (same rows / ordering as
        *predictions_df*).  Passed to ``ood_detector.score()`` or
        ``ood_detector.predict()``.
    ood_detector : object
        A fitted ``OODDetector`` instance from
        :mod:`electrofacies.qc.ood`.  Must expose either a
        ``predict(X) -> array[bool]`` method or a
        ``score(X) -> array[float]`` method.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``OOD_FLAG`` column.
    """
    df = predictions_df.copy()

    if ood_detector is None:
        logger.info(
            "No OOD detector provided; setting OOD_FLAG=False for all rows."
        )
        df["OOD_FLAG"] = False
        return df

    try:
        if hasattr(ood_detector, "predict"):
            ood_labels = ood_detector.predict(well_features)
            # Convention: predict returns True for OOD (anomalous) samples,
            # or -1 for outliers (sklearn convention).  Handle both.
            if hasattr(ood_labels, "dtype") and np.issubdtype(
                ood_labels.dtype, np.integer
            ):
                # sklearn convention: -1 = outlier, 1 = inlier
                ood_flags = ood_labels == -1
            else:
                ood_flags = np.asarray(ood_labels, dtype=bool)
        elif hasattr(ood_detector, "score"):
            scores = ood_detector.score(well_features)
            # Scores above 95th percentile of training distribution are OOD.
            ood_threshold = getattr(ood_detector, "threshold_", None)
            if ood_threshold is not None:
                ood_flags = scores > ood_threshold
            else:
                logger.warning(
                    "OOD detector has no threshold_; using 95th percentile "
                    "of well scores as fallback."
                )
                fallback_threshold = np.percentile(scores, 95)
                ood_flags = scores > fallback_threshold
        else:
            logger.warning(
                "OOD detector has neither predict() nor score(); "
                "setting OOD_FLAG=False."
            )
            df["OOD_FLAG"] = False
            return df

        df["OOD_FLAG"] = np.asarray(ood_flags, dtype=bool)

        n_ood = int(df["OOD_FLAG"].sum())
        logger.info(
            "OOD flags: %d / %d samples flagged as out-of-distribution "
            "(%.1f%%).",
            n_ood,
            len(df),
            100.0 * n_ood / max(len(df), 1),
        )

    except Exception:
        logger.exception(
            "OOD detection failed; setting OOD_FLAG=False for all rows."
        )
        df["OOD_FLAG"] = False

    return df


def assign_qc_status(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Combine flag columns into a single ``QC_STATUS`` label per sample.

    The status hierarchy is:

    - ``"LOW_CONF_AND_OOD"`` -- both ``LOW_CONFIDENCE`` and ``OOD_FLAG``
      are ``True``.
    - ``"LOW_CONFIDENCE"`` -- only the confidence flag is ``True``.
    - ``"OOD"`` -- only the OOD flag is ``True``.
    - ``"GOOD"`` -- no flags raised.

    If either flag column is missing, it is treated as all-``False``.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain ``LOW_CONFIDENCE`` and/or ``OOD_FLAG`` boolean
        columns (typically added by :func:`assign_confidence_flags` and
        :func:`assign_ood_flags`).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``QC_STATUS`` column.
    """
    df = predictions_df.copy()

    low_conf = df["LOW_CONFIDENCE"] if "LOW_CONFIDENCE" in df.columns else pd.Series(
        False, index=df.index
    )
    ood = df["OOD_FLAG"] if "OOD_FLAG" in df.columns else pd.Series(
        False, index=df.index
    )

    conditions = [
        low_conf & ood,
        low_conf & ~ood,
        ~low_conf & ood,
    ]
    choices = ["LOW_CONF_AND_OOD", "LOW_CONFIDENCE", "OOD"]
    df["QC_STATUS"] = np.select(conditions, choices, default="GOOD")

    status_counts = df["QC_STATUS"].value_counts().to_dict()
    logger.info("QC status distribution: %s", status_counts)

    return df


# ---------------------------------------------------------------------------
# Per-well summary
# ---------------------------------------------------------------------------

def compute_well_summary(
    predictions_df: pd.DataFrame,
    well_name: str,
    tier_used: str,
    algorithm: str,
) -> Dict[str, Any]:
    """Compute per-well summary statistics from the post-processed predictions.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Post-processed predictions with columns ``PREDICTED_FACIES``,
        ``CONFIDENCE_SCORE``, ``QC_STATUS`` (and optionally ``LOW_CONFIDENCE``,
        ``OOD_FLAG``).
    well_name : str
        Human-readable well name for the summary record.
    tier_used : str
        Model tier identifier (e.g. ``"tier_2"``).
    algorithm : str
        Algorithm name (e.g. ``"random_forest"``).

    Returns
    -------
    dict
        Summary dictionary with keys:

        - ``well_name``
        - ``tier_used``
        - ``algorithm``
        - ``n_samples`` : int
        - ``facies_distribution`` : dict of ``{facies: count}``
        - ``facies_fractions`` : dict of ``{facies: fraction}``
        - ``mean_confidence`` : float
        - ``median_confidence`` : float
        - ``min_confidence`` : float
        - ``pct_low_confidence`` : float (0-100)
        - ``pct_ood`` : float (0-100)
        - ``pct_good`` : float (0-100)
        - ``overall_qc_grade`` : str -- ``"PASS"``, ``"REVIEW"``, or
          ``"FAIL"``
    """
    n_samples = len(predictions_df)

    # Facies distribution
    facies_col = "PREDICTED_FACIES"
    if facies_col in predictions_df.columns:
        facies_counts = predictions_df[facies_col].value_counts().to_dict()
        facies_fractions = {
            k: round(v / max(n_samples, 1), 4) for k, v in facies_counts.items()
        }
    else:
        facies_counts = {}
        facies_fractions = {}

    # Confidence statistics
    if "CONFIDENCE_SCORE" in predictions_df.columns:
        mean_conf = float(predictions_df["CONFIDENCE_SCORE"].mean())
        median_conf = float(predictions_df["CONFIDENCE_SCORE"].median())
        min_conf = float(predictions_df["CONFIDENCE_SCORE"].min())
    else:
        mean_conf = 0.0
        median_conf = 0.0
        min_conf = 0.0

    # Flag percentages
    if "LOW_CONFIDENCE" in predictions_df.columns:
        pct_low_conf = 100.0 * predictions_df["LOW_CONFIDENCE"].sum() / max(n_samples, 1)
    else:
        pct_low_conf = 0.0

    if "OOD_FLAG" in predictions_df.columns:
        pct_ood = 100.0 * predictions_df["OOD_FLAG"].sum() / max(n_samples, 1)
    else:
        pct_ood = 0.0

    # QC status fractions
    if "QC_STATUS" in predictions_df.columns:
        pct_good = (
            100.0
            * (predictions_df["QC_STATUS"] == "GOOD").sum()
            / max(n_samples, 1)
        )
    else:
        pct_good = 100.0

    # Overall QC grade
    # PASS: >= 80% GOOD, REVIEW: >= 50% GOOD, FAIL: < 50% GOOD
    if pct_good >= 80.0:
        overall_qc_grade = "PASS"
    elif pct_good >= 50.0:
        overall_qc_grade = "REVIEW"
    else:
        overall_qc_grade = "FAIL"

    summary = {
        "well_name": well_name,
        "tier_used": tier_used,
        "algorithm": algorithm,
        "n_samples": n_samples,
        "facies_distribution": facies_counts,
        "facies_fractions": facies_fractions,
        "mean_confidence": round(mean_conf, 4),
        "median_confidence": round(median_conf, 4),
        "min_confidence": round(min_conf, 4),
        "pct_low_confidence": round(pct_low_conf, 2),
        "pct_ood": round(pct_ood, 2),
        "pct_good": round(pct_good, 2),
        "overall_qc_grade": overall_qc_grade,
    }

    logger.info(
        "Well summary for '%s': n_samples=%d, mean_conf=%.3f, "
        "pct_good=%.1f%%, grade=%s.",
        well_name,
        n_samples,
        mean_conf,
        pct_good,
        overall_qc_grade,
    )

    return summary
