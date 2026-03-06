"""Prediction confidence scoring and probability calibration.

Provides functions for quantifying prediction uncertainty from the class
probability vectors produced by tree-based classifiers (Random Forest,
XGBoost).  Three complementary metrics are computed per sample:

- **Max probability** -- highest class probability; direct confidence proxy.
- **Shannon entropy** -- information-theoretic uncertainty over the full
  probability distribution.  Higher values indicate greater ambiguity.
- **Margin** -- difference between the top-two class probabilities; larger
  margins mean clearer separation from the runner-up class.

Also includes a convenience wrapper around
:class:`~sklearn.calibration.CalibratedClassifierCV` for post-hoc probability
calibration on a held-out validation set.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual confidence metrics
# ---------------------------------------------------------------------------

def compute_max_probability(proba_matrix: np.ndarray) -> np.ndarray:
    """Return the maximum class probability for each sample.

    Parameters
    ----------
    proba_matrix : np.ndarray of shape (n_samples, n_classes)
        Per-class probability vectors (rows sum to 1).

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Maximum probability value per sample.
    """
    proba = np.asarray(proba_matrix, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(
            f"proba_matrix must be 2-D, got shape {proba.shape}."
        )
    return proba.max(axis=1)


def compute_entropy(proba_matrix: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy of each sample's probability vector.

    .. math::

        H = -\\sum_k p_k \\, \\log_2 p_k

    The convention :math:`0 \\cdot \\log_2 0 = 0` is applied so that zero
    probabilities do not produce ``NaN``.

    Parameters
    ----------
    proba_matrix : np.ndarray of shape (n_samples, n_classes)
        Per-class probability vectors.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Shannon entropy in bits.  A value of 0 means perfect certainty; the
        maximum equals :math:`\\log_2(n\\_classes)` for a uniform distribution.
    """
    proba = np.asarray(proba_matrix, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(
            f"proba_matrix must be 2-D, got shape {proba.shape}."
        )

    # Clip to avoid log(0); values <= 0 are set to a tiny positive number,
    # but their contribution is forced to zero below.
    safe = np.clip(proba, 1e-300, None)
    log_proba = np.log2(safe)

    # Enforce 0 * log(0) = 0.
    terms = np.where(proba > 0, proba * log_proba, 0.0)
    entropy = -terms.sum(axis=1)

    return entropy


def compute_margin(proba_matrix: np.ndarray) -> np.ndarray:
    """Compute the margin between the top-two class probabilities.

    For a binary or single-class edge case (n_classes < 2), the margin
    equals the max probability itself.

    Parameters
    ----------
    proba_matrix : np.ndarray of shape (n_samples, n_classes)
        Per-class probability vectors.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Difference ``p_1st - p_2nd`` per sample.  Ranges from 0 (tie)
        to 1 (single dominant class).
    """
    proba = np.asarray(proba_matrix, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(
            f"proba_matrix must be 2-D, got shape {proba.shape}."
        )

    if proba.shape[1] < 2:
        logger.warning(
            "Probability matrix has fewer than 2 classes (%d); "
            "margin equals max probability.",
            proba.shape[1],
        )
        return proba.max(axis=1)

    # Partition so that the last two elements are the two largest.
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    first = top2.max(axis=1)
    second = top2.min(axis=1)
    return first - second


# ---------------------------------------------------------------------------
# Combined confidence DataFrame
# ---------------------------------------------------------------------------

def compute_confidence_scores(proba_matrix: np.ndarray) -> pd.DataFrame:
    """Compute all three confidence metrics and return as a DataFrame.

    Parameters
    ----------
    proba_matrix : np.ndarray of shape (n_samples, n_classes)
        Per-class probability vectors.

    Returns
    -------
    pd.DataFrame
        Columns: ``max_probability``, ``entropy``, ``margin``.
    """
    proba = np.asarray(proba_matrix, dtype=np.float64)

    max_prob = compute_max_probability(proba)
    entropy = compute_entropy(proba)
    margin = compute_margin(proba)

    df = pd.DataFrame({
        "max_probability": max_prob,
        "entropy": entropy,
        "margin": margin,
    })

    logger.info(
        "Confidence scores — max_prob: mean=%.3f, entropy: mean=%.3f, "
        "margin: mean=%.3f (%d samples)",
        max_prob.mean(),
        entropy.mean(),
        margin.mean(),
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Low-confidence flagging
# ---------------------------------------------------------------------------

def flag_low_confidence(
    confidence_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.Series:
    """Flag samples whose max probability falls below *threshold*.

    Parameters
    ----------
    confidence_df : pd.DataFrame
        Must contain a ``max_probability`` column (as produced by
        :func:`compute_confidence_scores`).
    threshold : float, optional
        Probability threshold below which a sample is considered
        low-confidence.  Defaults to 0.5.

    Returns
    -------
    pd.Series of bool
        ``True`` for samples with ``max_probability < threshold``.
    """
    if "max_probability" not in confidence_df.columns:
        raise KeyError(
            "confidence_df must contain a 'max_probability' column."
        )

    mask = confidence_df["max_probability"] < threshold
    n_flagged = mask.sum()
    pct = 100.0 * n_flagged / len(mask) if len(mask) > 0 else 0.0

    logger.info(
        "Low-confidence flag (threshold=%.2f): %d / %d samples (%.1f%%)",
        threshold,
        n_flagged,
        len(mask),
        pct,
    )
    return mask


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

def calibrate_probabilities(
    model,
    X_val: Union[np.ndarray, pd.DataFrame],
    y_val: Union[np.ndarray, pd.Series],
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Wrap a trained classifier with post-hoc probability calibration.

    Uses :class:`~sklearn.calibration.CalibratedClassifierCV` with
    ``cv='prefit'`` so the original model is not re-trained.  Calibration
    is fitted on the provided validation data.

    Parameters
    ----------
    model
        A trained scikit-learn-compatible estimator that exposes
        ``predict_proba``.
    X_val : array-like of shape (n_samples, n_features)
        Validation feature matrix.
    y_val : array-like of shape (n_samples,)
        Validation target vector.
    method : str, optional
        Calibration method: ``'isotonic'`` (default) or ``'sigmoid'``
        (Platt scaling).

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model that can be used as a drop-in replacement for
        the original (exposes ``predict`` and ``predict_proba``).

    Raises
    ------
    ValueError
        If *method* is not one of ``'isotonic'`` or ``'sigmoid'``.
    """
    valid_methods = ("isotonic", "sigmoid")
    if method not in valid_methods:
        raise ValueError(
            f"Calibration method must be one of {valid_methods}, got '{method}'."
        )

    logger.info(
        "Calibrating probabilities with method='%s' on %d validation samples.",
        method,
        len(y_val),
    )

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)

    logger.info("Probability calibration complete.")
    return calibrated
