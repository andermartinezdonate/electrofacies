"""Train/validation splitting strategies for single-well electrofacies training.

Provides depth-blocked and grouped cross-validation splits that respect the
spatial autocorrelation inherent in well-log data.  A simple stratified split
is included as a fallback when depth information is unavailable.
"""

from __future__ import annotations

import logging
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Depth grouping
# ---------------------------------------------------------------------------

def make_depth_groups(depth_series: pd.Series, group_size: float = 50.0) -> np.ndarray:
    """Assign each sample to a depth block for grouped cross-validation.

    Parameters
    ----------
    depth_series : pd.Series
        Measured depth values (in feet).
    group_size : float, optional
        Size of each depth block in feet.  Defaults to 50.

    Returns
    -------
    np.ndarray
        Integer group labels with the same length as *depth_series*.
    """
    depth = np.asarray(depth_series, dtype=np.float64)
    if len(depth) == 0:
        return np.array([], dtype=np.intp)

    min_depth = np.nanmin(depth)
    groups = ((depth - min_depth) / group_size).astype(np.intp)
    n_groups = len(np.unique(groups))
    logger.info(
        "Created %d depth groups of ~%.0f ft (depth range %.1f - %.1f ft)",
        n_groups,
        group_size,
        np.nanmin(depth),
        np.nanmax(depth),
    )
    return groups


# ---------------------------------------------------------------------------
# Grouped K-Fold generator
# ---------------------------------------------------------------------------

def create_grouped_kfold_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,  # noqa: ARG001 — kept for API symmetry
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (train_idx, val_idx) tuples using :class:`~sklearn.model_selection.GroupKFold`.

    Entire depth groups are kept together so that no depth interval straddles
    the train/validation boundary.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    groups : np.ndarray of shape (n_samples,)
        Group labels produced by :func:`make_depth_groups`.
    n_splits : int, optional
        Number of folds (default 5).
    random_state : int, optional
        Accepted for API consistency; ``GroupKFold`` is deterministic.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        ``(train_indices, validation_indices)`` for each fold.
    """
    n_unique_groups = len(np.unique(groups))
    if n_splits > n_unique_groups:
        logger.warning(
            "Requested %d folds but only %d groups available; "
            "reducing n_splits to %d.",
            n_splits,
            n_unique_groups,
            n_unique_groups,
        )
        n_splits = n_unique_groups

    gkf = GroupKFold(n_splits=n_splits)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.debug(
            "Fold %d/%d — train %d samples, val %d samples",
            fold_idx + 1,
            n_splits,
            len(train_idx),
            len(val_idx),
        )
        yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Stratified split (fallback)
# ---------------------------------------------------------------------------

def create_stratified_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple stratified train/test index split (no depth awareness).

    Use as a fallback when reliable depth information is unavailable.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    test_size : float, optional
        Fraction of samples for the test set (default 0.25).
    random_state : int, optional
        Random seed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(train_indices, test_indices)``.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(sss.split(X, y))
    logger.info(
        "Stratified split — train %d, test %d (test_size=%.2f)",
        len(train_idx),
        len(test_idx),
        test_size,
    )
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Depth-blocked split
# ---------------------------------------------------------------------------

def create_depth_blocked_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    depth_series: pd.Series | np.ndarray,
    test_fraction: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by contiguous depth blocks — bottom N% of depth goes to test.

    This partially mitigates spatial autocorrelation for single-well training
    by ensuring that the test set is a contiguous interval rather than random
    rows scattered throughout the well.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix (used only for length validation).
    y : array-like of shape (n_samples,)
        Target vector (used only for length validation).
    depth_series : pd.Series or np.ndarray
        Measured depth values corresponding to each row of *X*.
    test_fraction : float, optional
        Fraction of the depth range to assign to the test set (default 0.25).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(train_indices, test_indices)``.
    """
    depth = np.asarray(depth_series, dtype=np.float64)
    n_samples = len(depth)

    if n_samples == 0:
        raise ValueError("Cannot split an empty dataset.")

    if len(X) != n_samples or len(y) != n_samples:
        raise ValueError(
            f"Length mismatch: X({len(X)}), y({len(y)}), "
            f"depth({n_samples})."
        )

    min_depth = np.nanmin(depth)
    max_depth = np.nanmax(depth)
    depth_range = max_depth - min_depth

    # The split threshold: everything above this depth is train,
    # everything at or below is test (deeper = bottom of well).
    split_depth = max_depth - test_fraction * depth_range

    train_mask = depth < split_depth
    test_mask = ~train_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    logger.info(
        "Depth-blocked split at %.1f ft — train %d samples (%.1f–%.1f ft), "
        "test %d samples (%.1f–%.1f ft)",
        split_depth,
        len(train_idx),
        min_depth,
        split_depth,
        len(test_idx),
        split_depth,
        max_depth,
    )

    if len(train_idx) == 0 or len(test_idx) == 0:
        logger.warning(
            "Depth-blocked split produced an empty partition; "
            "consider adjusting test_fraction (currently %.2f).",
            test_fraction,
        )

    return train_idx, test_idx
