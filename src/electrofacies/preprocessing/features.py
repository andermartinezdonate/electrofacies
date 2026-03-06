"""Feature engineering for the electrofacies prediction pipeline.

Generates z-scores, rolling statistics, diffs, and relative-depth features
from canonical well-log columns.  All engineering parameters are driven by
config to ensure consistency between training and inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from electrofacies.io.schemas import DEPTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suffix conventions (keep in sync with build_feature_columns)
# ---------------------------------------------------------------------------
_ZSCORE_SUFFIX = "_zscore"
_ROLL_MEAN_SUFFIX = "_rmean"
_ROLL_STD_SUFFIX = "_rstd"
_DIFF_SUFFIX = "_diff"
_RELDEPTH_COL = "relative_depth"

# ---------------------------------------------------------------------------
# Log-ratio / interaction features (petrophysically motivated)
# Each tuple: (output_name, log_A, log_B_or_None, description)
# Only computed when both constituent logs are in base_features.
# ---------------------------------------------------------------------------
_RATIO_DEFS: List[Tuple[str, str, Optional[str], str]] = [
    # Neutron-density separation — classic lithology/gas discriminator
    ("NPHI_RHOB_sep", "NPHI", "RHOB", "neutron-density separation"),
    # GR normalized by resistivity — clay vs clean
    ("GR_RESD_ratio", "GR", "RESD", "GR/resistivity ratio"),
    # Acoustic impedance proxy (DTC * RHOB)
    ("DTC_RHOB_imp", "DTC", "RHOB", "acoustic impedance proxy"),
    # Neutron-density product
    ("NPHI_RHOB_prod", "NPHI", "RHOB", "porosity-density product"),
    # Log10 of resistivity — RESD is log-distributed
    ("log_RESD", "RESD", None, "log10 resistivity"),
    # GR / RHOB — gamma-density ratio
    ("GR_RHOB_ratio", "GR", "RHOB", "GR/density ratio"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_training_stats(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    """Compute mean and standard deviation for each feature column.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame (rows may contain NaNs -- they are ignored).
    feature_cols : sequence of str
        Columns for which to compute statistics.

    Returns
    -------
    dict
        ``{col: (mean, std)}`` for every column in *feature_cols*.
        If a column is all-NaN or has zero std, ``std`` is set to ``1.0``
        to avoid division-by-zero downstream.
    """
    stats: Dict[str, Tuple[float, float]] = {}
    for col in feature_cols:
        if col not in df.columns:
            logger.warning(
                "compute_training_stats: column '%s' not in DataFrame; "
                "defaulting to (0.0, 1.0).",
                col,
            )
            stats[col] = (0.0, 1.0)
            continue
        series = df[col].dropna()
        if len(series) == 0:
            logger.warning(
                "Column '%s' is entirely NaN; using (0.0, 1.0).", col
            )
            stats[col] = (0.0, 1.0)
            continue
        mu = float(series.mean())
        sd = float(series.std(ddof=1))
        if sd == 0.0 or np.isnan(sd):
            logger.warning(
                "Column '%s' has zero or NaN std; setting std=1.0.", col
            )
            sd = 1.0
        stats[col] = (mu, sd)
    logger.info("Computed training stats for %d features.", len(stats))
    return stats


def build_feature_columns(
    base_features: Sequence[str],
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return the full list of engineered column names.

    The order is deterministic and matches the output of
    :func:`engineer_features`:

    1. z-scores
    2. rolling means
    3. rolling stds
    4. diffs
    5. relative depth

    Parameters
    ----------
    base_features : sequence of str
        Canonical log names used as raw features (e.g. ``["GR", "RESD", ...]``).
    config : dict, optional
        Feature-engineering config section.  Currently unused but reserved for
        future toggles (e.g. disabling specific feature families).

    Returns
    -------
    list[str]
        Ordered list of all engineered column names.
    """
    cfg = config or {}
    cols: List[str] = []

    if cfg.get("include_z_scores", True):
        cols.extend(f"{f}{_ZSCORE_SUFFIX}" for f in base_features)
    if cfg.get("include_rolling_mean", True):
        cols.extend(f"{f}{_ROLL_MEAN_SUFFIX}" for f in base_features)
    if cfg.get("include_rolling_std", True):
        cols.extend(f"{f}{_ROLL_STD_SUFFIX}" for f in base_features)
    if cfg.get("include_diff", True):
        cols.extend(f"{f}{_DIFF_SUFFIX}" for f in base_features)
    if cfg.get("include_ratios", True):
        base_set = set(base_features)
        for rname, log_a, log_b, _desc in _RATIO_DEFS:
            if log_a in base_set and (log_b is None or log_b in base_set):
                cols.append(rname)

    if cfg.get("include_relative_depth", True):
        cols.append(_RELDEPTH_COL)

    return cols


def engineer_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Optional[Dict[str, Any]] = None,
    training_stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Produce engineered features from canonical log columns.

    Features generated per base column:

    * **z-score**: ``(x - mu) / sd``.  Uses ``training_stats`` when provided
      (inference), or computes from the input data (training).
    * **rolling mean**: centred rolling average with ``window`` (default 5).
    * **rolling std**: centred rolling standard deviation.
    * **diff**: first-order difference (lag 1).
    * **relative depth**: depth normalised to [0, 1] within the well.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with canonical log columns.  Must have a numeric index
        (depth) **or** a ``DEPTH`` column.
    feature_cols : sequence of str
        Base canonical log column names to engineer from.
    config : dict, optional
        Feature-engineering parameters.  Expected keys (all optional with
        sane defaults):

        - ``rolling_window`` (int, default 5)
        - ``diff_lag`` (int, default 1)
        - ``include_z_scores`` (bool, default True)
        - ``include_rolling_mean`` (bool, default True)
        - ``include_rolling_std`` (bool, default True)
        - ``include_diff`` (bool, default True)
        - ``include_relative_depth`` (bool, default True)

    training_stats : dict, optional
        ``{col: (mean, std)}`` from :func:`compute_training_stats`.  When
        ``None``, statistics are computed from *df* (appropriate at training
        time only).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(engineered_df, feature_names)`` where *engineered_df* is a **copy**
        of the input with new columns appended, and *feature_names* is the
        ordered list of all generated column names.
    """
    cfg = config or {}
    rolling_window: int = cfg.get("rolling_window", 5)
    diff_lag: int = cfg.get("diff_lag", 1)

    if df.empty:
        logger.warning("engineer_features received an empty DataFrame.")
        names = build_feature_columns(feature_cols, cfg)
        return df.copy(), names

    out = df.copy()

    # Resolve depth array for relative-depth calculation
    depth = _resolve_depth(out)

    # Compute or use supplied statistics for z-scores
    if training_stats is None:
        stats = compute_training_stats(out, feature_cols)
    else:
        stats = training_stats

    new_cols: List[str] = []

    # --- z-scores ---
    if cfg.get("include_z_scores", True):
        for col in feature_cols:
            zcol = f"{col}{_ZSCORE_SUFFIX}"
            mu, sd = stats.get(col, (0.0, 1.0))
            if col in out.columns:
                out[zcol] = (out[col] - mu) / sd
            else:
                logger.warning(
                    "Feature column '%s' missing; z-score will be NaN.", col
                )
                out[zcol] = np.nan
            new_cols.append(zcol)

    # --- rolling mean ---
    if cfg.get("include_rolling_mean", True):
        for col in feature_cols:
            mcol = f"{col}{_ROLL_MEAN_SUFFIX}"
            if col in out.columns:
                out[mcol] = (
                    out[col]
                    .rolling(window=rolling_window, min_periods=1, center=True)
                    .mean()
                )
            else:
                out[mcol] = np.nan
            new_cols.append(mcol)

    # --- rolling std ---
    if cfg.get("include_rolling_std", True):
        for col in feature_cols:
            scol = f"{col}{_ROLL_STD_SUFFIX}"
            if col in out.columns:
                out[scol] = (
                    out[col]
                    .rolling(window=rolling_window, min_periods=1, center=True)
                    .std()
                )
                # Fill NaN from single-element windows with 0
                out[scol] = out[scol].fillna(0.0)
            else:
                out[scol] = np.nan
            new_cols.append(scol)

    # --- diff ---
    if cfg.get("include_diff", True):
        for col in feature_cols:
            dcol = f"{col}{_DIFF_SUFFIX}"
            if col in out.columns:
                out[dcol] = out[col].diff(periods=diff_lag)
            else:
                out[dcol] = np.nan
            new_cols.append(dcol)

    # --- log ratios / interactions ---
    if cfg.get("include_ratios", True):
        base_set = set(feature_cols)
        for rname, log_a, log_b, desc in _RATIO_DEFS:
            if log_a not in base_set:
                continue
            if log_b is not None and log_b not in base_set:
                continue
            if log_a not in out.columns:
                out[rname] = np.nan
                new_cols.append(rname)
                continue

            a = out[log_a]
            if rname == "NPHI_RHOB_sep":
                out[rname] = a - out[log_b]
            elif rname == "GR_RESD_ratio":
                out[rname] = a / out[log_b].clip(lower=0.01)
            elif rname == "DTC_RHOB_imp":
                out[rname] = a * out[log_b]
            elif rname == "NPHI_RHOB_prod":
                out[rname] = a * out[log_b]
            elif rname == "log_RESD":
                out[rname] = np.log10(a.clip(lower=0.01))
            elif rname == "GR_RHOB_ratio":
                out[rname] = a / out[log_b].clip(lower=1.0)
            new_cols.append(rname)

        n_ratios = sum(1 for rn, la, lb, _ in _RATIO_DEFS
                       if la in base_set and (lb is None or lb in base_set))
        if n_ratios > 0:
            logger.info("Added %d log-ratio features.", n_ratios)

    # --- relative depth ---
    if cfg.get("include_relative_depth", True):
        if depth is not None and len(depth) > 0:
            d_min = float(np.nanmin(depth))
            d_max = float(np.nanmax(depth))
            drange = d_max - d_min
            if drange == 0.0:
                logger.warning(
                    "Depth range is zero; relative_depth set to 0.0."
                )
                out[_RELDEPTH_COL] = 0.0
            else:
                out[_RELDEPTH_COL] = (depth - d_min) / drange
        else:
            logger.warning(
                "No depth information found; relative_depth set to NaN."
            )
            out[_RELDEPTH_COL] = np.nan
        new_cols.append(_RELDEPTH_COL)

    logger.info(
        "Engineered %d features from %d base columns.",
        len(new_cols),
        len(feature_cols),
    )
    return out, new_cols


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_depth(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract a depth array from the DataFrame index or a DEPTH column.

    Prefers the ``DEPTH`` column; falls back to the index if it is numeric.
    """
    if DEPTH in df.columns:
        return df[DEPTH].values.astype(float)
    if pd.api.types.is_numeric_dtype(df.index):
        return df.index.values.astype(float)
    logger.warning(
        "Cannot resolve depth: no '%s' column and index is not numeric.", DEPTH
    )
    return None
