"""High-level transformation utilities for the electrofacies pipeline.

Provides :class:`FaciesTransformer` -- a scikit-learn-style transformer that
encapsulates z-scoring and feature engineering -- plus standalone helpers for
winsorization and missing-value handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]

from electrofacies.preprocessing.features import (
    build_feature_columns,
    compute_training_stats,
    engineer_features,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FaciesTransformer
# ---------------------------------------------------------------------------


class FaciesTransformer:
    """Fit-transform wrapper that stores training statistics and engineers
    features consistently between training and inference.

    Typical workflow::

        transformer = FaciesTransformer(config=fe_config)
        train_df, feature_names = transformer.fit_transform(train_df, feature_cols)
        transformer.save("artifacts/transformer.joblib")

        # At inference time:
        transformer = FaciesTransformer.load("artifacts/transformer.joblib")
        well_df, feature_names = transformer.transform(well_df, feature_cols)

    Parameters
    ----------
    config : dict, optional
        Feature-engineering config section (``rolling_window``, ``diff_lag``,
        toggle flags, etc.).  Passed through to
        :func:`~electrofacies.preprocessing.features.engineer_features`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._stats: Optional[Dict[str, Tuple[float, float]]] = None
        self._feature_cols: Optional[List[str]] = None
        self._fitted: bool = False

    # -- Properties ----------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` if :meth:`fit` has been called."""
        return self._fitted

    @property
    def training_stats(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Per-column ``(mean, std)`` computed during :meth:`fit`."""
        return self._stats

    @property
    def feature_columns(self) -> Optional[List[str]]:
        """Base feature columns used during :meth:`fit`."""
        return self._feature_cols

    # -- Core interface ------------------------------------------------------

    def _apply_log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log10 transforms to configured columns.

        Columns listed in ``config["log_transforms"]`` are replaced with
        their ``log10`` values.  Values are clipped to a minimum of 1e-6
        before the transform to avoid ``-inf``.
        """
        log_cols = self.config.get("log_transforms", [])
        if not log_cols:
            return df
        out = df.copy()
        for col in log_cols:
            if col in out.columns:
                out[col] = np.log10(out[col].clip(lower=1e-6))
                logger.debug("Applied log10 transform to '%s'.", col)
        return out

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> "FaciesTransformer":
        """Compute and store training statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with canonical log columns.
        feature_cols : sequence of str
            Base canonical column names to compute stats for.

        Returns
        -------
        FaciesTransformer
            ``self``, for method chaining.
        """
        self._feature_cols = list(feature_cols)
        df_transformed = self._apply_log_transforms(df)
        self._stats = compute_training_stats(df_transformed, feature_cols)
        self._fitted = True
        logger.info(
            "FaciesTransformer fitted on %d rows, %d features.",
            len(df),
            len(feature_cols),
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply log transforms, z-scoring (using stored stats), and engineer all features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with canonical log columns.
        feature_cols : sequence of str, optional
            Base columns to transform.  Defaults to the columns used during
            :meth:`fit`.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            ``(engineered_df, feature_names)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError(
                "FaciesTransformer has not been fitted. Call fit() first."
            )
        if feature_cols is None:
            feature_cols = self._feature_cols
        if feature_cols is None:
            raise RuntimeError("No feature columns available.")

        df_transformed = self._apply_log_transforms(df)
        result_df, feat_names = engineer_features(
            df_transformed,
            feature_cols=feature_cols,
            config=self.config,
            training_stats=self._stats,
        )
        return result_df, feat_names

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Convenience method: fit on *df* then transform it.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.
        feature_cols : sequence of str
            Base canonical column names.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            ``(engineered_df, feature_names)``.
        """
        self.fit(df, feature_cols)
        return self.transform(df, feature_cols)

    # -- Persistence ---------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the transformer to disk with joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"artifacts/transformer.joblib"``).

        Raises
        ------
        ImportError
            If ``joblib`` is not installed.
        RuntimeError
            If the transformer has not been fitted.
        """
        if joblib is None:
            raise ImportError(
                "joblib is required for serialization. "
                "Install it with: pip install joblib"
            )
        if not self._fitted:
            raise RuntimeError(
                "Cannot save an unfitted FaciesTransformer."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config,
            "stats": self._stats,
            "feature_cols": self._feature_cols,
            "fitted": self._fitted,
        }
        joblib.dump(state, path)
        logger.info("FaciesTransformer saved to %s.", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FaciesTransformer":
        """Deserialize a transformer from disk.

        Parameters
        ----------
        path : str or Path
            Path to a previously saved transformer file.

        Returns
        -------
        FaciesTransformer
            Restored instance with training statistics intact.

        Raises
        ------
        ImportError
            If ``joblib`` is not installed.
        FileNotFoundError
            If *path* does not exist.
        """
        if joblib is None:
            raise ImportError(
                "joblib is required for deserialization. "
                "Install it with: pip install joblib"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Transformer file not found: {path}")
        state = joblib.load(path)
        obj = cls(config=state.get("config"))
        obj._stats = state.get("stats")
        obj._feature_cols = state.get("feature_cols")
        obj._fitted = state.get("fitted", False)
        logger.info("FaciesTransformer loaded from %s.", path)
        return obj

    def __repr__(self) -> str:  # pragma: no cover
        status = "fitted" if self._fitted else "unfitted"
        n_feats = len(self._feature_cols) if self._feature_cols else 0
        return (
            f"FaciesTransformer(status={status}, "
            f"n_base_features={n_feats}, "
            f"config_keys={list(self.config.keys())})"
        )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def winsorize(
    series: pd.Series,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    """Clip a series to quantile bounds (winsorization).

    Parameters
    ----------
    series : pd.Series
        Numeric series to clip.
    lower_q : float
        Lower quantile (default ``0.01`` = 1st percentile).
    upper_q : float
        Upper quantile (default ``0.99`` = 99th percentile).

    Returns
    -------
    pd.Series
        Clipped series.  NaN values are preserved.
    """
    if series.empty:
        logger.warning("winsorize received an empty Series.")
        return series.copy()

    non_null = series.dropna()
    if len(non_null) == 0:
        logger.warning("winsorize: all values are NaN; returning as-is.")
        return series.copy()

    lo = float(non_null.quantile(lower_q))
    hi = float(non_null.quantile(upper_q))

    if lo == hi:
        logger.warning(
            "winsorize: lower and upper quantile bounds are equal (%.4f); "
            "no clipping applied.",
            lo,
        )
        return series.copy()

    result = series.clip(lower=lo, upper=hi)
    n_clipped = int(((series < lo) | (series > hi)).sum())
    if n_clipped > 0:
        logger.info(
            "Winsorized %d values in '%s' to [%.4f, %.4f].",
            n_clipped,
            series.name or "series",
            lo,
            hi,
        )
    return result


def handle_missing(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    strategy: str = "ffill_bfill",
) -> pd.DataFrame:
    """Handle NaN values in feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_cols : sequence of str
        Columns in which to handle missing values.  Columns not present in
        *df* are silently skipped.
    strategy : str
        Strategy name:

        - ``"ffill_bfill"`` (default): forward-fill then back-fill.
          Appropriate for inference where dropping rows is undesirable.
        - ``"drop"``: drop any row with a NaN in *feature_cols*.
          Appropriate for training.
        - ``"zero"``: fill NaNs with ``0.0``.

    Returns
    -------
    pd.DataFrame
        DataFrame with NaNs handled according to *strategy*.

    Raises
    ------
    ValueError
        If *strategy* is not recognised.
    """
    valid_strategies = {"ffill_bfill", "drop", "zero"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown missing-data strategy '{strategy}'. "
            f"Choose from {valid_strategies}."
        )

    out = df.copy()
    cols_present = [c for c in feature_cols if c in out.columns]

    if not cols_present:
        logger.warning(
            "handle_missing: none of the specified feature columns are "
            "present in the DataFrame."
        )
        return out

    n_before = int(out[cols_present].isna().sum().sum())

    if strategy == "ffill_bfill":
        out[cols_present] = out[cols_present].ffill().bfill()
    elif strategy == "drop":
        before_len = len(out)
        out = out.dropna(subset=cols_present)
        dropped = before_len - len(out)
        if dropped > 0:
            logger.info(
                "handle_missing(drop): removed %d rows (%.1f%%).",
                dropped,
                dropped / max(before_len, 1) * 100,
            )
        return out
    elif strategy == "zero":
        out[cols_present] = out[cols_present].fillna(0.0)

    n_after = int(out[cols_present].isna().sum().sum())
    n_filled = n_before - n_after
    if n_filled > 0:
        logger.info(
            "handle_missing(%s): filled %d NaN values across %d columns.",
            strategy,
            n_filled,
            len(cols_present),
        )
    if n_after > 0:
        logger.warning(
            "handle_missing(%s): %d NaN values remain after imputation.",
            strategy,
            n_after,
        )

    return out
