"""Well-log data validation and quality control.

Provides range checking against physical limits, washout detection, flatline
detection, null-coverage auditing, and a master validation entry point that
aggregates all checks into a single QC report dictionary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from electrofacies.io.schemas import CANONICAL_LOGS, CALI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration loaders
# ---------------------------------------------------------------------------


def load_physical_ranges(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load *physical_ranges.yaml* and return the parsed dict.

    Parameters
    ----------
    config_path : str or Path
        Path to *physical_ranges.yaml*.

    Returns
    -------
    dict
        Full parsed YAML contents including ``ranges``, ``washout``, and
        ``quality`` sections.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded physical ranges from %s.", config_path)
    return config


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------


def validate_ranges(
    df: pd.DataFrame,
    ranges_config: Dict[str, Any],
) -> pd.DataFrame:
    """Check each log column against physical valid ranges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with canonical column names.
    ranges_config : dict
        The full parsed *physical_ranges.yaml* dict (must contain a
        ``"ranges"`` key).

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame of the same shape as *df* (only for columns present
        in *ranges_config*).  ``True`` = within valid range, ``False`` =
        out-of-range.  NaN values are marked ``False``.
    """
    ranges = ranges_config.get("ranges", {})
    flags = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col not in ranges:
            continue
        spec = ranges[col]
        lo = spec.get("min")
        hi = spec.get("max")
        series = df[col]
        valid = pd.Series(False, index=df.index, dtype=bool)
        not_null = series.notna()
        if lo is not None and hi is not None:
            valid[not_null] = (series[not_null] >= lo) & (series[not_null] <= hi)
        elif lo is not None:
            valid[not_null] = series[not_null] >= lo
        elif hi is not None:
            valid[not_null] = series[not_null] <= hi
        else:
            valid[not_null] = True
        flags[col] = valid

    n_violations = {
        col: int((~flags[col]).sum()) for col in flags.columns
    }
    if any(v > 0 for v in n_violations.values()):
        logger.warning("Range violations detected: %s", n_violations)
    else:
        logger.info("All values within physical ranges.")
    return flags


# ---------------------------------------------------------------------------
# Washout detection
# ---------------------------------------------------------------------------


def detect_washouts(
    df: pd.DataFrame,
    cali_col: str = CALI,
    bit_size: float = 8.75,
    excess_threshold: float = 2.0,
) -> pd.Series:
    """Detect washout intervals where caliper exceeds bit size by a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a caliper column.
    cali_col : str
        Name of the caliper column (default ``"CALI"``).
    bit_size : float
        Nominal bit size in inches (default ``8.75``).
    excess_threshold : float
        Caliper excess in inches above *bit_size* to flag as washout
        (default ``2.0``).

    Returns
    -------
    pd.Series
        Boolean mask: ``True`` where washout is detected.
    """
    if cali_col not in df.columns:
        logger.info(
            "Caliper column '%s' not present; skipping washout detection.",
            cali_col,
        )
        return pd.Series(False, index=df.index, dtype=bool, name="washout")

    cali = df[cali_col]
    washout = pd.Series(False, index=df.index, dtype=bool, name="washout")
    not_null = cali.notna()
    washout[not_null] = cali[not_null] > (bit_size + excess_threshold)

    n_washout = int(washout.sum())
    if n_washout > 0:
        frac = n_washout / len(df)
        logger.warning(
            "Washouts detected: %d samples (%.1f%%).",
            n_washout,
            frac * 100,
        )
    else:
        logger.info("No washouts detected.")
    return washout


# ---------------------------------------------------------------------------
# Flatline detection
# ---------------------------------------------------------------------------


def detect_flatlines(
    series: pd.Series,
    window: int = 20,
    std_threshold: float = 0.001,
) -> pd.Series:
    """Detect flatlined (stuck) intervals in a log curve.

    A flatline is identified where the rolling standard deviation over
    *window* samples falls below *std_threshold*.

    Parameters
    ----------
    series : pd.Series
        Single log-curve series.
    window : int
        Rolling window size (default ``20``).
    std_threshold : float
        Standard-deviation threshold below which the signal is considered
        stuck (default ``0.001``).

    Returns
    -------
    pd.Series
        Boolean mask: ``True`` where the curve is flatlined.
    """
    name = series.name or "series"
    if series.empty:
        logger.warning("detect_flatlines received an empty Series ('%s').", name)
        return pd.Series(dtype=bool, name=f"{name}_flatline")

    if series.isna().all():
        logger.warning(
            "All values NaN in '%s'; marking entire series as flatlined.", name
        )
        return pd.Series(True, index=series.index, dtype=bool, name=f"{name}_flatline")

    roll_std = series.rolling(window=window, min_periods=max(1, window // 2)).std()
    flatline = roll_std.fillna(0.0) < std_threshold

    # Don't flag NaN regions as flatline — mark them False
    flatline[series.isna()] = False

    n_flat = int(flatline.sum())
    if n_flat > 0:
        frac = n_flat / len(series)
        logger.warning(
            "Flatline detected in '%s': %d samples (%.1f%%).",
            name,
            n_flat,
            frac * 100,
        )
    return flatline.rename(f"{name}_flatline")


# ---------------------------------------------------------------------------
# Null coverage
# ---------------------------------------------------------------------------


def check_null_coverage(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    threshold: float = 0.30,
) -> Dict[str, Any]:
    """Compute null fraction per column and flag those exceeding a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : sequence of str, optional
        Columns to check.  If ``None``, all columns are checked.
    threshold : float
        Maximum allowed null fraction (default ``0.30``).

    Returns
    -------
    dict
        ``{"coverage": {col: frac_null, ...}, "flagged": [cols exceeding threshold]}``.
    """
    if columns is None:
        columns = list(df.columns)

    n_rows = len(df)
    coverage: Dict[str, float] = {}
    flagged: List[str] = []

    for col in columns:
        if col not in df.columns:
            coverage[col] = 1.0  # entirely missing
            flagged.append(col)
            continue
        if n_rows == 0:
            coverage[col] = 1.0
            flagged.append(col)
            continue
        frac_null = float(df[col].isna().sum()) / n_rows
        coverage[col] = round(frac_null, 4)
        if frac_null > threshold:
            flagged.append(col)

    if flagged:
        logger.warning(
            "Columns exceeding %.0f%% null threshold: %s",
            threshold * 100,
            flagged,
        )
    else:
        logger.info("All checked columns within null-coverage threshold.")

    return {"coverage": coverage, "flagged": flagged}


# ---------------------------------------------------------------------------
# Master validation
# ---------------------------------------------------------------------------


def validate_well(
    df: pd.DataFrame,
    ranges_config: Dict[str, Any],
    required_logs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run all validation checks on a well DataFrame.

    This is the top-level quality-control entry point.  It aggregates range
    validation, null-coverage, washout, and flatline checks into a single
    report dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Well data with canonical column names.
    ranges_config : dict
        Parsed *physical_ranges.yaml* contents.
    required_logs : list of str, optional
        Canonical log names that must be present for the well to be
        considered valid.  Defaults to ``CANONICAL_LOGS`` from schemas.

    Returns
    -------
    dict
        Comprehensive QC report with keys:

        - ``valid`` (bool): overall pass/fail
        - ``issues`` (list[str]): human-readable issue descriptions
        - ``null_coverage`` (dict): output of :func:`check_null_coverage`
        - ``range_violations`` (dict): ``{col: n_violations}``
        - ``washout_fraction`` (float): fraction of washout intervals
        - ``flatline_report`` (dict): ``{col: n_flatline_samples}``
        - ``usable_logs`` (list[str]): canonical logs present and passing QC
    """
    if required_logs is None:
        required_logs = list(CANONICAL_LOGS)

    quality_cfg = ranges_config.get("quality", {})
    washout_cfg = ranges_config.get("washout", {})
    null_threshold = quality_cfg.get("max_null_fraction", 0.30)
    flatline_window = quality_cfg.get("flatline_window", 20)
    flatline_std_thr = quality_cfg.get("flatline_std_threshold", 0.001)

    issues: List[str] = []
    usable_logs: List[str] = []

    # --- Empty check ---
    if df.empty:
        logger.error("validate_well received an empty DataFrame.")
        return {
            "valid": False,
            "issues": ["DataFrame is empty."],
            "null_coverage": {},
            "range_violations": {},
            "washout_fraction": 0.0,
            "flatline_report": {},
            "usable_logs": [],
        }

    # --- Check presence of required logs ---
    present_logs = [log for log in required_logs if log in df.columns]
    missing_logs = [log for log in required_logs if log not in df.columns]
    if missing_logs:
        issues.append(f"Missing required logs: {missing_logs}")

    # --- Null coverage ---
    null_report = check_null_coverage(
        df, columns=present_logs, threshold=null_threshold
    )

    # --- Range validation ---
    range_flags = validate_ranges(df, ranges_config)
    range_violations: Dict[str, int] = {}
    for col in range_flags.columns:
        n_bad = int((~range_flags[col]).sum())
        if n_bad > 0:
            range_violations[col] = n_bad
            frac = n_bad / len(df)
            if frac > 0.10:
                issues.append(
                    f"{col}: {n_bad} range violations ({frac:.1%})."
                )

    # --- Washout detection ---
    bit_size = washout_cfg.get("bit_size_default", 8.75)
    excess = washout_cfg.get("caliper_excess_inches", 2.0)
    washout_mask = detect_washouts(
        df, cali_col=CALI, bit_size=bit_size, excess_threshold=excess
    )
    washout_fraction = float(washout_mask.sum()) / len(df) if len(df) > 0 else 0.0
    if washout_fraction > 0.20:
        issues.append(
            f"High washout fraction: {washout_fraction:.1%}."
        )

    # --- Flatline detection ---
    flatline_report: Dict[str, int] = {}
    for col in present_logs:
        fl_mask = detect_flatlines(
            df[col], window=flatline_window, std_threshold=flatline_std_thr
        )
        n_flat = int(fl_mask.sum())
        if n_flat > 0:
            flatline_report[col] = n_flat
            frac = n_flat / len(df)
            if frac > 0.10:
                issues.append(
                    f"{col}: flatline detected in {n_flat} samples ({frac:.1%})."
                )

    # --- Determine usable logs ---
    flagged_null = set(null_report.get("flagged", []))
    for col in present_logs:
        if col in flagged_null:
            issues.append(
                f"{col}: null fraction {null_report['coverage'].get(col, '?'):.1%} "
                f"exceeds threshold."
            )
            continue
        # Accept if range violations are < 10% of total
        n_viol = range_violations.get(col, 0)
        if len(df) > 0 and (n_viol / len(df)) > 0.10:
            continue
        usable_logs.append(col)

    # --- Overall validity ---
    valid = len(issues) == 0

    report = {
        "valid": valid,
        "issues": issues,
        "null_coverage": null_report,
        "range_violations": range_violations,
        "washout_fraction": round(washout_fraction, 4),
        "flatline_report": flatline_report,
        "usable_logs": usable_logs,
    }

    if valid:
        logger.info("Well validation PASSED. Usable logs: %s", usable_logs)
    else:
        logger.warning(
            "Well validation found %d issue(s). Usable logs: %s",
            len(issues),
            usable_logs,
        )

    return report
