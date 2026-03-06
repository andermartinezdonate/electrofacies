"""Output writers for the electrofacies pipeline.

Functions for persisting predictions, per-well reports, and batch summary
tables.  All writers use the ``logging`` module for diagnostics; nothing is
printed to stdout.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import lasio
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_predictions_csv(
    predictions_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Write a predictions DataFrame to CSV.

    The DataFrame is expected to be indexed by depth.  The depth index is
    written as an explicit column so the file is self-contained.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions indexed by depth (must include at least a ``"FACIES"``
        column).
    output_path : str or Path
        Destination CSV path.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        predictions_df.to_csv(output_path, index=True, float_format="%.4f")
        logger.info("Wrote predictions CSV: %s (%d rows)", output_path, len(predictions_df))
    except Exception:
        logger.exception("Failed to write predictions CSV to %s", output_path)
        raise

    return output_path


def write_predictions_las(
    predictions_df: pd.DataFrame,
    original_las_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """Append predicted facies curves to a copy of the original LAS file.

    Reads the original LAS, appends every column from *predictions_df* as a
    new curve (aligning on the depth index), and writes the augmented file
    to *output_path*.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions indexed by depth.  Commonly contains ``"FACIES"`` and
        per-class probability columns.
    original_las_path : str or Path
        Path to the source ``.las`` file.
    output_path : str or Path
        Destination ``.las`` path.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    original_las_path = Path(original_las_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not original_las_path.is_file():
        raise FileNotFoundError(f"Original LAS file not found: {original_las_path}")

    try:
        las = lasio.read(str(original_las_path), ignore_header_errors=True)
    except Exception:
        logger.exception("Failed to read original LAS %s for augmentation", original_las_path)
        raise

    # The LAS depth index.
    las_depth = las.index

    for col in predictions_df.columns:
        # Reindex predictions onto the LAS depth grid.  Depths not present
        # in predictions will be filled with the LAS null value.
        pred_series = predictions_df[col]
        aligned = np.full(len(las_depth), las.well["NULL"].value if "NULL" in [h.mnemonic for h in las.well] else -999.25)

        for i, d in enumerate(las_depth):
            if d in pred_series.index:
                aligned[i] = pred_series.loc[d]

        # Determine a sensible unit string.
        unit = ""
        descr = col
        if col.upper() == "FACIES":
            unit = "code"
            descr = "Predicted lithofacies"
        elif col.upper().startswith("PROB_"):
            unit = "v/v"
            descr = f"Class probability: {col}"
        elif col.upper() == "CONFIDENCE":
            unit = "v/v"
            descr = "Prediction confidence"
        elif col.upper() == "QC_FLAG":
            unit = "flag"
            descr = "QC flag"

        las.append_curve(col, aligned, unit=unit, descr=descr)

    try:
        with open(output_path, mode="w") as f:
            las.write(f, version=2.0)
        logger.info("Wrote augmented LAS: %s (%d new curves)", output_path, len(predictions_df.columns))
    except Exception:
        logger.exception("Failed to write augmented LAS to %s", output_path)
        raise

    return output_path


def write_well_report(
    well_result: Dict,
    output_dir: Union[str, Path],
) -> Path:
    """Write a per-well JSON report with QC summary.

    Parameters
    ----------
    well_result : dict
        A dictionary describing the processing outcome for a single well.
        Expected keys (all optional except ``well_name``):

        - ``well_name`` (str)
        - ``uwi`` (str)
        - ``tier_used`` (str)
        - ``algorithm`` (str)
        - ``qc_summary`` (dict)
        - ``metadata`` (dict)
        - ``status`` (str) — ``"success"`` or ``"failed"``
        - ``error`` (str)

    output_dir : str or Path
        Directory in which to write the report.

    Returns
    -------
    Path
        Resolved path of the written JSON file.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    well_name = well_result.get("well_name", "unknown_well")
    # Sanitise the well name for use as a filename.
    safe_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in well_name)
    report_path = output_dir / f"{safe_name}_report.json"

    report = {
        "well_name": well_name,
        "uwi": well_result.get("uwi", ""),
        "tier_used": well_result.get("tier_used", ""),
        "algorithm": well_result.get("algorithm", ""),
        "status": well_result.get("status", "unknown"),
        "error": well_result.get("error", None),
        "qc_summary": well_result.get("qc_summary", {}),
        "metadata": well_result.get("metadata", {}),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=_json_serialiser)
        logger.info("Wrote well report: %s", report_path)
    except Exception:
        logger.exception("Failed to write well report for %s", well_name)
        raise

    return report_path


def write_batch_summary(
    all_results: List[Dict],
    output_path: Union[str, Path],
) -> Path:
    """Write a summary CSV of all wells processed in a batch run.

    Each row corresponds to one well.  Columns include well name, UWI,
    processing status, tier used, algorithm, and selected QC metrics.

    Parameters
    ----------
    all_results : list of dict
        One dict per well.  Same structure as *well_result* in
        :func:`write_well_report`.
    output_path : str or Path
        Destination CSV path.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for res in all_results:
        qc = res.get("qc_summary", {})
        row = {
            "well_name": res.get("well_name", ""),
            "uwi": res.get("uwi", ""),
            "status": res.get("status", "unknown"),
            "tier_used": res.get("tier_used", ""),
            "algorithm": res.get("algorithm", ""),
            "num_predictions": qc.get("num_predictions", ""),
            "mean_confidence": qc.get("mean_confidence", ""),
            "low_confidence_frac": qc.get("low_confidence_fraction", ""),
            "ood_fraction": qc.get("ood_fraction", ""),
            "log_coverage": qc.get("log_coverage", ""),
            "error": res.get("error", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    try:
        df.to_csv(output_path, index=False)
        logger.info(
            "Wrote batch summary: %s (%d wells, %d succeeded)",
            output_path,
            len(df),
            (df["status"] == "success").sum(),
        )
    except Exception:
        logger.exception("Failed to write batch summary to %s", output_path)
        raise

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_serialiser(obj):
    """Fallback serialiser for ``json.dump`` to handle numpy/pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
