"""QC report generation for electrofacies predictions.

Produces per-well quality summaries and batch-level reports that aggregate
confidence statistics, out-of-distribution flags, and facies distributions
into human-readable and machine-parseable formats.

Every well receives a **QC grade**:

- ``PASS``   -- <20 % low-confidence samples AND <10 % OOD.
- ``REVIEW`` -- <40 % low-confidence samples AND <20 % OOD.
- ``FAIL``   -- anything worse.

Reports are written as CSV (for programmatic consumption) and plain-text
(for quick human review).
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-sample QC status
# ---------------------------------------------------------------------------

def format_qc_status(low_confidence: bool, ood_flag: bool) -> str:
    """Combine boolean QC flags into a single human-readable status string.

    Parameters
    ----------
    low_confidence : bool
        ``True`` if the sample has low prediction confidence.
    ood_flag : bool
        ``True`` if the sample is flagged as out-of-distribution.

    Returns
    -------
    str
        One of ``'GOOD'``, ``'LOW_CONFIDENCE'``, ``'OOD'``, or
        ``'LOW_CONF_AND_OOD'``.
    """
    if low_confidence and ood_flag:
        return "LOW_CONF_AND_OOD"
    if low_confidence:
        return "LOW_CONFIDENCE"
    if ood_flag:
        return "OOD"
    return "GOOD"


# ---------------------------------------------------------------------------
# Per-well QC summary
# ---------------------------------------------------------------------------

def generate_well_qc(
    predictions_df: pd.DataFrame,
    well_name: str,
    tier: str,
    algorithm: str,
) -> dict:
    """Generate a quality-control summary for a single well's predictions.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Prediction results for one well.  Expected columns:

        - ``FACIES`` -- predicted facies label or integer code.
        - ``max_probability`` -- highest class probability (from
          :func:`~electrofacies.qc.confidence.compute_confidence_scores`).
        - ``ood_flag`` -- boolean OOD flag (from
          :class:`~electrofacies.qc.ood.OODDetector`).

        The DataFrame may also contain probability columns, entropy, margin,
        etc., but they are not required for the summary.
    well_name : str
        Human-readable well identifier.
    tier : str
        Model tier identifier (e.g. ``'tier_1'``).
    algorithm : str
        Algorithm name (e.g. ``'random_forest'``, ``'xgboost'``).

    Returns
    -------
    dict
        Nested dictionary with keys: ``n_samples``, ``n_predicted``,
        ``facies_distribution``, ``confidence``, ``ood``, ``qc_grade``,
        ``tier_used``, ``algorithm_used``, ``timestamp``.
    """
    n_samples = len(predictions_df)

    # ----- facies distribution -----
    facies_col = next(
        (c for c in ("PREDICTED_FACIES", "FACIES") if c in predictions_df.columns),
        None,
    )
    has_facies = facies_col is not None
    if has_facies:
        facies_series = predictions_df[facies_col]
        n_predicted = int(facies_series.notna().sum())
        counts = Counter(facies_series.dropna())
        facies_distribution = {
            str(facies): {
                "count": int(cnt),
                "fraction": round(cnt / n_predicted, 4) if n_predicted > 0 else 0.0,
            }
            for facies, cnt in sorted(counts.items(), key=lambda x: -x[1])
        }
    else:
        n_predicted = 0
        facies_distribution = {}
        logger.warning(
            "Well '%s': no FACIES column found in predictions_df.", well_name
        )

    # ----- confidence statistics -----
    conf_col = next(
        (c for c in ("CONFIDENCE_SCORE", "max_probability") if c in predictions_df.columns),
        None,
    )
    has_conf = conf_col is not None
    if has_conf:
        conf = predictions_df[conf_col]
        conf_valid = conf.dropna()
        pct_below = (
            float((conf_valid < 0.5).sum() / len(conf_valid) * 100.0)
            if len(conf_valid) > 0
            else 0.0
        )
        confidence_stats: Dict = {
            "mean": round(float(conf_valid.mean()), 4) if len(conf_valid) > 0 else None,
            "median": round(float(conf_valid.median()), 4) if len(conf_valid) > 0 else None,
            "min": round(float(conf_valid.min()), 4) if len(conf_valid) > 0 else None,
            "pct_below_threshold": round(pct_below, 2),
        }
    else:
        pct_below = 0.0
        confidence_stats = {
            "mean": None,
            "median": None,
            "min": None,
            "pct_below_threshold": 0.0,
        }
        logger.warning(
            "Well '%s': no max_probability column found.", well_name
        )

    # ----- OOD statistics -----
    ood_col = next(
        (c for c in ("OOD_FLAG", "ood_flag") if c in predictions_df.columns),
        None,
    )
    has_ood = ood_col is not None
    if has_ood:
        ood_series = predictions_df[ood_col].astype(bool)
        n_ood = int(ood_series.sum())
        pct_ood = round(
            float(n_ood / len(ood_series) * 100.0) if len(ood_series) > 0 else 0.0,
            2,
        )
    else:
        n_ood = 0
        pct_ood = 0.0
        logger.warning(
            "Well '%s': no ood_flag column found.", well_name
        )

    ood_stats: Dict = {
        "n_flagged": n_ood,
        "pct_flagged": pct_ood,
    }

    # ----- QC grade -----
    qc_grade = _assign_qc_grade(pct_below, pct_ood)

    summary = {
        "well_name": well_name,
        "n_samples": n_samples,
        "n_predicted": n_predicted,
        "facies_distribution": facies_distribution,
        "confidence": confidence_stats,
        "ood": ood_stats,
        "qc_grade": qc_grade,
        "tier_used": tier,
        "algorithm_used": algorithm,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Well QC '%s': grade=%s, n_predicted=%d, pct_low_conf=%.1f%%, "
        "pct_ood=%.1f%%.",
        well_name,
        qc_grade,
        n_predicted,
        pct_below,
        pct_ood,
    )
    return summary


# ---------------------------------------------------------------------------
# Batch report
# ---------------------------------------------------------------------------

def generate_batch_report(
    well_summaries: List[dict],
    output_dir: Union[str, Path],
) -> str:
    """Generate batch-level QC reports from a list of per-well summaries.

    Produces two files under *output_dir*:

    1. ``batch_summary.csv`` -- one row per well with key metrics.
    2. ``batch_qc_report.txt`` -- human-readable summary.

    Parameters
    ----------
    well_summaries : list of dict
        Each element is a per-well summary as returned by
        :func:`generate_well_qc`.
    output_dir : str or Path
        Directory in which to write the report files.  Created if it does
        not exist.

    Returns
    -------
    str
        Absolute path to the written ``batch_summary.csv``.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "batch_summary.csv"
    txt_path = output_dir / "batch_qc_report.txt"

    # ----- CSV summary -----
    rows = []
    for ws in well_summaries:
        rows.append({
            "well_name": ws.get("well_name", ""),
            "uwi": ws.get("uwi", ""),
            "n_samples": ws.get("n_samples", 0),
            "tier": ws.get("tier_used", ""),
            "algorithm": ws.get("algorithm_used", ""),
            "qc_grade": ws.get("qc_grade", ""),
            "pct_low_conf": ws.get("confidence", {}).get("pct_below_threshold", 0.0),
            "pct_ood": ws.get("ood", {}).get("pct_flagged", 0.0),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False)
    logger.info("Wrote batch summary CSV: %s (%d wells).", csv_path, len(rows))

    # ----- Human-readable text report -----
    total_wells = len(well_summaries)
    grade_counts = Counter(ws.get("qc_grade", "UNKNOWN") for ws in well_summaries)

    # Aggregate facies distribution.
    agg_facies: Counter = Counter()
    total_predicted = 0
    for ws in well_summaries:
        for facies, info in ws.get("facies_distribution", {}).items():
            count = info.get("count", 0) if isinstance(info, dict) else 0
            agg_facies[facies] += count
            total_predicted += count

    # Tier usage distribution.
    tier_counts = Counter(ws.get("tier_used", "unknown") for ws in well_summaries)

    lines = [
        "=" * 72,
        "ELECTROFACIES BATCH QC REPORT",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 72,
        "",
        f"Total wells processed: {total_wells}",
        "",
        "QC Grade Distribution:",
        f"  PASS   : {grade_counts.get('PASS', 0)}",
        f"  REVIEW : {grade_counts.get('REVIEW', 0)}",
        f"  FAIL   : {grade_counts.get('FAIL', 0)}",
        "",
        f"Total predicted samples: {total_predicted}",
        "",
        "Aggregate Facies Distribution:",
    ]

    for facies, count in sorted(agg_facies.items(), key=lambda x: -x[1]):
        frac = count / total_predicted * 100.0 if total_predicted > 0 else 0.0
        lines.append(f"  {facies:30s}  {count:>8d}  ({frac:5.1f}%)")

    lines.append("")
    lines.append("Tier Usage:")
    for tier, cnt in sorted(tier_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {tier:30s}  {cnt:>4d} wells")

    lines.append("")
    lines.append("=" * 72)
    lines.append("")

    txt_content = "\n".join(lines)
    txt_path.write_text(txt_content, encoding="utf-8")
    logger.info("Wrote batch QC text report: %s.", txt_path)

    return str(csv_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_qc_grade(pct_low_confidence: float, pct_ood: float) -> str:
    """Determine the QC grade from percentage thresholds.

    Parameters
    ----------
    pct_low_confidence : float
        Percentage of samples with max_probability below the confidence
        threshold (0--100).
    pct_ood : float
        Percentage of samples flagged as out-of-distribution (0--100).

    Returns
    -------
    str
        ``'PASS'``, ``'REVIEW'``, or ``'FAIL'``.
    """
    if pct_low_confidence < 20.0 and pct_ood < 10.0:
        return "PASS"
    if pct_low_confidence < 40.0 and pct_ood < 20.0:
        return "REVIEW"
    return "FAIL"
