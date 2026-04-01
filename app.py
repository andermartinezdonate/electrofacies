"""Streamlit web application for the Electrofacies Prediction System.

Upload one or more LAS files, automatically predict lithofacies using ML
models trained on the PDB03 well from the Delaware Mountain Group, and
download results with full QC diagnostics.

Built by Ander Martinez-Donate — University of Texas at Austin.
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Project root and config paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("electrofacies.app")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# UT Austin palette & constants
# ---------------------------------------------------------------------------
UT_BURNT_ORANGE = "#BF5700"
UT_CHARCOAL = "#333F48"
PLOT_BG = "#FFFFFF"
PLOT_GRID_COLOR = "#CCCCCC"

FACIES_COLORS: Dict[str, str] = {
    "massive_sandstone": "#f3e6b3",
    "structured_sandstone": "#e6b657",
    "sandy_siltstone": "#a7663a",
    "siltstone": "#505660",
    "calciturbidite": "#2ca25f",
    "clast_supported_conglomerate": "#7e3fa0",
    "missing_strata": "#ffffff",
}

FACIES_ORDER = [
    "massive_sandstone",
    "structured_sandstone",
    "sandy_siltstone",
    "siltstone",
    "calciturbidite",
    "clast_supported_conglomerate",
]

# Darkened log colors for white background
LOG_COLORS: Dict[str, str] = {
    "GR": "#1a7a3e",
    "RESD": "#222222",
    "RHOB": "#c41020",
    "NPHI": "#2060a0",
    "DTC": "#d06000",
}

TRACK_SPECS = [
    ("GR", 0, 300, False, "GR", "API"),
    ("RESD", 0.1, 10000, True, "RESD", "ohm.m"),
    ("RHOB", 1.9, 2.9, False, "RHOB", "g/cm\u00b3"),
    ("NPHI", 0.45, -0.05, False, "NPHI", "v/v"),
    ("DTC", 140, 40, False, "DTC", "\u00b5s/ft"),
]

QC_STATUS_COLORS = {
    "GOOD": "#2ca25f",
    "LOW_CONFIDENCE": "#e6a817",
    "OOD": "#de2d26",
    "LOW_CONF_AND_OOD": "#8b0000",
}

QC_GRADE_COLORS = {
    "PASS": "#2ca25f",
    "REVIEW": "#e6a817",
    "FAIL": "#de2d26",
}


# ===================================================================
# Cached config loaders
# ===================================================================

@st.cache_resource
def load_mnemonic_map_cached():
    from electrofacies.preprocessing.standardize import load_mnemonic_map
    return load_mnemonic_map(CONFIGS_DIR / "mnemonic_aliases.yaml")


@st.cache_resource
def load_tier_config_cached():
    from electrofacies.inference.tier_router import load_tier_config
    return load_tier_config(CONFIGS_DIR / "model_tiers.yaml")


@st.cache_resource
def load_physical_ranges_cached():
    from electrofacies.preprocessing.validate import load_physical_ranges
    return load_physical_ranges(CONFIGS_DIR / "physical_ranges.yaml")


@st.cache_resource
def load_facies_schema_cached():
    with open(CONFIGS_DIR / "facies_schema.yaml") as fh:
        return yaml.safe_load(fh)


@st.cache_resource
def load_default_config_cached():
    with open(CONFIGS_DIR / "default.yaml") as fh:
        return yaml.safe_load(fh)


@st.cache_resource
def load_all_model_metrics() -> Dict[str, Dict]:
    """Load metrics.json from every artifact bundle."""
    metrics = {}
    if not ARTIFACTS_DIR.is_dir():
        return metrics
    for bundle_dir in sorted(ARTIFACTS_DIR.iterdir()):
        if not bundle_dir.is_dir():
            continue
        metrics_path = bundle_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as fh:
                data = json.load(fh)
            # Derive a readable name from the directory
            name = bundle_dir.name
            # Strip timestamp suffix  e.g. tier_1_random_forest_20260309T...
            parts = name.split("_")
            # Find the timestamp part (starts with 20)
            label_parts = []
            for p in parts:
                if p.startswith("20") and len(p) > 8:
                    break
                label_parts.append(p)
            label = "_".join(label_parts)
            metrics[label] = data
    return metrics


def load_tier_models_cached(artifacts_dir: str, tier_name: str):
    from electrofacies.inference.tier_router import load_tier_models
    return load_tier_models(artifacts_dir, tier_name)


# ===================================================================
# Pipeline helpers
# ===================================================================

def read_las_from_upload(uploaded_file) -> Tuple[Dict[str, Any], str]:
    """Parse an uploaded LAS file via a temp file.

    Returns (well_data, temp_las_path) — the temp path is needed later for
    LAS download via write_predictions_las().
    """
    from electrofacies.io.readers import read_las

    with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    return read_las(tmp_path), tmp_path


def standardize_well(curves_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    from electrofacies.preprocessing.standardize import standardize_columns
    mnemonic_map = load_mnemonic_map_cached()
    return standardize_columns(curves_df, mnemonic_map)


def validate_well_data(df: pd.DataFrame) -> Dict[str, Any]:
    from electrofacies.preprocessing.validate import validate_well
    ranges_config = load_physical_ranges_cached()
    return validate_well(df, ranges_config)


def determine_available_logs_for_well(df: pd.DataFrame) -> List[str]:
    from electrofacies.inference.tier_router import determine_available_logs
    from electrofacies.io.schemas import CANONICAL_LOGS
    return determine_available_logs(df, CANONICAL_LOGS)


def select_tier(available_logs: List[str]):
    from electrofacies.inference.tier_router import select_best_tier
    tier_config = load_tier_config_cached()
    return select_best_tier(available_logs, tier_config["tiers"])


def run_prediction(
    well_data: pd.DataFrame,
    tier_name: str,
    artifacts_dir: Path,
    config: Dict,
) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """Run the full prediction pipeline for a single well.

    Returns (algorithm_name, predictions_df) or (None, None) on failure.
    """
    from electrofacies.inference.predict import (
        predict_all_algorithms,
        select_best_prediction,
    )
    from electrofacies.inference.postprocess import (
        assign_confidence_flags,
        assign_ood_flags,
        assign_qc_status,
        modal_filter,
    )
    from electrofacies.preprocessing.transform import FaciesTransformer

    tier_models = load_tier_models_cached(str(artifacts_dir), tier_name)
    if not tier_models:
        return None, None

    # Load transformer from the first bundle
    first_bundle = next(iter(tier_models.values()))
    transformer = first_bundle.get("transformer")

    if transformer is None:
        bundle_dir = first_bundle.get("_bundle_dir", "")
        if bundle_dir:
            candidate = Path(bundle_dir) / "transformer.joblib"
            if candidate.exists():
                transformer = FaciesTransformer.load(str(candidate))

    if transformer is None:
        for algo_name_inner, bundle in tier_models.items():
            meta = bundle.get("metadata", {})
            bdir = meta.get("bundle_dir", "")
            if bdir:
                t_path = Path(bdir) / "transformer.joblib"
                if t_path.exists():
                    transformer = FaciesTransformer.load(str(t_path))
                    break

    if transformer is None:
        fe_config = config.get("feature_engineering", {})
        transformer = FaciesTransformer(config=fe_config)
        tier_cfg = load_tier_config_cached()
        tier_info = tier_cfg["tiers"].get(tier_name, {})
        required_logs = tier_info.get("required_logs", [])
        available = [c for c in required_logs if c in well_data.columns]
        if available:
            transformer.fit(well_data, available)

    # Run predictions
    all_preds = predict_all_algorithms(well_data, tier_models, transformer, config)
    if not all_preds:
        return None, None

    algo_name, best_preds = select_best_prediction(all_preds)

    # Post-process
    if "PREDICTED_FACIES" in best_preds.columns:
        best_preds["PREDICTED_FACIES"] = modal_filter(
            best_preds["PREDICTED_FACIES"], window=3
        )

    best_preds = assign_confidence_flags(best_preds, threshold=0.5)

    # OOD detection — try loading detector from bundle
    ood_detector = first_bundle.get("ood_detector")
    if ood_detector is not None:
        best_preds = assign_ood_flags(best_preds, well_data, ood_detector)
    else:
        best_preds["OOD_FLAG"] = False

    best_preds = assign_qc_status(best_preds)

    return algo_name, best_preds


def compute_summary_for_well(
    well_name: str,
    predictions_df: pd.DataFrame,
    tier_name: str,
    algo_name: str,
    val_report: Dict,
) -> Dict[str, Any]:
    """Build one row for the multi-well summary table."""
    from electrofacies.inference.postprocess import compute_well_summary

    summary = compute_well_summary(predictions_df, well_name, tier_name, algo_name)

    # Dominant facies
    if "PREDICTED_FACIES" in predictions_df.columns:
        dominant = predictions_df["PREDICTED_FACIES"].value_counts().index[0]
        summary["dominant_facies"] = format_facies_name(dominant)
    else:
        summary["dominant_facies"] = "N/A"

    summary["washout_pct"] = round(val_report.get("washout_fraction", 0) * 100, 1)
    return summary


# ===================================================================
# Orchestrator — process all uploaded wells
# ===================================================================

def process_all_wells(uploaded_files) -> List[Dict[str, Any]]:
    """Process every uploaded LAS file and return a list of result dicts."""
    config = load_default_config_cached()
    artifacts_dir = PROJECT_ROOT / config.get("paths", {}).get("artifacts_dir", "artifacts")

    results = []
    progress_bar = st.progress(0, text="Processing wells...")

    for idx, uploaded_file in enumerate(uploaded_files):
        well_result: Dict[str, Any] = {
            "filename": uploaded_file.name,
            "status": "failed",
            "error": None,
        }

        try:
            # 1. Read LAS
            well_data, temp_las_path = read_las_from_upload(uploaded_file)
            curves_df = well_data["curves"]
            metadata = well_data["metadata"]
            well_name = metadata.get("well_name", uploaded_file.name)
            well_result["well_name"] = well_name
            well_result["metadata"] = metadata
            well_result["temp_las_path"] = temp_las_path

            # 2. Standardize
            std_df, mapping_report = standardize_well(curves_df)
            well_result["std_df"] = std_df
            well_result["mapping_report"] = mapping_report

            # 3. Validate
            val_report = validate_well_data(std_df)
            well_result["validation_report"] = val_report

            # 4. Available logs & tier
            available_logs = determine_available_logs_for_well(std_df)
            well_result["available_logs"] = available_logs

            tier_result = select_tier(available_logs)
            if tier_result is None:
                well_result["error"] = (
                    f"No model tier satisfied. Available logs: {available_logs}. "
                    "Need at least GR and RESD."
                )
                results.append(well_result)
                progress_bar.progress(
                    (idx + 1) / len(uploaded_files),
                    text=f"Processed {idx + 1}/{len(uploaded_files)} wells...",
                )
                continue

            tier_name, tier_info = tier_result
            well_result["tier_name"] = tier_name
            well_result["tier_info"] = tier_info

            # 5. Check models exist
            try:
                tier_models = load_tier_models_cached(str(artifacts_dir), tier_name)
                if not tier_models:
                    raise ValueError("No model bundles loaded")
            except Exception as e:
                well_result["error"] = f"No trained models for {tier_name}: {e}"
                results.append(well_result)
                progress_bar.progress(
                    (idx + 1) / len(uploaded_files),
                    text=f"Processed {idx + 1}/{len(uploaded_files)} wells...",
                )
                continue

            # 6. Predict
            algo_name, predictions_df = run_prediction(
                std_df, tier_name, artifacts_dir, config
            )
            if predictions_df is None:
                well_result["error"] = "Prediction pipeline returned no results."
                results.append(well_result)
                progress_bar.progress(
                    (idx + 1) / len(uploaded_files),
                    text=f"Processed {idx + 1}/{len(uploaded_files)} wells...",
                )
                continue

            well_result["algo_name"] = algo_name
            well_result["predictions_df"] = predictions_df

            # 7. Summary
            well_result["summary"] = compute_summary_for_well(
                well_name, predictions_df, tier_name, algo_name, val_report
            )
            well_result["status"] = "success"

        except Exception as e:
            logger.exception("Failed to process %s", uploaded_file.name)
            well_result["error"] = str(e)

        results.append(well_result)
        progress_bar.progress(
            (idx + 1) / len(uploaded_files),
            text=f"Processed {idx + 1}/{len(uploaded_files)} wells...",
        )

    progress_bar.empty()
    return results


# ===================================================================
# Visualization helpers (light theme)
# ===================================================================

def format_facies_name(name: str) -> str:
    return name.replace("_", " ").title()


def plot_log_tracks(
    well_df: pd.DataFrame,
    predictions_df: Optional[pd.DataFrame] = None,
    validation_report: Optional[Dict] = None,
    title: str = "",
) -> plt.Figure:
    """Multi-track well-log display with optional facies, confidence, QC strips
    and washout overlays.  White background version.
    """
    from electrofacies.preprocessing.validate import detect_washouts

    available = [t for t in TRACK_SPECS if t[0] in well_df.columns]
    n_logs = len(available)

    has_facies = predictions_df is not None and "PREDICTED_FACIES" in predictions_df.columns
    has_conf = predictions_df is not None and "CONFIDENCE_SCORE" in predictions_df.columns
    has_qc = predictions_df is not None and "QC_STATUS" in predictions_df.columns
    n_strips = sum([has_facies, has_conf, has_qc])
    n_total = max(n_logs + n_strips, 1)

    width_ratios = [1.2] * n_logs + [0.5] * n_strips
    if not width_ratios:
        width_ratios = [1]

    fig_width = sum(width_ratios) * 1.0 + 1.5
    fig_height = 12

    fig, axes = plt.subplots(
        1, n_total,
        figsize=(fig_width, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05},
    )
    if n_total == 1:
        axes = np.array([axes])

    fig.patch.set_facecolor(PLOT_BG)

    depths = well_df.index.values.astype(float)
    d_min, d_max = np.nanmin(depths), np.nanmax(depths)

    # Detect washout zones for overlay
    washout_mask = detect_washouts(well_df)

    axes[0].set_ylim(d_max, d_min)
    axes[0].set_ylabel("Depth (ft)", fontsize=9, color=UT_CHARCOAL)

    for idx, (mnem, xmin, xmax, log_scale, label, unit) in enumerate(available):
        ax = axes[idx]
        ax.set_facecolor(PLOT_BG)
        curve = well_df[mnem].values.astype(float)
        color = LOG_COLORS.get(mnem, "#555555")

        if log_scale:
            ax.set_xscale("log")

        ax.plot(curve, depths, color=color, linewidth=0.8, alpha=0.9)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(f"{label} ({unit})", fontsize=8, color=color)
        ax.tick_params(axis="x", colors=color, labelsize=7)
        ax.tick_params(axis="y", colors=UT_CHARCOAL, labelsize=7)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.3, color=PLOT_GRID_COLOR)
        ax.grid(True, which="major", axis="x", linewidth=0.3, alpha=0.3, color=PLOT_GRID_COLOR)

        for spine in ax.spines.values():
            spine.set_color("#BBBBBB")
            spine.set_linewidth(0.5)

        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

        # Washout overlay — gray hatched zones
        if washout_mask.any():
            washout_depths = depths[washout_mask.values]
            if len(washout_depths) > 0:
                # Group consecutive washout samples into intervals
                intervals = _group_consecutive_depths(
                    washout_depths, depths
                )
                for y0, y1 in intervals:
                    ax.axhspan(
                        y0, y1, facecolor="#CCCCCC", alpha=0.3,
                        edgecolor="none",
                    )

    # Prediction strips
    strip_idx = n_logs

    if has_facies:
        ax = axes[strip_idx]
        ax.set_facecolor(PLOT_BG)
        pred_depths = (
            predictions_df["DEPTH"].values
            if "DEPTH" in predictions_df.columns
            else predictions_df.index.values
        )
        pred_depths = np.asarray(pred_depths, dtype=float)
        facies_vals = predictions_df["PREDICTED_FACIES"].values

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        if len(pred_depths) >= 2:
            for i in range(len(pred_depths) - 1):
                clr = FACIES_COLORS.get(str(facies_vals[i]), "#cccccc")
                ax.axhspan(pred_depths[i], pred_depths[i + 1], facecolor=clr, edgecolor="none")
            clr = FACIES_COLORS.get(str(facies_vals[-1]), "#cccccc")
            step = pred_depths[-1] - pred_depths[-2]
            ax.axhspan(pred_depths[-1], pred_depths[-1] + step, facecolor=clr, edgecolor="none")

        ax.set_xlabel("Facies", fontsize=8, color=UT_CHARCOAL)
        ax.xaxis.set_label_position("top")
        for spine in ax.spines.values():
            spine.set_color("#BBBBBB")
            spine.set_linewidth(0.3)
        strip_idx += 1

    if has_conf:
        ax = axes[strip_idx]
        ax.set_facecolor(PLOT_BG)
        pred_depths = (
            predictions_df["DEPTH"].values
            if "DEPTH" in predictions_df.columns
            else predictions_df.index.values
        )
        pred_depths = np.asarray(pred_depths, dtype=float)
        conf_vals = predictions_df["CONFIDENCE_SCORE"].values.astype(float)

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        if len(pred_depths) >= 2:
            for i in range(len(pred_depths) - 1):
                c = np.clip(conf_vals[i], 0, 1)
                r, g = 1.0 - c, c
                ax.axhspan(pred_depths[i], pred_depths[i + 1],
                           facecolor=(r, g, 0.2), edgecolor="none")
            c = np.clip(conf_vals[-1], 0, 1)
            step = pred_depths[-1] - pred_depths[-2]
            ax.axhspan(pred_depths[-1], pred_depths[-1] + step,
                       facecolor=(1.0 - c, c, 0.2), edgecolor="none")

        ax.set_xlabel("Conf.", fontsize=8, color=UT_CHARCOAL)
        ax.xaxis.set_label_position("top")
        for spine in ax.spines.values():
            spine.set_color("#BBBBBB")
            spine.set_linewidth(0.3)
        strip_idx += 1

    if has_qc:
        ax = axes[strip_idx]
        ax.set_facecolor(PLOT_BG)
        pred_depths = (
            predictions_df["DEPTH"].values
            if "DEPTH" in predictions_df.columns
            else predictions_df.index.values
        )
        pred_depths = np.asarray(pred_depths, dtype=float)
        qc_vals = predictions_df["QC_STATUS"].values

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        if len(pred_depths) >= 2:
            for i in range(len(pred_depths) - 1):
                clr = QC_STATUS_COLORS.get(str(qc_vals[i]), "#999999")
                ax.axhspan(pred_depths[i], pred_depths[i + 1], facecolor=clr, edgecolor="none")
            clr = QC_STATUS_COLORS.get(str(qc_vals[-1]), "#999999")
            step = pred_depths[-1] - pred_depths[-2]
            ax.axhspan(pred_depths[-1], pred_depths[-1] + step, facecolor=clr, edgecolor="none")

        ax.set_xlabel("QC", fontsize=8, color=UT_CHARCOAL)
        ax.xaxis.set_label_position("top")
        for spine in ax.spines.values():
            spine.set_color("#BBBBBB")
            spine.set_linewidth(0.3)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=UT_CHARCOAL, y=1.01)

    fig.tight_layout()
    return fig


def _group_consecutive_depths(
    washout_depths: np.ndarray, all_depths: np.ndarray
) -> List[Tuple[float, float]]:
    """Group consecutive washout depth samples into (y_top, y_bot) intervals."""
    if len(washout_depths) == 0:
        return []
    # Typical step size
    step = np.median(np.diff(all_depths)) if len(all_depths) > 1 else 0.5
    intervals = []
    start = washout_depths[0]
    prev = washout_depths[0]
    for d in washout_depths[1:]:
        if abs(d - prev) > step * 2:
            intervals.append((start, prev + step))
            start = d
        prev = d
    intervals.append((start, prev + step))
    return intervals


def make_facies_legend() -> plt.Figure:
    """Horizontal facies colour legend — light theme."""
    fig, ax = plt.subplots(figsize=(8, 0.6))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)

    patches = [
        mpatches.Patch(
            facecolor=FACIES_COLORS.get(f, "#ccc"),
            edgecolor="#999999",
            linewidth=0.5,
            label=format_facies_name(f),
        )
        for f in FACIES_ORDER
    ]
    ax.legend(
        handles=patches, loc="center", ncol=len(patches),
        frameon=False, fontsize=8, labelcolor=UT_CHARCOAL,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_facies_pie(predictions_df: pd.DataFrame) -> plt.Figure:
    facies_counts = predictions_df["PREDICTED_FACIES"].value_counts()
    colors = [FACIES_COLORS.get(f, "#cccccc") for f in facies_counts.index]
    labels = [format_facies_name(f) for f in facies_counts.index]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)

    wedges, texts, autotexts = ax.pie(
        facies_counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", pctdistance=0.8,
        wedgeprops={"edgecolor": PLOT_BG, "linewidth": 1.5},
        textprops={"fontsize": 8, "color": UT_CHARCOAL},
    )
    for t in autotexts:
        t.set_fontsize(7)
        t.set_color(UT_CHARCOAL)

    ax.set_title("Facies Distribution", fontsize=11, color=UT_CHARCOAL, pad=10)
    fig.tight_layout()
    return fig


def plot_confidence_histogram(predictions_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor("#FAFAFA")

    conf = predictions_df["CONFIDENCE_SCORE"].values
    ax.hist(conf, bins=30, color=UT_BURNT_ORANGE, edgecolor=PLOT_BG, alpha=0.85)
    ax.axvline(0.5, color="#de2d26", linestyle="--", linewidth=1.2, label="Threshold (0.5)")
    ax.set_xlabel("Confidence Score", fontsize=9, color=UT_CHARCOAL)
    ax.set_ylabel("Count", fontsize=9, color=UT_CHARCOAL)
    ax.set_title("Confidence Distribution", fontsize=11, color=UT_CHARCOAL)
    ax.tick_params(colors=UT_CHARCOAL, labelsize=7)
    ax.legend(fontsize=8, labelcolor=UT_CHARCOAL, facecolor=PLOT_BG, edgecolor="#CCC")

    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
    fig.tight_layout()
    return fig


# ===================================================================
# Download helpers
# ===================================================================

def generate_csv_download(predictions_df: pd.DataFrame) -> str:
    buf = io.StringIO()
    predictions_df.to_csv(buf, index=True, float_format="%.4f")
    return buf.getvalue()


def generate_las_download(
    predictions_df: pd.DataFrame,
    temp_las_path: str,
    facies_schema: Dict,
) -> Optional[bytes]:
    """Generate a LAS file with predictions appended.

    Facies are encoded as integer codes (0-5) since LAS only stores numeric
    curves.
    """
    from electrofacies.io.writers import write_predictions_las

    # Build a numeric-only predictions DataFrame for LAS writing
    las_df = pd.DataFrame(index=predictions_df.index)

    if "PREDICTED_FACIES" in predictions_df.columns:
        # Map facies names to integer codes
        facies_info = facies_schema.get("facies", {})
        code_map = {name: info.get("code", i) for i, (name, info) in enumerate(facies_info.items())}
        las_df["FACIES"] = predictions_df["PREDICTED_FACIES"].map(code_map).fillna(-1).astype(int)

    if "CONFIDENCE_SCORE" in predictions_df.columns:
        las_df["CONFIDENCE"] = predictions_df["CONFIDENCE_SCORE"]

    if "QC_STATUS" in predictions_df.columns:
        qc_code_map = {"GOOD": 0, "LOW_CONFIDENCE": 1, "OOD": 2, "LOW_CONF_AND_OOD": 3}
        las_df["QC_FLAG"] = predictions_df["QC_STATUS"].map(qc_code_map).fillna(-1).astype(int)

    try:
        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as out_tmp:
            out_path = out_tmp.name

        write_predictions_las(las_df, temp_las_path, out_path)

        with open(out_path, "rb") as fh:
            return fh.read()
    except Exception as e:
        logger.exception("Failed to generate LAS download: %s", e)
        return None


def generate_png_download(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor=PLOT_BG)
    buf.seek(0)
    return buf.getvalue()


# ===================================================================
# UI building blocks
# ===================================================================

def render_footer():
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.85em; padding: 10px 0;'>"
        "Built by <b>Ander Martinez-Doñate</b> &nbsp;|&nbsp; "
        "<a href='https://www.linkedin.com/in/andermart' target='_blank'>LinkedIn</a> &nbsp;|&nbsp; "
        "<a href='https://github.com/andermartinezdonate' target='_blank'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_facies_descriptions():
    """Expandable facies description cards."""
    facies_schema = load_facies_schema_cached()
    with st.expander("Facies Classification Scheme", expanded=False):
        cols = st.columns(3)
        for i, facies_name in enumerate(FACIES_ORDER):
            info = facies_schema["facies"].get(facies_name, {})
            color = FACIES_COLORS.get(facies_name, "#ccc")
            with cols[i % 3]:
                st.markdown(
                    f"<div style='border-left: 4px solid {color}; padding-left: 8px; "
                    f"margin-bottom: 10px;'>"
                    f"<b>{format_facies_name(facies_name)}</b><br>"
                    f"<small>{info.get('description', '')}</small></div>",
                    unsafe_allow_html=True,
                )


def render_model_metrics():
    """Expandable model accuracy/kappa table."""
    with st.expander("About the Model", expanded=False):
        st.markdown(
            "Models are trained on the **PDB03** well from the "
            "**Delaware Mountain Group** (Permian Basin, West Texas). "
            "Each tier uses **Random Forest**, **XGBoost**, and **Extra Trees**; the algorithm "
            "with the highest mean confidence is automatically selected at "
            "inference time."
        )

        metrics = load_all_model_metrics()
        if not metrics:
            st.info("No model metrics found in artifacts directory.")
            return

        rows = []
        for label, data in sorted(metrics.items()):
            parts = label.split("_")
            tier_label = f"{parts[0].title()} {parts[1]}" if len(parts) >= 2 else label
            algo_label = "_".join(parts[2:]) if len(parts) > 2 else ""
            algo_label = algo_label.replace("_", " ").title()

            rows.append({
                "Tier": tier_label,
                "Algorithm": algo_label,
                "Accuracy": f"{data.get('accuracy', 0):.1%}",
                "Balanced Acc.": f"{data.get('balanced_accuracy', 0):.1%}",
                "Cohen's Kappa": f"{data.get('cohen_kappa', 0):.3f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption(
            "Metrics computed on a held-out 25% test set using depth-grouped "
            "cross-validation. Training data: PDB03 well, 6 facies classes."
        )


# ===================================================================
# Per-well display (5 tabs)
# ===================================================================

def render_well_tabs(result: Dict[str, Any]):
    """Render the 5-tab view for a single well result."""
    well_name = result.get("well_name", result["filename"])
    std_df = result["std_df"]
    predictions_df = result["predictions_df"]
    val_report = result.get("validation_report", {})
    mapping_report = result.get("mapping_report", {})
    summary = result.get("summary", {})
    tier_name = result.get("tier_name", "")
    algo_name = result.get("algo_name", "")
    metadata = result.get("metadata", {})

    # Well header metrics
    meta_cols = st.columns(5)
    with meta_cols[0]:
        st.metric("Tier", tier_name.replace("_", " ").title())
    with meta_cols[1]:
        st.metric("Algorithm", algo_name.replace("_", " ").title() if algo_name else "N/A")
    with meta_cols[2]:
        mean_conf = summary.get("mean_confidence", 0)
        st.metric("Mean Confidence", f"{mean_conf:.1%}")
    with meta_cols[3]:
        grade = summary.get("overall_qc_grade", "N/A")
        st.metric("QC Grade", grade)
    with meta_cols[4]:
        n_samples = summary.get("n_samples", len(predictions_df))
        st.metric("Samples", f"{n_samples:,}")

    # Tabs
    tab_log, tab_stats, tab_data, tab_qc, tab_dl = st.tabs(
        ["Log Display", "Summary Stats", "Data Table", "QC Details", "Downloads"]
    )

    # --- Tab 1: Log Display ---
    with tab_log:
        fig_legend = make_facies_legend()
        st.pyplot(fig_legend, use_container_width=True)
        plt.close(fig_legend)

        fig = plot_log_tracks(
            std_df, predictions_df,
            validation_report=val_report,
            title=well_name,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # --- Tab 2: Summary Stats ---
    with tab_stats:
        chart_cols = st.columns(2)
        with chart_cols[0]:
            if "PREDICTED_FACIES" in predictions_df.columns:
                fig_pie = plot_facies_pie(predictions_df)
                st.pyplot(fig_pie, use_container_width=True)
                plt.close(fig_pie)

        with chart_cols[1]:
            if "CONFIDENCE_SCORE" in predictions_df.columns:
                fig_hist = plot_confidence_histogram(predictions_df)
                st.pyplot(fig_hist, use_container_width=True)
                plt.close(fig_hist)

        # Facies distribution table
        if "PREDICTED_FACIES" in predictions_df.columns:
            st.markdown("**Facies Distribution**")
            facies_counts = predictions_df["PREDICTED_FACIES"].value_counts()
            dist_df = pd.DataFrame({
                "Facies": [format_facies_name(f) for f in facies_counts.index],
                "Count": facies_counts.values,
                "Percentage": [f"{v / len(predictions_df) * 100:.1f}%" for v in facies_counts.values],
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # --- Tab 3: Data Table ---
    with tab_data:
        display_cols = [
            c for c in [
                "DEPTH", "PREDICTED_FACIES", "CONFIDENCE_SCORE",
                "MODEL_TIER", "ALGORITHM", "QC_STATUS",
            ]
            if c in predictions_df.columns
        ]
        st.dataframe(
            predictions_df[display_cols],
            use_container_width=True,
            hide_index=True,
            height=500,
        )
        st.caption(f"{len(predictions_df):,} rows total.")

    # --- Tab 4: QC Details ---
    with tab_qc:
        _render_qc_details(val_report, mapping_report, predictions_df, std_df)

    # --- Tab 5: Downloads ---
    with tab_dl:
        _render_downloads(well_name, predictions_df, std_df, result, val_report)


def _render_qc_details(
    val_report: Dict,
    mapping_report: Dict,
    predictions_df: pd.DataFrame,
    std_df: pd.DataFrame,
):
    """Full QC transparency tab."""
    # Overall QC status breakdown
    if "QC_STATUS" in predictions_df.columns:
        st.markdown("**QC Status Breakdown**")
        qc_counts = predictions_df["QC_STATUS"].value_counts()
        qc_df = pd.DataFrame({
            "Status": qc_counts.index,
            "Count": qc_counts.values,
            "Percentage": [
                f"{v / len(predictions_df) * 100:.1f}%"
                for v in qc_counts.values
            ],
        })
        st.dataframe(qc_df, use_container_width=True, hide_index=True)

    # Validation issues
    issues = val_report.get("issues", [])
    if issues:
        st.markdown("**Validation Issues**")
        for issue in issues:
            st.warning(issue)
    else:
        st.success("All validation checks passed.")

    # Null coverage
    null_cov = val_report.get("null_coverage", {}).get("coverage", {})
    if null_cov:
        st.markdown("**Null Coverage per Log**")
        cov_df = pd.DataFrame([
            {"Log": k, "Null %": f"{v * 100:.1f}%"}
            for k, v in null_cov.items()
        ])
        st.dataframe(cov_df, use_container_width=True, hide_index=True)

    # Washout fraction
    washout_frac = val_report.get("washout_fraction", 0)
    st.markdown(f"**Washout Fraction:** {washout_frac:.1%}")

    # Flatline detections
    flatline_report = val_report.get("flatline_report", {})
    if flatline_report:
        st.markdown("**Flatline Detections**")
        fl_df = pd.DataFrame([
            {"Log": k, "Flatline Samples": v, "Fraction": f"{v / len(std_df) * 100:.1f}%"}
            for k, v in flatline_report.items()
        ])
        st.dataframe(fl_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("**Flatline Detections:** None")

    # Range violations
    range_viol = val_report.get("range_violations", {})
    if range_viol:
        st.markdown("**Range Violations**")
        rv_df = pd.DataFrame([
            {"Log": k, "Violations": v, "Fraction": f"{v / len(std_df) * 100:.1f}%"}
            for k, v in range_viol.items()
        ])
        st.dataframe(rv_df, use_container_width=True, hide_index=True)

    # Mnemonic mapping
    st.markdown("**Mnemonic Mapping Report**")
    mapped = {k: v for k, v in mapping_report.items() if v != "unmapped"}
    unmapped = {k: v for k, v in mapping_report.items() if v == "unmapped"}

    if mapped:
        map_df = pd.DataFrame([
            {"Original": k, "Canonical": v} for k, v in mapped.items()
        ])
        st.dataframe(map_df, use_container_width=True, hide_index=True)

    if unmapped:
        st.caption(
            f"Unmapped curves ({len(unmapped)}): {', '.join(unmapped.keys())}"
        )


def _render_downloads(
    well_name: str,
    predictions_df: pd.DataFrame,
    std_df: pd.DataFrame,
    result: Dict,
    val_report: Dict,
):
    """Three download buttons: CSV, LAS, PNG."""
    dl_cols = st.columns(3)

    # CSV
    with dl_cols[0]:
        csv_data = generate_csv_download(predictions_df)
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_data,
            file_name=f"{well_name}_electrofacies.csv",
            mime="text/csv",
        )

    # LAS
    with dl_cols[1]:
        temp_las_path = result.get("temp_las_path")
        if temp_las_path:
            facies_schema = load_facies_schema_cached()
            las_data = generate_las_download(
                predictions_df, temp_las_path, facies_schema
            )
            if las_data:
                st.download_button(
                    label="Download Predictions (LAS)",
                    data=las_data,
                    file_name=f"{well_name}_electrofacies.las",
                    mime="application/octet-stream",
                )
            else:
                st.warning("LAS download generation failed.")
        else:
            st.warning("Original LAS path not available for augmented download.")

    # PNG
    with dl_cols[2]:
        fig_dl = plot_log_tracks(
            std_df, predictions_df,
            validation_report=val_report,
            title=well_name,
        )
        png_data = generate_png_download(fig_dl)
        plt.close(fig_dl)
        st.download_button(
            label="Download Log Display (PNG)",
            data=png_data,
            file_name=f"{well_name}_log_display.png",
            mime="image/png",
        )


# ===================================================================
# Main UI
# ===================================================================

def main():
    st.set_page_config(
        page_title="Electrofacies Predictor",
        page_icon="\U0001faa8",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ---- Title with burnt orange underline ----
    st.markdown(
        f"<h1 style='text-align: center; margin-bottom: 0;'>"
        f"\U0001faa8 Electrofacies Prediction System</h1>"
        f"<div style='text-align: center; margin: 0 auto 20px auto; "
        f"width: 120px; height: 4px; background-color: {UT_BURNT_ORANGE}; "
        f"border-radius: 2px;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align: center; color: #666; max-width: 700px; margin: 0 auto 20px auto;'>"
        "Upload LAS well-log files to predict lithofacies using machine learning "
        "models trained on the Delaware Mountain Group (Permian Basin, West Texas). "
        "Multi-well batch processing with full QC diagnostics.</p>",
        unsafe_allow_html=True,
    )

    # ---- Multi-file uploader ----
    uploaded_files = st.file_uploader(
        "Upload LAS well-log files",
        type=["las", "LAS"],
        accept_multiple_files=True,
        help="Standard LAS 2.0 or 3.0 format. Up to 50 MB per file.",
    )

    # ---- Landing page (no files uploaded) ----
    if not uploaded_files:
        _render_landing_page()
        return

    # ---- Process uploaded files ----
    # Use session state to avoid re-processing on every Streamlit rerun
    cache_key = tuple(f.name + str(f.size) for f in uploaded_files)

    if (
        "results" not in st.session_state
        or st.session_state.get("cache_key") != cache_key
    ):
        st.session_state["results"] = process_all_wells(uploaded_files)
        st.session_state["cache_key"] = cache_key

    results = st.session_state["results"]

    # ---- Summary table ----
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    if failed:
        for r in failed:
            st.warning(
                f"**{r.get('well_name', r['filename'])}** — {r.get('error', 'Unknown error')}"
            )

    if successful:
        st.markdown("### Multi-Well Summary")
        summary_rows = []
        for r in successful:
            s = r["summary"]
            summary_rows.append({
                "Well": r.get("well_name", r["filename"]),
                "Dominant Facies": s.get("dominant_facies", "N/A"),
                "Avg Confidence": f"{s.get('mean_confidence', 0):.1%}",
                "QC Grade": s.get("overall_qc_grade", "N/A"),
                "Tier": r.get("tier_name", "").replace("_", " ").title(),
                "Algorithm": (r.get("algo_name") or "").replace("_", " ").title(),
                "Washout %": f"{s.get('washout_pct', 0):.1f}%",
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ---- Per-well expanders ----
    auto_expand = len(successful) == 1

    for r in successful:
        well_name = r.get("well_name", r["filename"])
        grade = r.get("summary", {}).get("overall_qc_grade", "")
        grade_color = QC_GRADE_COLORS.get(grade, "#666")

        with st.expander(
            f"{well_name}  —  QC: {grade}",
            expanded=auto_expand,
        ):
            render_well_tabs(r)

    # ---- Bottom sections ----
    render_facies_descriptions()
    render_model_metrics()
    render_footer()


def _render_landing_page():
    """Show info sections when no files are uploaded."""
    st.divider()

    st.markdown("### How It Works")
    cols = st.columns(4)
    steps = [
        ("1. Upload", "Drop one or more LAS files above. We support 144+ vendor mnemonic aliases."),
        ("2. Validate", "Physical range checks, washout detection, flatline detection, null coverage audit."),
        ("3. Predict", "Auto-selects the best model tier based on available logs. Runs RF, XGBoost & Extra Trees."),
        ("4. Download", "Get predictions with confidence scores, QC flags, and publication-ready figures."),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"#### {title}")
            st.markdown(desc)

    # Facies descriptions
    render_facies_descriptions()

    # Training data display
    render_training_data()

    # Model metrics
    render_model_metrics()

    # Tier info
    with st.expander("Model Tier System", expanded=False):
        st.markdown("""
| Tier | Name | Logs Required | Priority |
|------|------|---------------|----------|
| Tier 1 | Full Suite | GR, RESD, RHOB, NPHI, DTC | Highest |
| Tier 2 | No Sonic | GR, RESD, RHOB, NPHI | High |
| Tier 3 | Triple Combo | GR, RESD, RHOB | Medium |
| Tier 4 | Minimal | GR, RESD | Baseline |

The system automatically selects the highest-priority tier whose required
logs are all available with at least 70% non-null coverage.
""")

    render_footer()


@st.cache_data
def _load_training_data():
    """Load the PDB03 training dataset for display."""
    config = load_default_config_cached()
    training_path = PROJECT_ROOT / config["paths"]["training_data"]
    if not training_path.exists():
        return None
    df = pd.read_excel(training_path, engine="openpyxl")
    return df


def render_training_data():
    """Show the training well's log curves and lithofacies."""
    with st.expander("Training Data — PDB03 Well", expanded=False):
        raw_df = _load_training_data()
        if raw_df is None:
            st.warning("Training data file not found.")
            return

        config = load_default_config_cached()
        train_cfg = config["training"]
        raw_to_canon = train_cfg["raw_to_canonical"]
        depth_col = train_cfg["depth_col"]
        target_col = train_cfg["target_col"]

        st.markdown(
            f"**{len(raw_df):,} samples** at 0.5 ft spacing &nbsp;|&nbsp; "
            f"**5 log curves** &nbsp;|&nbsp; "
            f"**{raw_df[target_col].nunique()} lithofacies classes**"
        )

        # Build a display DataFrame with canonical names
        rename = {raw: canon for raw, canon in raw_to_canon.items() if raw in raw_df.columns}
        disp = raw_df.rename(columns=rename).copy()
        if depth_col in disp.columns:
            disp = disp.rename(columns={depth_col: "DEPTH"})
        disp = disp.set_index("DEPTH")

        # Plot: log curves + lithofacies strip
        canon_logs = list(raw_to_canon.values())
        available = [t for t in TRACK_SPECS if t[0] in disp.columns]
        n_logs = len(available)
        n_total = n_logs + 1  # +1 for lithofacies strip

        width_ratios = [1.2] * n_logs + [0.5]
        fig_width = sum(width_ratios) * 1.0 + 1.5
        fig_height = 14

        fig, axes = plt.subplots(
            1, n_total,
            figsize=(fig_width, fig_height),
            sharey=True,
            gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05},
        )
        fig.patch.set_facecolor(PLOT_BG)

        depths = disp.index.values.astype(float)
        d_min, d_max = np.nanmin(depths), np.nanmax(depths)
        axes[0].set_ylim(d_max, d_min)
        axes[0].set_ylabel("Depth (ft)", fontsize=9, color=UT_CHARCOAL)

        for idx, (mnem, xmin, xmax, log_scale, label, unit) in enumerate(available):
            ax = axes[idx]
            ax.set_facecolor(PLOT_BG)
            curve = disp[mnem].values.astype(float)
            color = LOG_COLORS.get(mnem, "#555555")
            if log_scale:
                ax.set_xscale("log")
            ax.plot(curve, depths, color=color, linewidth=0.6, alpha=0.9)
            ax.set_xlim(xmin, xmax)
            ax.set_xlabel(f"{label} ({unit})", fontsize=8, color=color)
            ax.tick_params(axis="x", colors=color, labelsize=7)
            ax.tick_params(axis="y", colors=UT_CHARCOAL, labelsize=7)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.3, color=PLOT_GRID_COLOR)
            ax.grid(True, which="major", axis="x", linewidth=0.3, alpha=0.3, color=PLOT_GRID_COLOR)

        # Lithofacies strip
        ax_facies = axes[n_logs]
        ax_facies.set_facecolor(PLOT_BG)
        if target_col in raw_df.columns:
            facies_labels = raw_df[target_col].values
            facies_schema = load_facies_schema_cached()
            # Canonicalize for color lookup
            from electrofacies.preprocessing.standardize import canonicalize_facies_labels
            canon_labels = canonicalize_facies_labels(
                pd.Series(facies_labels), facies_schema
            ).values

            for i in range(len(depths) - 1):
                label = canon_labels[i]
                color = FACIES_COLORS.get(label, "#dddddd")
                ax_facies.axhspan(depths[i], depths[i + 1], color=color, alpha=0.9)

        ax_facies.set_xlim(0, 1)
        ax_facies.set_xticks([])
        ax_facies.set_xlabel("Lithofacies", fontsize=8, color=UT_CHARCOAL)
        ax_facies.xaxis.set_label_position("top")

        fig.suptitle("PDB03 — Training Well (Delaware Mountain Group)",
                     fontsize=11, color=UT_CHARCOAL, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Facies distribution table
        if target_col in raw_df.columns:
            from electrofacies.preprocessing.standardize import canonicalize_facies_labels as _canon
            facies_schema = load_facies_schema_cached()
            canon_series = _canon(raw_df[target_col], facies_schema)
            counts = canon_series.value_counts()
            dist_df = pd.DataFrame({
                "Lithofacies": [format_facies_name(f) for f in counts.index],
                "Count": counts.values,
                "%": [f"{v / len(raw_df) * 100:.1f}%" for v in counts.values],
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
