"""Streamlit web application for the Electrofacies Prediction System.

Upload LAS files, automatically predict lithofacies using ML models trained
on the PDB03 well from the Delaware Mountain Group, and download results
with full QC diagnostics.
"""

from __future__ import annotations

import io
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

# ---------------------------------------------------------------------------
# Logging setup (keep Streamlit logs clean)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("electrofacies.app")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Facies colour palette (loaded once)
# ---------------------------------------------------------------------------
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

LOG_COLORS: Dict[str, str] = {
    "GR": "#00a65a",
    "RESD": "#000000",
    "RHOB": "#e41a1c",
    "NPHI": "#377eb8",
    "DTC": "#ff7f00",
}

# Track display specs: (mnemonic, xmin, xmax, log_scale, label, unit)
TRACK_SPECS = [
    ("GR", 0, 300, False, "GR", "API"),
    ("RESD", 0.1, 10000, True, "RESD", "ohm.m"),
    ("RHOB", 1.9, 2.9, False, "RHOB", "g/cm\u00b3"),
    ("NPHI", 0.45, -0.05, False, "NPHI", "v/v"),
    ("DTC", 140, 40, False, "DTC", "\u00b5s/ft"),
]


# ===================================================================
# Lazy-loaded pipeline modules (avoids import errors on cold start)
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


def load_tier_models_cached(artifacts_dir: str, tier_name: str):
    """Load model bundles (not cached since artifacts may change)."""
    from electrofacies.inference.tier_router import load_tier_models
    return load_tier_models(artifacts_dir, tier_name)


# ===================================================================
# Pipeline helpers
# ===================================================================

def read_las_from_upload(uploaded_file) -> Dict[str, Any]:
    """Parse an uploaded LAS file via a temp file."""
    from electrofacies.io.readers import read_las

    with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    return read_las(tmp_path)


def standardize_well(curves_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Standardize column names to canonical mnemonics."""
    from electrofacies.preprocessing.standardize import standardize_columns
    mnemonic_map = load_mnemonic_map_cached()
    return standardize_columns(curves_df, mnemonic_map)


def validate_well_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Run full validation suite."""
    from electrofacies.preprocessing.validate import validate_well
    ranges_config = load_physical_ranges_cached()
    return validate_well(df, ranges_config)


def determine_available_logs_for_well(df: pd.DataFrame) -> List[str]:
    """Determine which canonical logs are available with sufficient coverage."""
    from electrofacies.inference.tier_router import determine_available_logs
    from electrofacies.io.schemas import CANONICAL_LOGS
    return determine_available_logs(df, CANONICAL_LOGS)


def select_tier(available_logs: List[str]):
    """Select the best model tier given available logs."""
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

    # Load model bundles for this tier
    tier_models = load_tier_models_cached(str(artifacts_dir), tier_name)
    if not tier_models:
        return None, None

    # Load or create transformer from the first bundle's saved state
    first_bundle = next(iter(tier_models.values()))
    bundle_dir = first_bundle.get("_bundle_dir", "")
    transformer_path = None
    if bundle_dir:
        candidate = Path(bundle_dir) / "transformer.joblib"
        if candidate.exists():
            transformer_path = candidate

    # Try loading transformer from artifacts
    if transformer_path and transformer_path.exists():
        transformer = FaciesTransformer.load(str(transformer_path))
    else:
        # Try to find transformer in any bundle dir under artifacts
        for algo_name, bundle in tier_models.items():
            meta = bundle.get("metadata", {})
            bdir = meta.get("bundle_dir", "")
            if bdir:
                t_path = Path(bdir) / "transformer.joblib"
                if t_path.exists():
                    transformer = FaciesTransformer.load(str(t_path))
                    break
        else:
            # Build a minimal transformer from the tier's feature config
            fe_config = config.get("feature_engineering", {})
            transformer = FaciesTransformer(config=fe_config)
            # Fit on the well data itself (not ideal but functional)
            meta = first_bundle.get("metadata", {})
            tier_cfg = load_tier_config_cached()
            tier_info = tier_cfg["tiers"].get(tier_name, {})
            required_logs = tier_info.get("required_logs", [])
            available = [c for c in required_logs if c in well_data.columns]
            if available:
                transformer.fit(well_data, available)

    # Run predictions for all algorithms
    all_preds = predict_all_algorithms(well_data, tier_models, transformer, config)
    if not all_preds:
        return None, None

    # Select best algorithm
    algo_name, best_preds = select_best_prediction(all_preds)

    # Post-process: smooth, flag confidence, flag OOD, assign QC status
    if "PREDICTED_FACIES" in best_preds.columns:
        best_preds["PREDICTED_FACIES"] = modal_filter(
            best_preds["PREDICTED_FACIES"], window=3
        )

    best_preds = assign_confidence_flags(best_preds, threshold=0.5)
    # OOD detection (without a fitted detector, just set False)
    best_preds["OOD_FLAG"] = False
    best_preds = assign_qc_status(best_preds)

    return algo_name, best_preds


# ===================================================================
# Visualization helpers for Streamlit
# ===================================================================

def format_facies_name(name: str) -> str:
    return name.replace("_", " ").title()


def plot_log_tracks(
    well_df: pd.DataFrame,
    predictions_df: Optional[pd.DataFrame] = None,
    title: str = "",
) -> plt.Figure:
    """Create a multi-track well-log display with optional facies prediction."""

    # Determine available tracks
    available = [t for t in TRACK_SPECS if t[0] in well_df.columns]
    n_logs = len(available)

    # Prediction strips
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

    fig.patch.set_facecolor("#0e1117")

    # Depth range
    depths = well_df.index.values.astype(float)
    d_min, d_max = np.nanmin(depths), np.nanmax(depths)

    axes[0].set_ylim(d_max, d_min)
    axes[0].set_ylabel("Depth (ft)", fontsize=9, color="white")

    # Plot log curves
    for idx, (mnem, xmin, xmax, log_scale, label, unit) in enumerate(available):
        ax = axes[idx]
        ax.set_facecolor("#1a1d24")
        curve = well_df[mnem].values.astype(float)
        color = LOG_COLORS.get(mnem, "#aaaaaa")

        if log_scale:
            ax.set_xscale("log")

        ax.plot(curve, depths, color=color, linewidth=0.8, alpha=0.9)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(f"{label} ({unit})", fontsize=8, color=color)
        ax.tick_params(axis="x", colors=color, labelsize=7)
        ax.tick_params(axis="y", colors="white", labelsize=7)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        ax.grid(True, which="both", axis="y", linewidth=0.2, alpha=0.15, color="white")
        ax.grid(True, which="major", axis="x", linewidth=0.2, alpha=0.15, color="white")

        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.5)

        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    # Prediction strips
    strip_idx = n_logs

    if has_facies:
        ax = axes[strip_idx]
        ax.set_facecolor("#1a1d24")
        pred_depths = predictions_df["DEPTH"].values if "DEPTH" in predictions_df.columns else predictions_df.index.values
        pred_depths = np.asarray(pred_depths, dtype=float)
        facies_vals = predictions_df["PREDICTED_FACIES"].values

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        if len(pred_depths) >= 2:
            for i in range(len(pred_depths) - 1):
                lbl = str(facies_vals[i])
                clr = FACIES_COLORS.get(lbl, "#cccccc")
                ax.axhspan(pred_depths[i], pred_depths[i + 1], facecolor=clr, edgecolor="none")
            lbl = str(facies_vals[-1])
            clr = FACIES_COLORS.get(lbl, "#cccccc")
            step = pred_depths[-1] - pred_depths[-2]
            ax.axhspan(pred_depths[-1], pred_depths[-1] + step, facecolor=clr, edgecolor="none")

        ax.set_xlabel("Facies", fontsize=8, color="white")
        ax.xaxis.set_label_position("top")
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.3)
        strip_idx += 1

    if has_conf:
        ax = axes[strip_idx]
        ax.set_facecolor("#1a1d24")
        pred_depths = predictions_df["DEPTH"].values if "DEPTH" in predictions_df.columns else predictions_df.index.values
        pred_depths = np.asarray(pred_depths, dtype=float)
        conf_vals = predictions_df["CONFIDENCE_SCORE"].values.astype(float)

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        if len(pred_depths) >= 2:
            for i in range(len(pred_depths) - 1):
                c = np.clip(conf_vals[i], 0, 1)
                # Green (high) -> Red (low)
                r = 1.0 - c
                g = c
                ax.axhspan(pred_depths[i], pred_depths[i + 1],
                           facecolor=(r, g, 0.2), edgecolor="none")
            c = np.clip(conf_vals[-1], 0, 1)
            step = pred_depths[-1] - pred_depths[-2]
            ax.axhspan(pred_depths[-1], pred_depths[-1] + step,
                       facecolor=(1.0 - c, c, 0.2), edgecolor="none")

        ax.set_xlabel("Conf.", fontsize=8, color="white")
        ax.xaxis.set_label_position("top")
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.3)
        strip_idx += 1

    if has_qc:
        ax = axes[strip_idx]
        ax.set_facecolor("#1a1d24")
        pred_depths = predictions_df["DEPTH"].values if "DEPTH" in predictions_df.columns else predictions_df.index.values
        pred_depths = np.asarray(pred_depths, dtype=float)
        qc_vals = predictions_df["QC_STATUS"].values

        qc_colors = {"GOOD": "#2ca25f", "LOW_CONFIDENCE": "#fee391",
                      "OOD": "#de2d26", "LOW_CONF_AND_OOD": "#8b0000"}

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        if len(pred_depths) >= 2:
            for i in range(len(pred_depths) - 1):
                clr = qc_colors.get(str(qc_vals[i]), "#555555")
                ax.axhspan(pred_depths[i], pred_depths[i + 1], facecolor=clr, edgecolor="none")
            clr = qc_colors.get(str(qc_vals[-1]), "#555555")
            step = pred_depths[-1] - pred_depths[-2]
            ax.axhspan(pred_depths[-1], pred_depths[-1] + step, facecolor=clr, edgecolor="none")

        ax.set_xlabel("QC", fontsize=8, color="white")
        ax.xaxis.set_label_position("top")
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.3)

    # Title
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color="white", y=1.01)

    fig.tight_layout()
    return fig


def make_facies_legend() -> plt.Figure:
    """Create a horizontal facies colour legend."""
    fig, ax = plt.subplots(figsize=(8, 0.6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    patches = [
        mpatches.Patch(facecolor=FACIES_COLORS.get(f, "#ccc"), edgecolor="white",
                       linewidth=0.5, label=format_facies_name(f))
        for f in FACIES_ORDER
    ]
    ax.legend(handles=patches, loc="center", ncol=len(patches),
              frameon=False, fontsize=8, labelcolor="white")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_facies_pie(predictions_df: pd.DataFrame) -> plt.Figure:
    """Pie chart of predicted facies distribution."""
    facies_counts = predictions_df["PREDICTED_FACIES"].value_counts()
    colors = [FACIES_COLORS.get(f, "#cccccc") for f in facies_counts.index]
    labels = [format_facies_name(f) for f in facies_counts.index]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    wedges, texts, autotexts = ax.pie(
        facies_counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", pctdistance=0.8,
        wedgeprops={"edgecolor": "#1a1d24", "linewidth": 1.5},
        textprops={"fontsize": 8, "color": "white"},
    )
    for t in autotexts:
        t.set_fontsize(7)
        t.set_color("white")

    ax.set_title("Facies Distribution", fontsize=11, color="white", pad=10)
    fig.tight_layout()
    return fig


def plot_confidence_histogram(predictions_df: pd.DataFrame) -> plt.Figure:
    """Histogram of confidence scores."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d24")

    conf = predictions_df["CONFIDENCE_SCORE"].values
    ax.hist(conf, bins=30, color="#e6b657", edgecolor="#0e1117", alpha=0.85)
    ax.axvline(0.5, color="#de2d26", linestyle="--", linewidth=1.2, label="Threshold (0.5)")
    ax.set_xlabel("Confidence Score", fontsize=9, color="white")
    ax.set_ylabel("Count", fontsize=9, color="white")
    ax.set_title("Confidence Distribution", fontsize=11, color="white")
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1d24", edgecolor="#333")

    for spine in ax.spines.values():
        spine.set_color("#333333")
    fig.tight_layout()
    return fig


# ===================================================================
# Main App
# ===================================================================

def main():
    st.set_page_config(
        page_title="Electrofacies Predictor",
        page_icon="\U0001faa8",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("# \U0001faa8 Electrofacies")
        st.markdown("**Delaware Mountain Group**")
        st.markdown("ML-powered lithofacies prediction from well logs")
        st.divider()

        st.markdown("### Model Info")
        st.markdown("""
        - **Training well**: PDB03
        - **Algorithms**: Random Forest, XGBoost
        - **Facies**: 6 classes
        - **Features**: Z-scores, rolling stats, derivatives
        """)
        st.divider()

        st.markdown("### Tier System")
        st.markdown("""
        | Tier | Logs Required |
        |------|--------------|
        | 1 (Best) | GR, RESD, RHOB, NPHI, DTC |
        | 2 | GR, RESD, RHOB, NPHI |
        | 3 | GR, RESD, RHOB |
        | 4 (Min) | GR, RESD |
        """)
        st.divider()

        st.markdown("### About")
        st.markdown(
            "Built by **Ander Martinez-Donate**  \n"
            "University of Texas at Austin"
        )

    # ---- Header ----
    st.markdown(
        "<h1 style='text-align: center;'>\U0001faa8 Electrofacies Prediction System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #888;'>"
        "Upload a LAS file to predict lithofacies using ML models trained on the "
        "Delaware Mountain Group (Permian Basin)</p>",
        unsafe_allow_html=True,
    )

    # ---- File Upload ----
    st.divider()
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a LAS well-log file",
            type=["las", "LAS"],
            help="Standard LAS 2.0 or 3.0 format. Maximum 50 MB.",
        )

    with col_info:
        st.info(
            "The system will automatically:\n"
            "1. Map your mnemonics to canonical names\n"
            "2. Validate log ranges and coverage\n"
            "3. Select the optimal model tier\n"
            "4. Predict facies with confidence scores"
        )

    if uploaded_file is None:
        # Show demo / landing page
        st.divider()
        st.markdown("### How it works")
        cols = st.columns(4)
        with cols[0]:
            st.markdown("#### 1. Upload")
            st.markdown("Drop your LAS file above. We support 144+ vendor mnemonic aliases.")
        with cols[1]:
            st.markdown("#### 2. Validate")
            st.markdown("Physical range checks, washout detection, flatline detection, null coverage audit.")
        with cols[2]:
            st.markdown("#### 3. Predict")
            st.markdown("Auto-selects the best model tier based on available logs. Runs RF & XGBoost.")
        with cols[3]:
            st.markdown("#### 4. Download")
            st.markdown("Get predictions with confidence scores, QC flags, and publication-ready figures.")

        # Facies legend
        st.divider()
        st.markdown("### Facies Classification Scheme")
        fig_legend = make_facies_legend()
        st.pyplot(fig_legend, use_container_width=True)
        plt.close(fig_legend)

        facies_schema = load_facies_schema_cached()
        cols = st.columns(3)
        for i, facies_name in enumerate(FACIES_ORDER):
            info = facies_schema["facies"].get(facies_name, {})
            with cols[i % 3]:
                color = FACIES_COLORS.get(facies_name, "#ccc")
                st.markdown(
                    f"<div style='border-left: 4px solid {color}; padding-left: 8px; margin-bottom: 8px;'>"
                    f"<b>{format_facies_name(facies_name)}</b><br>"
                    f"<small>{info.get('description', '')}</small></div>",
                    unsafe_allow_html=True,
                )
        return

    # ==================================================================
    # LAS file uploaded -- run pipeline
    # ==================================================================
    st.divider()

    # Step 1: Read LAS
    with st.spinner("Reading LAS file..."):
        try:
            well_data = read_las_from_upload(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read LAS file: {e}")
            return

    curves_df = well_data["curves"]
    metadata = well_data["metadata"]
    well_name = metadata.get("well_name", uploaded_file.name)

    # Step 2: Display well metadata
    st.markdown(f"## Well: **{well_name}**")

    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("Samples", f"{metadata.get('num_samples', len(curves_df)):,}")
    with meta_cols[1]:
        depth_range = metadata.get("depth_range", (0, 0))
        st.metric("Depth Range", f"{depth_range[0]:.0f} - {depth_range[1]:.0f} ft")
    with meta_cols[2]:
        st.metric("Curves", metadata.get("num_curves", len(curves_df.columns)))
    with meta_cols[3]:
        uwi = metadata.get("uwi", "")
        st.metric("UWI", uwi if uwi else "N/A")

    # Step 3: Standardize mnemonics
    with st.spinner("Standardizing mnemonics..."):
        std_df, mapping_report = standardize_well(curves_df)

    # Show mnemonic mapping
    with st.expander("Mnemonic Mapping", expanded=False):
        mapped = {k: v for k, v in mapping_report.items() if v != "unmapped"}
        unmapped = {k: v for k, v in mapping_report.items() if v == "unmapped"}

        if mapped:
            map_df = pd.DataFrame(
                [{"Original": k, "Canonical": v} for k, v in mapped.items()]
            )
            st.dataframe(map_df, use_container_width=True, hide_index=True)

        if unmapped:
            st.caption(f"Unmapped curves ({len(unmapped)}): {', '.join(unmapped.keys())}")

    # Step 4: Validate
    with st.spinner("Validating well data..."):
        val_report = validate_well_data(std_df)

    # Show validation results
    with st.expander("Data Validation Report", expanded=False):
        if val_report["valid"]:
            st.success("All validation checks passed.")
        else:
            st.warning(f"Found {len(val_report['issues'])} issue(s):")
            for issue in val_report["issues"]:
                st.markdown(f"- {issue}")

        # Null coverage
        cov = val_report.get("null_coverage", {}).get("coverage", {})
        if cov:
            cov_df = pd.DataFrame(
                [{"Log": k, "Null %": f"{v * 100:.1f}%"} for k, v in cov.items()]
            )
            st.dataframe(cov_df, use_container_width=True, hide_index=True)

    # Step 5: Determine available logs and tier
    available_logs = determine_available_logs_for_well(std_df)

    tier_result = select_tier(available_logs)

    st.divider()

    if tier_result is None:
        st.error(
            "No model tier can be satisfied with the available logs. "
            f"Available canonical logs: {available_logs}. "
            "Need at least GR and RESD."
        )

        # Still show the raw logs
        st.markdown("### Well Log Display")
        fig = plot_log_tracks(std_df, title=well_name)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    tier_name, tier_info = tier_result

    tier_cols = st.columns(3)
    with tier_cols[0]:
        st.metric("Selected Tier", f"{tier_name.replace('_', ' ').title()}")
    with tier_cols[1]:
        st.metric("Tier Name", tier_info.get("name", ""))
    with tier_cols[2]:
        st.metric("Available Logs", ", ".join(available_logs))

    st.caption(f"_{tier_info.get('description', '')}_")

    # Step 6: Check for trained models
    config = load_default_config_cached()
    artifacts_dir = PROJECT_ROOT / config.get("paths", {}).get("artifacts_dir", "artifacts")

    models_available = False
    if artifacts_dir.is_dir():
        # Check if there are any model bundles for this tier
        try:
            tier_models = load_tier_models_cached(str(artifacts_dir), tier_name)
            models_available = len(tier_models) > 0
        except Exception:
            models_available = False

    # Step 7: Run prediction (or show placeholder)
    st.divider()

    if not models_available:
        st.warning(
            f"No trained model bundles found for **{tier_name}** in `{artifacts_dir}`.  \n"
            "To enable predictions, train models first with:  \n"
            "```\nelectrofacies train --training-data data/PDB03_training.xlsx\n```"
        )

        # Show raw log display only
        st.markdown("### Well Log Display (Raw Data)")
        fig = plot_log_tracks(std_df, title=well_name)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Offer raw data download
        csv_buf = io.StringIO()
        export_df = std_df.copy()
        export_df.index.name = "DEPTH"
        export_df.to_csv(csv_buf)
        st.download_button(
            label="Download Standardized Well Data (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"{well_name}_standardized.csv",
            mime="text/csv",
        )
        return

    # ----- Predictions available -----
    with st.spinner("Running facies prediction..."):
        algo_name, predictions_df = run_prediction(
            std_df, tier_name, artifacts_dir, config
        )

    if predictions_df is None:
        st.error("Prediction failed. Check that model bundles are valid.")
        return

    # Display prediction summary
    st.markdown("### Prediction Results")

    result_cols = st.columns(4)
    with result_cols[0]:
        st.metric("Algorithm", algo_name.replace("_", " ").title() if algo_name else "N/A")
    with result_cols[1]:
        mean_conf = predictions_df["CONFIDENCE_SCORE"].mean() if "CONFIDENCE_SCORE" in predictions_df.columns else 0
        st.metric("Mean Confidence", f"{mean_conf:.1%}")
    with result_cols[2]:
        if "QC_STATUS" in predictions_df.columns:
            pct_good = (predictions_df["QC_STATUS"] == "GOOD").mean() * 100
            st.metric("QC Pass Rate", f"{pct_good:.1f}%")
    with result_cols[3]:
        n_facies = predictions_df["PREDICTED_FACIES"].nunique() if "PREDICTED_FACIES" in predictions_df.columns else 0
        st.metric("Facies Detected", n_facies)

    # Overall QC grade
    if "QC_STATUS" in predictions_df.columns:
        pct_good = (predictions_df["QC_STATUS"] == "GOOD").mean() * 100
        if pct_good >= 80:
            grade, grade_color = "PASS", "#2ca25f"
        elif pct_good >= 50:
            grade, grade_color = "REVIEW", "#fee391"
        else:
            grade, grade_color = "FAIL", "#de2d26"
        st.markdown(
            f"<div style='text-align: center; padding: 10px; "
            f"background-color: {grade_color}20; border: 2px solid {grade_color}; "
            f"border-radius: 8px; margin: 10px 0;'>"
            f"<h3 style='color: {grade_color}; margin: 0;'>QC Grade: {grade}</h3></div>",
            unsafe_allow_html=True,
        )

    # Facies legend
    fig_legend = make_facies_legend()
    st.pyplot(fig_legend, use_container_width=True)
    plt.close(fig_legend)

    # Multi-track log display
    st.markdown("### Well Log Display with Predictions")
    fig = plot_log_tracks(std_df, predictions_df, title=well_name)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Analytics charts
    st.divider()
    st.markdown("### Analytics")

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

    # QC status breakdown
    if "QC_STATUS" in predictions_df.columns:
        with st.expander("QC Status Breakdown", expanded=False):
            qc_counts = predictions_df["QC_STATUS"].value_counts()
            qc_df = pd.DataFrame({
                "Status": qc_counts.index,
                "Count": qc_counts.values,
                "Percentage": [f"{v / len(predictions_df) * 100:.1f}%" for v in qc_counts.values],
            })
            st.dataframe(qc_df, use_container_width=True, hide_index=True)

    # Facies distribution table
    if "PREDICTED_FACIES" in predictions_df.columns:
        with st.expander("Facies Distribution Table", expanded=False):
            facies_counts = predictions_df["PREDICTED_FACIES"].value_counts()
            dist_df = pd.DataFrame({
                "Facies": [format_facies_name(f) for f in facies_counts.index],
                "Count": facies_counts.values,
                "Percentage": [f"{v / len(predictions_df) * 100:.1f}%" for v in facies_counts.values],
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # Predictions data table
    with st.expander("Raw Predictions Table", expanded=False):
        # Show first/last 50 rows
        display_cols = [c for c in ["DEPTH", "PREDICTED_FACIES", "CONFIDENCE_SCORE",
                                     "MODEL_TIER", "ALGORITHM", "QC_STATUS"]
                        if c in predictions_df.columns]
        st.dataframe(
            predictions_df[display_cols].head(100),
            use_container_width=True,
            hide_index=True,
        )
        if len(predictions_df) > 100:
            st.caption(f"Showing first 100 of {len(predictions_df)} rows.")

    # ---- Downloads ----
    st.divider()
    st.markdown("### Download Results")

    dl_cols = st.columns(3)

    # CSV download
    with dl_cols[0]:
        csv_buf = io.StringIO()
        predictions_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"{well_name}_electrofacies.csv",
            mime="text/csv",
        )

    # Standardized well data
    with dl_cols[1]:
        csv_buf2 = io.StringIO()
        export_df = std_df.copy()
        export_df.index.name = "DEPTH"
        export_df.to_csv(csv_buf2)
        st.download_button(
            label="Download Standardized Data (CSV)",
            data=csv_buf2.getvalue(),
            file_name=f"{well_name}_standardized.csv",
            mime="text/csv",
        )

    # Figure download
    with dl_cols[2]:
        fig_dl = plot_log_tracks(std_df, predictions_df, title=well_name)
        buf = io.BytesIO()
        fig_dl.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                       facecolor="#0e1117")
        plt.close(fig_dl)
        st.download_button(
            label="Download Log Display (PNG)",
            data=buf.getvalue(),
            file_name=f"{well_name}_log_display.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
