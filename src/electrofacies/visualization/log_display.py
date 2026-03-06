"""Well-log display figures for the electrofacies pipeline.

Provides multi-track log displays with facies predictions, confidence
strips, QC flags, and side-by-side comparison of true vs. predicted facies.
All figures are publication-quality and save to PNG (optionally TIFF).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FT_TO_M: float = 0.3048

FACIES_COLORS: Dict[str, str] = {
    "massive_sandstone": "#f3e6b3",
    "structured_sandstone": "#e6b657",
    "sandy_siltstone": "#a7663a",
    "siltstone": "#505660",
    "calciturbidite": "#2ca25f",
    "clast_supported_conglomerate": "#7e3fa0",
    "missing_strata": "#ffffff",
}

LOG_COLORS: Dict[str, str] = {
    "GR": "#00a65a",
    "RESD": "#000000",
    "RHOB": "#e41a1c",
    "NPHI": "#377eb8",
    "DTC": "#ff7f00",
}

# Track definitions: (mnemonic, x_min, x_max, log_scale, invert, label, unit)
_TRACK_DEFS: List[Tuple[str, float, float, bool, bool, str, str]] = [
    ("GR", 0, 300, False, False, "GR", "API"),
    ("RESD", 0.1, 10000, True, False, "RESD", "ohm.m"),
    ("RHOB", 1.9, 2.9, False, False, "RHOB", "g/cm\u00b3"),
    ("NPHI", 0.45, -0.05, False, True, "NPHI", "v/v"),
    ("DTC", 140, 40, False, True, "DTC", "\u00b5s/ft"),
]

# QC status colours
_QC_COLORS: Dict[str, str] = {
    "GOOD": "#2ca25f",
    "LOW_CONF": "#fee391",
    "OOD": "#de2d26",
}
_QC_DEFAULT_COLOR: str = "#ffffff"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pub_rcparams() -> Dict[str, Any]:
    """Return a dict of rcParams suitable for publication figures."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.4,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }


def _format_facies_name(name: str) -> str:
    """Convert a snake_case facies name to a human-readable label."""
    return name.replace("_", " ").title()


def _save_figure(fig: plt.Figure, output_path: str | Path, dpi: int) -> None:
    """Save figure to PNG and optionally TIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(output_path), dpi=dpi, facecolor="white")
    logger.info("Saved figure to %s", output_path)

    # If the caller requested TIFF, save that format too
    if output_path.suffix.lower() in (".tiff", ".tif"):
        # Already saved as the primary format
        pass
    else:
        # Also save a TIFF copy alongside the PNG
        tiff_path = output_path.with_suffix(".tiff")
        # Only save TIFF if the user explicitly provided a .tiff path;
        # otherwise just save the requested format.  We check whether the
        # original path ended in .png — if so, skip auto-TIFF.
        if output_path.suffix.lower() != ".png":
            fig.savefig(str(tiff_path), dpi=dpi, facecolor="white")
            logger.info("Saved TIFF copy to %s", tiff_path)


def _paint_facies_strip(
    ax: plt.Axes,
    depths: np.ndarray,
    facies_labels: np.ndarray,
    facies_colors: Dict[str, str],
) -> None:
    """Fill a thin axis with horizontal colour bars for each facies interval."""
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    if len(depths) < 2:
        return

    for i in range(len(depths) - 1):
        label = str(facies_labels[i])
        color = facies_colors.get(label, "#cccccc")
        ax.axhspan(depths[i], depths[i + 1], facecolor=color, edgecolor="none")

    # Handle last sample
    if len(depths) >= 2:
        label = str(facies_labels[-1])
        color = facies_colors.get(label, "#cccccc")
        step = depths[-1] - depths[-2]
        ax.axhspan(depths[-1], depths[-1] + step, facecolor=color, edgecolor="none")


def _paint_confidence_strip(
    ax: plt.Axes,
    depths: np.ndarray,
    confidence: np.ndarray,
) -> None:
    """Fill an axis with grayscale bars: dark = high confidence, light = low."""
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    if len(depths) < 2:
        return

    for i in range(len(depths) - 1):
        # Clamp confidence to [0, 1]
        conf = np.clip(confidence[i], 0.0, 1.0)
        # Dark (0.0 gray) = high confidence, light (0.85 gray) = low
        gray_val = 1.0 - conf * 0.85
        ax.axhspan(
            depths[i],
            depths[i + 1],
            facecolor=str(gray_val),
            edgecolor="none",
        )

    # Last sample
    if len(depths) >= 2:
        conf = np.clip(confidence[-1], 0.0, 1.0)
        gray_val = 1.0 - conf * 0.85
        step = depths[-1] - depths[-2]
        ax.axhspan(
            depths[-1],
            depths[-1] + step,
            facecolor=str(gray_val),
            edgecolor="none",
        )


def _paint_qc_strip(
    ax: plt.Axes,
    depths: np.ndarray,
    qc_status: np.ndarray,
) -> None:
    """Fill an axis with QC status colours."""
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    if len(depths) < 2:
        return

    for i in range(len(depths) - 1):
        status = str(qc_status[i]).upper()
        color = _QC_COLORS.get(status, _QC_DEFAULT_COLOR)
        ax.axhspan(depths[i], depths[i + 1], facecolor=color, edgecolor="none")

    # Last sample
    if len(depths) >= 2:
        status = str(qc_status[-1]).upper()
        color = _QC_COLORS.get(status, _QC_DEFAULT_COLOR)
        step = depths[-1] - depths[-2]
        ax.axhspan(
            depths[-1],
            depths[-1] + step,
            facecolor=color,
            edgecolor="none",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_well_predictions(
    well_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: str | Path,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a multi-track well log display with facies predictions.

    Parameters
    ----------
    well_df : pd.DataFrame
        Log curves indexed by depth (feet).  Expected columns include any
        subset of ``GR``, ``RESD``, ``RHOB``, ``NPHI``, ``DTC``.
    predictions_df : pd.DataFrame
        Predictions indexed by depth (feet).  Must contain ``FACIES``.
        Optional columns: ``CONFIDENCE`` (float 0-1), ``QC_STATUS`` (str).
    output_path : str or Path
        Destination file path (PNG or TIFF).
    title : str, optional
        Figure title.  If *None*, no title is rendered.
    dpi : int, optional
        Resolution for the saved image (default 300).

    Returns
    -------
    tuple[Figure, ndarray]
        Matplotlib Figure and array of Axes for optional further customization.
    """
    with plt.rc_context(_pub_rcparams()):
        # -----------------------------------------------------------------
        # Determine which log tracks to include
        # -----------------------------------------------------------------
        available_tracks = [
            t for t in _TRACK_DEFS if t[0] in well_df.columns
        ]
        if not available_tracks:
            logger.warning(
                "No recognized log curves in well_df columns: %s",
                list(well_df.columns),
            )

        n_log_tracks = len(available_tracks)

        # Prediction-based tracks
        has_facies = "FACIES" in predictions_df.columns
        has_confidence = "CONFIDENCE" in predictions_df.columns
        has_qc = "QC_STATUS" in predictions_df.columns

        n_extra = sum([has_facies, has_confidence, has_qc])
        n_total = n_log_tracks + n_extra

        if n_total == 0:
            logger.error(
                "Nothing to plot: no log curves and no prediction columns."
            )
            fig, ax = plt.subplots()
            return fig, np.array([ax])

        # -----------------------------------------------------------------
        # Figure layout
        # -----------------------------------------------------------------
        # Log tracks get width 1.2; strip tracks get width 0.5
        width_ratios = [1.2] * n_log_tracks + [0.5] * n_extra
        fig_width = sum(width_ratios) * 0.9 + 1.0  # padding for labels

        # Depth range
        depths_ft = well_df.index.values.astype(float)
        pred_depths_ft = predictions_df.index.values.astype(float)
        all_depths = np.concatenate([depths_ft, pred_depths_ft])
        depth_min = np.nanmin(all_depths)
        depth_max = np.nanmax(all_depths)
        depth_min_m = depth_min * FT_TO_M
        depth_max_m = depth_max * FT_TO_M

        fig_height = max(8, (depth_max - depth_min) / 150)
        fig_height = min(fig_height, 24)

        fig, axes = plt.subplots(
            1,
            n_total,
            figsize=(fig_width, fig_height),
            sharey=True,
            gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05},
        )
        if n_total == 1:
            axes = np.array([axes])

        # -----------------------------------------------------------------
        # Y-axis (depth)
        # -----------------------------------------------------------------
        # Depth increases downward
        axes[0].set_ylim(depth_max_m, depth_min_m)
        axes[0].set_ylabel("Depth (m)", fontsize=8)

        # -----------------------------------------------------------------
        # Plot log tracks
        # -----------------------------------------------------------------
        for idx, (mnem, xmin, xmax, log_scale, invert, label, unit) in enumerate(
            available_tracks
        ):
            ax = axes[idx]
            curve = well_df[mnem].values.astype(float)
            depth_m = depths_ft * FT_TO_M

            color = LOG_COLORS.get(mnem, "#333333")

            if log_scale:
                ax.set_xscale("log")

            ax.plot(curve, depth_m, color=color, linewidth=0.6, alpha=0.9)

            # Set x limits (handle inverted axes via xmin > xmax)
            if invert:
                ax.set_xlim(xmin, xmax)
            else:
                ax.set_xlim(xmin, xmax)

            ax.set_xlabel(f"{label} ({unit})", fontsize=7, color=color)
            ax.tick_params(axis="x", colors=color, labelsize=6)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()

            # Subtle grid
            ax.grid(True, which="both", axis="y", linewidth=0.2, alpha=0.3)
            ax.grid(True, which="major", axis="x", linewidth=0.2, alpha=0.3)

            # Clean spines
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["top"].set_linewidth(0.5)

            # Turn off y-tick labels on all but the first track
            if idx > 0:
                ax.tick_params(axis="y", labelleft=False)

        # -----------------------------------------------------------------
        # Plot prediction strips
        # -----------------------------------------------------------------
        strip_idx = n_log_tracks
        pred_depth_m = pred_depths_ft * FT_TO_M

        if has_facies:
            ax = axes[strip_idx]
            _paint_facies_strip(
                ax,
                pred_depth_m,
                predictions_df["FACIES"].values,
                FACIES_COLORS,
            )
            ax.set_xlabel("Facies", fontsize=7)
            ax.xaxis.set_label_position("top")
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.3)
            ax.spines["top"].set_linewidth(0.3)
            strip_idx += 1

        if has_confidence:
            ax = axes[strip_idx]
            _paint_confidence_strip(
                ax,
                pred_depth_m,
                predictions_df["CONFIDENCE"].values.astype(float),
            )
            ax.set_xlabel("Conf.", fontsize=7)
            ax.xaxis.set_label_position("top")
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.3)
            ax.spines["top"].set_linewidth(0.3)
            strip_idx += 1

        if has_qc:
            ax = axes[strip_idx]
            _paint_qc_strip(
                ax,
                pred_depth_m,
                predictions_df["QC_STATUS"].values,
            )
            ax.set_xlabel("QC", fontsize=7)
            ax.xaxis.set_label_position("top")
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.3)
            ax.spines["top"].set_linewidth(0.3)
            strip_idx += 1

        # -----------------------------------------------------------------
        # Secondary y-axis in feet on the rightmost track
        # -----------------------------------------------------------------
        ax_right = axes[-1].twinx()
        ax_right.set_ylim(depth_max, depth_min)
        ax_right.set_ylabel("Depth (ft)", fontsize=8)
        ax_right.spines["left"].set_visible(False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)

        # -----------------------------------------------------------------
        # Facies legend (below the figure)
        # -----------------------------------------------------------------
        if has_facies:
            unique_facies = sorted(
                predictions_df["FACIES"].dropna().unique(),
                key=lambda f: list(FACIES_COLORS.keys()).index(f)
                if f in FACIES_COLORS
                else 999,
            )
            legend_patches = [
                mpatches.Patch(
                    facecolor=FACIES_COLORS.get(f, "#cccccc"),
                    edgecolor="black",
                    linewidth=0.4,
                    label=_format_facies_name(f),
                )
                for f in unique_facies
            ]
            fig.legend(
                handles=legend_patches,
                loc="lower center",
                ncol=min(len(legend_patches), 4),
                frameon=True,
                edgecolor="#cccccc",
                fontsize=7,
                bbox_to_anchor=(0.5, -0.02),
            )

        # -----------------------------------------------------------------
        # Title and layout
        # -----------------------------------------------------------------
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

        fig.tight_layout()
        _save_figure(fig, output_path, dpi)

    return fig, axes


def plot_well_comparison(
    well_df: pd.DataFrame,
    true_facies_col: str,
    predicted_facies_col: str,
    output_path: str | Path,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """Side-by-side display of logs, true facies, predicted facies, and match strip.

    Parameters
    ----------
    well_df : pd.DataFrame
        DataFrame indexed by depth (feet) containing log curves, a column
        for true facies labels (``true_facies_col``), and a column for
        predicted facies labels (``predicted_facies_col``).
    true_facies_col : str
        Column name holding the ground-truth facies labels.
    predicted_facies_col : str
        Column name holding the predicted facies labels.
    output_path : str or Path
        Destination file path.
    dpi : int, optional
        Resolution for the saved image (default 300).

    Returns
    -------
    tuple[Figure, ndarray]
        Matplotlib Figure and array of Axes.
    """
    with plt.rc_context(_pub_rcparams()):
        # -----------------------------------------------------------------
        # Determine available log tracks
        # -----------------------------------------------------------------
        available_tracks = [
            t for t in _TRACK_DEFS if t[0] in well_df.columns
        ]
        n_log_tracks = len(available_tracks)

        # 3 extra strips: true facies, predicted facies, match/mismatch
        n_strips = 3
        n_total = n_log_tracks + n_strips

        width_ratios = [1.2] * n_log_tracks + [0.5] * n_strips

        depths_ft = well_df.index.values.astype(float)
        depth_min = np.nanmin(depths_ft)
        depth_max = np.nanmax(depths_ft)
        depth_min_m = depth_min * FT_TO_M
        depth_max_m = depth_max * FT_TO_M

        fig_width = sum(width_ratios) * 0.9 + 1.0
        fig_height = max(8, (depth_max - depth_min) / 150)
        fig_height = min(fig_height, 24)

        fig, axes = plt.subplots(
            1,
            n_total,
            figsize=(fig_width, fig_height),
            sharey=True,
            gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05},
        )
        if n_total == 1:
            axes = np.array([axes])

        # Y-axis (depth in metres)
        axes[0].set_ylim(depth_max_m, depth_min_m)
        axes[0].set_ylabel("Depth (m)", fontsize=8)

        depth_m = depths_ft * FT_TO_M

        # -----------------------------------------------------------------
        # Plot log tracks
        # -----------------------------------------------------------------
        for idx, (mnem, xmin, xmax, log_scale, invert, label, unit) in enumerate(
            available_tracks
        ):
            ax = axes[idx]
            curve = well_df[mnem].values.astype(float)
            color = LOG_COLORS.get(mnem, "#333333")

            if log_scale:
                ax.set_xscale("log")

            ax.plot(curve, depth_m, color=color, linewidth=0.6, alpha=0.9)

            if invert:
                ax.set_xlim(xmin, xmax)
            else:
                ax.set_xlim(xmin, xmax)

            ax.set_xlabel(f"{label} ({unit})", fontsize=7, color=color)
            ax.tick_params(axis="x", colors=color, labelsize=6)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()

            ax.grid(True, which="both", axis="y", linewidth=0.2, alpha=0.3)
            ax.grid(True, which="major", axis="x", linewidth=0.2, alpha=0.3)

            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["top"].set_linewidth(0.5)

            if idx > 0:
                ax.tick_params(axis="y", labelleft=False)

        # -----------------------------------------------------------------
        # True facies strip
        # -----------------------------------------------------------------
        ax_true = axes[n_log_tracks]
        true_labels = well_df[true_facies_col].values
        _paint_facies_strip(ax_true, depth_m, true_labels, FACIES_COLORS)
        ax_true.set_xlabel("True", fontsize=7)
        ax_true.xaxis.set_label_position("top")
        for spine in ("bottom", "right"):
            ax_true.spines[spine].set_visible(False)
        ax_true.spines["left"].set_linewidth(0.3)
        ax_true.spines["top"].set_linewidth(0.3)

        # -----------------------------------------------------------------
        # Predicted facies strip
        # -----------------------------------------------------------------
        ax_pred = axes[n_log_tracks + 1]
        pred_labels = well_df[predicted_facies_col].values
        _paint_facies_strip(ax_pred, depth_m, pred_labels, FACIES_COLORS)
        ax_pred.set_xlabel("Predicted", fontsize=7)
        ax_pred.xaxis.set_label_position("top")
        for spine in ("bottom", "right"):
            ax_pred.spines[spine].set_visible(False)
        ax_pred.spines["left"].set_linewidth(0.3)
        ax_pred.spines["top"].set_linewidth(0.3)

        # -----------------------------------------------------------------
        # Match / mismatch strip
        # -----------------------------------------------------------------
        ax_match = axes[n_log_tracks + 2]
        ax_match.set_xlim(0, 1)
        ax_match.set_xticks([])

        match_colors = {True: "#2ca25f", False: "#de2d26"}  # green / red
        matches = np.array(
            [
                str(t) == str(p)
                for t, p in zip(true_labels, pred_labels)
            ]
        )

        if len(depth_m) >= 2:
            for i in range(len(depth_m) - 1):
                color = match_colors[bool(matches[i])]
                ax_match.axhspan(
                    depth_m[i], depth_m[i + 1],
                    facecolor=color, edgecolor="none",
                )
            # Last sample
            step = depth_m[-1] - depth_m[-2]
            color = match_colors[bool(matches[-1])]
            ax_match.axhspan(
                depth_m[-1], depth_m[-1] + step,
                facecolor=color, edgecolor="none",
            )

        ax_match.set_xlabel("Match", fontsize=7)
        ax_match.xaxis.set_label_position("top")
        for spine in ("bottom", "right"):
            ax_match.spines[spine].set_visible(False)
        ax_match.spines["left"].set_linewidth(0.3)
        ax_match.spines["top"].set_linewidth(0.3)

        # -----------------------------------------------------------------
        # Secondary y-axis in feet
        # -----------------------------------------------------------------
        ax_right = axes[-1].twinx()
        ax_right.set_ylim(depth_max, depth_min)
        ax_right.set_ylabel("Depth (ft)", fontsize=8)
        ax_right.spines["left"].set_visible(False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)

        # -----------------------------------------------------------------
        # Legend
        # -----------------------------------------------------------------
        all_labels = np.concatenate([true_labels, pred_labels])
        unique_facies = sorted(
            set(str(f) for f in all_labels if pd.notna(f)),
            key=lambda f: list(FACIES_COLORS.keys()).index(f)
            if f in FACIES_COLORS
            else 999,
        )
        legend_patches = [
            mpatches.Patch(
                facecolor=FACIES_COLORS.get(f, "#cccccc"),
                edgecolor="black",
                linewidth=0.4,
                label=_format_facies_name(f),
            )
            for f in unique_facies
        ]
        # Add match/mismatch entries
        legend_patches.append(
            mpatches.Patch(
                facecolor="#2ca25f",
                edgecolor="black",
                linewidth=0.4,
                label="Correct",
            )
        )
        legend_patches.append(
            mpatches.Patch(
                facecolor="#de2d26",
                edgecolor="black",
                linewidth=0.4,
                label="Mismatch",
            )
        )

        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=min(len(legend_patches), 4),
            frameon=True,
            edgecolor="#cccccc",
            fontsize=7,
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.tight_layout()
        _save_figure(fig, output_path, dpi)

    return fig, axes
