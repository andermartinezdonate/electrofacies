"""Manuscript-ready figures for the electrofacies publication.

Provides calibration curves, facies-coloured crossplots, stacked facies-
proportion charts, and a convenience wrapper to generate every standard
figure in one call.  All plots use matplotlib directly (no seaborn).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
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

LOG_COLORS: Dict[str, str] = {
    "GR": "#00a65a",
    "RESD": "#000000",
    "RHOB": "#e41a1c",
    "NPHI": "#377eb8",
    "DTC": "#ff7f00",
}

# Ordered facies list (coarsest to finest, then special)
_FACIES_ORDER: List[str] = [
    "massive_sandstone",
    "structured_sandstone",
    "sandy_siltstone",
    "siltstone",
    "calciturbidite",
    "clast_supported_conglomerate",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pub_rcparams() -> Dict[str, Any]:
    """Return scoped rcParams for publication figures."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.4,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }


def _format_facies_name(name: str) -> str:
    """Convert snake_case facies name to a human-readable label."""
    return name.replace("_", " ").capitalize()


def _save_figure(fig: plt.Figure, output_path: str | Path, dpi: int) -> None:
    """Save figure to the requested path; also save TIFF if appropriate."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, facecolor="white")
    logger.info("Saved figure to %s", output_path)

    if output_path.suffix.lower() in (".tiff", ".tif"):
        pass  # already saved as TIFF
    elif output_path.suffix.lower() != ".png":
        tiff_path = output_path.with_suffix(".tiff")
        fig.savefig(str(tiff_path), dpi=dpi, facecolor="white")
        logger.info("Saved TIFF copy to %s", tiff_path)


def _facies_sort_key(name: str) -> int:
    """Return the canonical display-order index for a facies name."""
    try:
        return _FACIES_ORDER.index(name)
    except ValueError:
        return 999


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    y_true: Union[np.ndarray, pd.Series, List],
    y_proba: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
    n_bins: int = 10,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Reliability diagram showing calibrated vs. actual probability per class.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (same encoding as *class_names*).
    y_proba : np.ndarray of shape (n_samples, n_classes)
        Predicted probability matrix.  Column order must match *class_names*.
    class_names : list of str
        Ordered class names matching the columns of *y_proba*.
    output_path : str or Path
        Destination file path.
    n_bins : int, optional
        Number of probability bins (default 10).
    dpi : int, optional
        Output resolution (default 300).

    Returns
    -------
    tuple[Figure, Axes]
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.ndim != 2 or y_proba.shape[1] != len(class_names):
        raise ValueError(
            f"y_proba shape {y_proba.shape} does not match "
            f"{len(class_names)} class names."
        )

    with plt.rc_context(_pub_rcparams()):
        fig, ax = plt.subplots(figsize=(5, 5))

        # Perfect calibration line
        ax.plot(
            [0, 1], [0, 1],
            linestyle="--",
            color="#888888",
            linewidth=0.8,
            label="Perfect calibration",
        )

        bin_edges = np.linspace(0, 1, n_bins + 1)

        for cls_idx, cls_name in enumerate(class_names):
            proba = y_proba[:, cls_idx]
            binary_true = (y_true == cls_name).astype(float)

            mean_predicted = []
            fraction_positive = []

            for b in range(n_bins):
                lo = bin_edges[b]
                hi = bin_edges[b + 1]
                mask = (proba >= lo) & (proba < hi)
                if b == n_bins - 1:
                    mask = (proba >= lo) & (proba <= hi)

                if mask.sum() == 0:
                    continue

                mean_predicted.append(proba[mask].mean())
                fraction_positive.append(binary_true[mask].mean())

            color = FACIES_COLORS.get(cls_name, None)
            # Use a darker shade for lines if the facies colour is too light
            if color:
                import matplotlib.colors as mcolors

                rgb = mcolors.to_rgb(color)
                luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                if luminance > 0.7:
                    # Darken the colour for visibility
                    color = tuple(max(0, c - 0.3) for c in rgb)

            ax.plot(
                mean_predicted,
                fraction_positive,
                marker="o",
                markersize=4,
                linewidth=1.2,
                color=color,
                label=_format_facies_name(cls_name),
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Fraction of positives", fontsize=9)
        ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")

        ax.legend(
            loc="lower right",
            fontsize=7,
            frameon=True,
            edgecolor="#cccccc",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linewidth=0.3, alpha=0.4)

        fig.tight_layout()
        _save_figure(fig, output_path, dpi)

    return fig, ax


def plot_crossplots(
    well_df: pd.DataFrame,
    facies_col: str,
    output_path: str | Path,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """2x2 crossplot grid with points coloured by facies.

    Panels:
        1. GR vs RESD (log y-axis)
        2. DTC vs RHOB
        3. GR vs NPHI
        4. RHOB vs NPHI

    Parameters
    ----------
    well_df : pd.DataFrame
        Log curves with a facies column.  Expected columns include
        ``GR``, ``RESD``, ``RHOB``, ``NPHI``, ``DTC``, and ``facies_col``.
    facies_col : str
        Column name holding the facies labels.
    output_path : str or Path
        Destination file path.
    dpi : int, optional
        Output resolution (default 300).

    Returns
    -------
    tuple[Figure, ndarray]
        The Figure and 2x2 Axes array.
    """
    # Define the four crossplot pairs: (x_col, y_col, y_log)
    crossplot_defs = [
        ("GR", "RESD", True),
        ("DTC", "RHOB", False),
        ("GR", "NPHI", False),
        ("RHOB", "NPHI", False),
    ]

    with plt.rc_context(_pub_rcparams()):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5), constrained_layout=True)
        axes_flat = axes.flatten()

        # Determine unique facies in display order
        all_facies = well_df[facies_col].dropna().unique()
        sorted_facies = sorted(all_facies, key=_facies_sort_key)

        for ax_idx, (x_col, y_col, y_log) in enumerate(crossplot_defs):
            ax = axes_flat[ax_idx]

            # Check that required columns exist
            if x_col not in well_df.columns or y_col not in well_df.columns:
                ax.text(
                    0.5, 0.5,
                    f"Missing: {x_col} or {y_col}",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=9, color="#888888",
                )
                ax.set_title(f"{x_col} vs {y_col}", fontsize=9)
                continue

            # Plot each facies as a separate scatter layer
            for facies_name in sorted_facies:
                mask = well_df[facies_col] == facies_name
                subset = well_df.loc[mask]

                if subset.empty:
                    continue

                color = FACIES_COLORS.get(facies_name, "#cccccc")

                ax.scatter(
                    subset[x_col],
                    subset[y_col],
                    c=color,
                    edgecolors="black",
                    linewidths=0.2,
                    s=12,
                    alpha=0.65,
                    label=_format_facies_name(facies_name),
                    zorder=2,
                )

            if y_log:
                ax.set_yscale("log")

            # Axis labels with units
            x_unit = _log_unit(x_col)
            y_unit = _log_unit(y_col)
            ax.set_xlabel(f"{x_col} ({x_unit})" if x_unit else x_col, fontsize=8)
            ax.set_ylabel(f"{y_col} ({y_unit})" if y_unit else y_col, fontsize=8)
            ax.set_title(f"{x_col} vs {y_col}", fontsize=9, fontweight="bold")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, linewidth=0.2, alpha=0.3)

        # Shared legend outside right
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="center right",
                bbox_to_anchor=(1.15, 0.5),
                fontsize=7,
                frameon=True,
                edgecolor="#cccccc",
                title="Facies",
                title_fontsize=8,
            )

        _save_figure(fig, output_path, dpi)

    return fig, axes


def _log_unit(mnemonic: str) -> str:
    """Return the display unit for a log mnemonic."""
    units = {
        "GR": "API",
        "RESD": "ohm.m",
        "RHOB": "g/cm\u00b3",
        "NPHI": "v/v",
        "DTC": "\u00b5s/ft",
    }
    return units.get(mnemonic, "")


def plot_facies_proportions(
    well_summaries: Dict[str, Dict[str, float]],
    output_path: str | Path,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked bar chart of facies proportions per well.

    Parameters
    ----------
    well_summaries : dict
        ``{well_name: {facies_name: proportion}}`` where proportions are
        in the range 0-1 (fractions) or 0-100 (percentages; auto-detected).
    output_path : str or Path
        Destination file path.
    dpi : int, optional
        Output resolution (default 300).

    Returns
    -------
    tuple[Figure, Axes]
    """
    if not well_summaries:
        logger.warning("Empty well_summaries; nothing to plot.")
        fig, ax = plt.subplots()
        return fig, ax

    well_names = list(well_summaries.keys())

    # Collect all facies names across wells, sorted canonically
    all_facies: set = set()
    for summary in well_summaries.values():
        all_facies.update(summary.keys())
    sorted_facies = sorted(all_facies, key=_facies_sort_key)

    # Build proportions matrix (wells x facies)
    data = np.zeros((len(well_names), len(sorted_facies)), dtype=float)
    for wi, wname in enumerate(well_names):
        for fi, fname in enumerate(sorted_facies):
            data[wi, fi] = well_summaries[wname].get(fname, 0.0)

    # Auto-detect fraction vs percentage
    if data.max() <= 1.0 and data.max() > 0:
        data *= 100.0

    with plt.rc_context(_pub_rcparams()):
        fig_width = max(5, len(well_names) * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, 5))

        x = np.arange(len(well_names))
        bar_width = 0.7
        bottoms = np.zeros(len(well_names))

        for fi, fname in enumerate(sorted_facies):
            color = FACIES_COLORS.get(fname, "#cccccc")
            ax.bar(
                x,
                data[:, fi],
                bottom=bottoms,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                label=_format_facies_name(fname),
            )
            bottoms += data[:, fi]

        ax.set_xticks(x)
        ax.set_xticklabels(well_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Proportion (%)", fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_title(
            "Facies Proportions by Well",
            fontsize=10,
            fontweight="bold",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

        # Legend outside right
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            frameon=True,
            edgecolor="#cccccc",
        )

        fig.tight_layout()
        _save_figure(fig, output_path, dpi)

    return fig, ax


def generate_all_manuscript_figures(
    results: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Generate all standard manuscript figures from a results dictionary.

    This is a convenience wrapper that calls the individual plotting
    functions.  The *results* dictionary is expected to contain the
    following keys (all optional -- missing keys are skipped):

    - ``"y_true"`` : array-like ground-truth labels
    - ``"y_pred"`` : array-like predicted labels
    - ``"y_proba"`` : (n_samples, n_classes) probability matrix
    - ``"class_names"`` : ordered list of class name strings
    - ``"well_df"`` : pd.DataFrame with log curves and a facies column
    - ``"facies_col"`` : str column name for facies in *well_df*
    - ``"well_summaries"`` : dict for :func:`plot_facies_proportions`
    - ``"feature_importances"`` : array-like importances
    - ``"feature_names"`` : list of str
    - ``"model_comparison"`` : dict for :func:`plot_model_comparison`

    Parameters
    ----------
    results : dict
        Collection of arrays and metadata described above.
    output_dir : str or Path
        Directory in which to save all figures.

    Returns
    -------
    dict
        ``{figure_name: Path}`` mapping of generated figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    class_names = results.get("class_names", _FACIES_ORDER)
    y_true = results.get("y_true")
    y_pred = results.get("y_pred")
    y_proba = results.get("y_proba")

    # Import confusion module functions (sibling module)
    try:
        from electrofacies.visualization.confusion import (
            plot_confusion_matrix,
            plot_feature_importance,
            plot_model_comparison,
        )
    except ImportError:
        logger.warning(
            "Could not import confusion module; skipping confusion-based figures."
        )
        plot_confusion_matrix = None
        plot_feature_importance = None
        plot_model_comparison = None

    # Confusion matrix
    if y_true is not None and y_pred is not None and plot_confusion_matrix is not None:
        path = output_dir / "confusion_matrix.png"
        try:
            plot_confusion_matrix(y_true, y_pred, class_names, path)
            generated["confusion_matrix"] = path
            logger.info("Generated confusion matrix: %s", path)
        except Exception:
            logger.exception("Failed to generate confusion matrix.")

    # Calibration curve
    if y_true is not None and y_proba is not None:
        path = output_dir / "calibration_curve.png"
        try:
            plot_calibration_curve(y_true, y_proba, class_names, path)
            generated["calibration_curve"] = path
            logger.info("Generated calibration curve: %s", path)
        except Exception:
            logger.exception("Failed to generate calibration curve.")

    # Crossplots
    well_df = results.get("well_df")
    facies_col = results.get("facies_col", "FACIES")
    if well_df is not None:
        path = output_dir / "crossplots.png"
        try:
            plot_crossplots(well_df, facies_col, path)
            generated["crossplots"] = path
            logger.info("Generated crossplots: %s", path)
        except Exception:
            logger.exception("Failed to generate crossplots.")

    # Facies proportions
    well_summaries = results.get("well_summaries")
    if well_summaries is not None:
        path = output_dir / "facies_proportions.png"
        try:
            plot_facies_proportions(well_summaries, path)
            generated["facies_proportions"] = path
            logger.info("Generated facies proportions: %s", path)
        except Exception:
            logger.exception("Failed to generate facies proportions.")

    # Feature importance
    importances = results.get("feature_importances")
    feature_names = results.get("feature_names")
    if (
        importances is not None
        and feature_names is not None
        and plot_feature_importance is not None
    ):
        path = output_dir / "feature_importance.png"
        try:
            plot_feature_importance(importances, feature_names, path)
            generated["feature_importance"] = path
            logger.info("Generated feature importance: %s", path)
        except Exception:
            logger.exception("Failed to generate feature importance.")

    # Model comparison
    model_comparison = results.get("model_comparison")
    if model_comparison is not None and plot_model_comparison is not None:
        path = output_dir / "model_comparison.png"
        try:
            plot_model_comparison(model_comparison, path)
            generated["model_comparison"] = path
            logger.info("Generated model comparison: %s", path)
        except Exception:
            logger.exception("Failed to generate model comparison.")

    logger.info(
        "generate_all_manuscript_figures complete: %d / 6 figures generated.",
        len(generated),
    )
    return generated
