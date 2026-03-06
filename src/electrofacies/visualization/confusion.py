"""Classification evaluation figures for the electrofacies pipeline.

Provides publication-quality confusion matrices (counts and normalized),
model-comparison bar charts, and feature-importance plots.  All figures
use matplotlib directly (no seaborn) and save to PNG (optionally TIFF).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pub_rcparams() -> Dict[str, Any]:
    """Return rcParams for publication figures (scoped, non-global)."""
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
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }


def _format_class_name(name: str) -> str:
    """Convert snake_case class name to readable label."""
    return name.replace("_", " ").capitalize()


def _save_figure(fig: plt.Figure, output_path: str | Path, dpi: int) -> None:
    """Save figure to the requested format; save TIFF if extension matches."""
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


def _compute_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> np.ndarray:
    """Build a confusion matrix (rows=true, cols=predicted) aligned to *class_names*."""
    n = len(class_names)
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti = name_to_idx.get(str(t))
        pi = name_to_idx.get(str(p))
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    return cm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    class_names: List[str],
    output_path: str | Path,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """Publication-quality confusion matrix with counts and row-normalized panels.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    class_names : list of str
        Ordered class names (used for row/column labels).
    output_path : str or Path
        Destination file path.
    title : str, optional
        Suptitle for the figure.
    dpi : int, optional
        Output resolution (default 300).

    Returns
    -------
    tuple[Figure, ndarray]
        The Figure and a 1-D array of two Axes (counts, normalized).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm_counts = _compute_confusion(y_true, y_pred, class_names)

    # Row-normalised matrix (percentages)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_counts.astype(float) / row_sums_safe * 100.0

    nice_names = [_format_class_name(n) for n in class_names]
    n_classes = len(class_names)

    with plt.rc_context(_pub_rcparams()):
        fig, axes = plt.subplots(
            1, 2,
            figsize=(5.5 + n_classes * 0.6, 2.8 + n_classes * 0.45),
            constrained_layout=True,
        )

        for ax_idx, (matrix, fmt, panel_title, cbar_label) in enumerate(
            [
                (cm_counts, "d", "Counts", "Count"),
                (cm_norm, ".1f", "Row-normalized (%)", "Percent"),
            ]
        ):
            ax = axes[ax_idx]

            # Use YlOrBr colormap
            cmap = plt.cm.YlOrBr
            vmax = matrix.max() if ax_idx == 0 else 100.0
            im = ax.imshow(
                matrix,
                interpolation="nearest",
                cmap=cmap,
                aspect="equal",
                vmin=0,
                vmax=vmax,
            )

            # White grid lines between cells
            for edge in range(n_classes + 1):
                ax.axhline(edge - 0.5, color="white", linewidth=1.5)
                ax.axvline(edge - 0.5, color="white", linewidth=1.5)

            # Annotate cells
            thresh = vmax * 0.6
            for i in range(n_classes):
                for j in range(n_classes):
                    val = matrix[i, j]
                    on_diagonal = i == j

                    # Choose text colour for readability
                    text_color = "white" if val > thresh else "black"

                    # Format value
                    if fmt == "d":
                        text = f"{int(val)}"
                    else:
                        text = f"{val:.1f}"

                    fontweight = "bold" if on_diagonal else "normal"

                    txt = ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        fontweight=fontweight,
                    )
                    # Path effect for readability on mid-tone cells
                    if val > thresh:
                        txt.set_path_effects(
                            [
                                pe.withStroke(linewidth=2, foreground="black"),
                            ]
                        )
                    else:
                        txt.set_path_effects(
                            [
                                pe.withStroke(linewidth=2, foreground="white"),
                            ]
                        )

            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(nice_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(nice_names, fontsize=7)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.set_title(panel_title, fontsize=10, fontweight="bold")

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(cbar_label, fontsize=7)

        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold", y=1.03)

        _save_figure(fig, output_path, dpi)

    return fig, axes


def plot_model_comparison(
    results_dict: Dict[str, float],
    output_path: str | Path,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar chart comparing balanced accuracy across models or tiers.

    Parameters
    ----------
    results_dict : dict
        ``{model_or_tier_name: balanced_accuracy}`` mapping.  Values should
        be in the range 0-1 (or 0-100; auto-detected).
    output_path : str or Path
        Destination file path.
    dpi : int, optional
        Output resolution (default 300).

    Returns
    -------
    tuple[Figure, Axes]
    """
    with plt.rc_context(_pub_rcparams()):
        names = list(results_dict.keys())
        values = np.array(list(results_dict.values()), dtype=float)

        # Auto-detect if values are in 0-1 or 0-100
        if np.nanmax(values) <= 1.0:
            values_pct = values * 100.0
        else:
            values_pct = values

        nice_names = [_format_class_name(n) for n in names]

        fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.0), 4))

        # Colour gradient from light to dark based on value
        cmap = plt.cm.YlOrBr
        norm = plt.Normalize(vmin=max(0, values_pct.min() - 15), vmax=100)
        colors = [cmap(norm(v)) for v in values_pct]

        bars = ax.bar(
            range(len(names)),
            values_pct,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            width=0.65,
        )

        # Annotate bars
        for bar, val in zip(bars, values_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(nice_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Balanced Accuracy (%)", fontsize=9)
        ax.set_ylim(0, min(110, values_pct.max() + 12))
        ax.set_title("Model / Tier Comparison", fontsize=10, fontweight="bold")

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

        fig.tight_layout()
        _save_figure(fig, output_path, dpi)

    return fig, ax


def plot_feature_importance(
    importances: Union[np.ndarray, List[float]],
    feature_names: List[str],
    output_path: str | Path,
    top_n: int = 15,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    importances : array-like
        Feature importance values (e.g. from a tree-based model).
    feature_names : list of str
        Names corresponding to each importance value.
    output_path : str or Path
        Destination file path.
    top_n : int, optional
        Number of top features to display (default 15).
    dpi : int, optional
        Output resolution (default 300).

    Returns
    -------
    tuple[Figure, Axes]
    """
    importances = np.asarray(importances, dtype=float)
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Length mismatch: importances ({len(importances)}) vs "
            f"feature_names ({len(feature_names)})."
        )

    # Sort descending and take top_n
    sorted_idx = np.argsort(importances)[::-1]
    top_idx = sorted_idx[:top_n]

    top_importances = importances[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    # Reverse for horizontal bar chart (top feature at the top)
    top_importances = top_importances[::-1]
    top_names = top_names[::-1]

    with plt.rc_context(_pub_rcparams()):
        fig_height = max(3.5, len(top_names) * 0.35)
        fig, ax = plt.subplots(figsize=(5, fig_height))

        # Colour gradient
        cmap = plt.cm.YlOrBr
        max_imp = top_importances.max() if top_importances.max() > 0 else 1.0
        norm = plt.Normalize(vmin=0, vmax=max_imp)
        colors = [cmap(norm(v)) for v in top_importances]

        y_pos = range(len(top_names))
        ax.barh(
            y_pos,
            top_importances,
            color=colors,
            edgecolor="black",
            linewidth=0.4,
            height=0.7,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_title(
            f"Top {len(top_names)} Feature Importances",
            fontsize=10,
            fontweight="bold",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linewidth=0.3, alpha=0.4)

        fig.tight_layout()
        _save_figure(fig, output_path, dpi)

    return fig, ax
