"""Model evaluation utilities for the electrofacies prediction pipeline.

Provides functions for single-model evaluation, grouped cross-validation
scoring, and multi-model comparison.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a fitted model on a held-out test set.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn (or imblearn) estimator/pipeline with a
        ``predict`` method.
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_samples,)
        True labels for the test set (integer-encoded).
    class_names : list[str] or None, optional
        Human-readable facies names, ordered by integer code.  If ``None``,
        unique sorted labels from *y_test* are used.

    Returns
    -------
    dict
        Comprehensive metrics dictionary with keys:

        - ``accuracy`` : float
        - ``balanced_accuracy`` : float
        - ``cohen_kappa`` : float
        - ``per_class`` : dict mapping class name to
          ``{precision, recall, f1, support}``
        - ``confusion_matrix`` : list[list[int]]
        - ``classification_report`` : str (formatted table)
    """
    y_test = np.asarray(y_test)
    y_pred = model.predict(X_test)

    labels = np.unique(np.concatenate([y_test, y_pred]))
    labels.sort()

    # Build target names list aligned with labels
    if class_names is not None:
        # Handle both integer-coded and string labels
        class_set = set(class_names)
        if all(str(lbl) in class_set for lbl in labels):
            # Labels are already strings matching class_names
            target_names = [str(lbl) for lbl in labels]
        else:
            # Labels are integer codes — index into class_names
            target_names = [
                class_names[int(lbl)] if int(lbl) < len(class_names) else f"class_{lbl}"
                for lbl in labels
            ]
    else:
        target_names = [str(lbl) for lbl in labels]

    # Global metrics
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    kappa = float(cohen_kappa_score(y_test, y_pred))

    # Per-class metrics
    prec = precision_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    per_class: Dict[str, Dict[str, float]] = {}
    for i, lbl in enumerate(labels):
        name = target_names[i]
        support = int(np.sum(y_test == lbl))
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": support,
        }

    # Text report
    report_str = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "cohen_kappa": kappa,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
    }

    logger.info(
        "Evaluation — accuracy=%.4f  balanced_accuracy=%.4f  kappa=%.4f",
        acc,
        bal_acc,
        kappa,
    )
    logger.debug("Classification report:\n%s", report_str)

    return metrics


# ---------------------------------------------------------------------------
# Grouped cross-validation evaluation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run grouped (or stratified) k-fold CV and aggregate metrics.

    Parameters
    ----------
    model : estimator
        An *unfitted* scikit-learn estimator/pipeline that will be cloned
        for each fold.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector (integer-encoded).
    groups : np.ndarray or None, optional
        Group labels for ``GroupKFold``.  If ``None``, ``StratifiedKFold``
        is used instead.
    n_splits : int, optional
        Number of CV folds (default 5).
    class_names : list[str] or None, optional
        Human-readable facies names.

    Returns
    -------
    dict
        Keys:

        - ``per_fold`` : list[dict] — metrics from :func:`evaluate_model`
          for each fold.
        - ``mean_accuracy`` : float
        - ``std_accuracy`` : float
        - ``mean_balanced_accuracy`` : float
        - ``std_balanced_accuracy`` : float
        - ``mean_cohen_kappa`` : float
        - ``std_cohen_kappa`` : float
        - ``n_splits`` : int — actual number of folds used.
    """
    from sklearn.base import clone

    X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
    y = np.asarray(y)

    # Choose CV strategy
    if groups is not None:
        n_unique = len(np.unique(groups))
        actual_splits = min(n_splits, n_unique)
        if actual_splits < n_splits:
            logger.warning(
                "Reducing n_splits from %d to %d (only %d groups).",
                n_splits,
                actual_splits,
                n_unique,
            )
        cv = GroupKFold(n_splits=actual_splits)
        split_iter = cv.split(X, y, groups)
    else:
        actual_splits = n_splits
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
        split_iter = cv.split(X, y)

    per_fold: List[Dict[str, Any]] = []
    accuracies: List[float] = []
    balanced_accs: List[float] = []
    kappas: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        logger.info("CV fold %d/%d — train %d, val %d",
                     fold_idx + 1, actual_splits, len(train_idx), len(val_idx))

        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        y_train = y[train_idx]
        X_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
        y_val = y[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        fold_metrics = evaluate_model(fold_model, X_val, y_val, class_names)
        fold_metrics["fold"] = fold_idx
        per_fold.append(fold_metrics)

        accuracies.append(fold_metrics["accuracy"])
        balanced_accs.append(fold_metrics["balanced_accuracy"])
        kappas.append(fold_metrics["cohen_kappa"])

    result = {
        "per_fold": per_fold,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_balanced_accuracy": float(np.mean(balanced_accs)),
        "std_balanced_accuracy": float(np.std(balanced_accs)),
        "mean_cohen_kappa": float(np.mean(kappas)),
        "std_cohen_kappa": float(np.std(kappas)),
        "n_splits": actual_splits,
    }

    logger.info(
        "CV summary (%d folds) — bal_acc=%.4f +/- %.4f, kappa=%.4f +/- %.4f",
        actual_splits,
        result["mean_balanced_accuracy"],
        result["std_balanced_accuracy"],
        result["mean_cohen_kappa"],
        result["std_cohen_kappa"],
    )
    return result


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Build a comparison table from multiple models' evaluation metrics.

    Parameters
    ----------
    results_dict : dict
        Mapping of ``model_name`` (e.g. ``"tier_1/random_forest"``) to a
        metrics dictionary as returned by :func:`evaluate_model`.

    Returns
    -------
    pd.DataFrame
        One row per model, columns include ``accuracy``,
        ``balanced_accuracy``, ``cohen_kappa``, and per-class F1 scores.
        Sorted descending by ``balanced_accuracy``.
    """
    rows: List[Dict[str, Any]] = []

    for model_name, metrics in results_dict.items():
        row: Dict[str, Any] = {
            "model": model_name,
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "cohen_kappa": metrics.get("cohen_kappa"),
        }
        # Per-class F1 scores
        per_class = metrics.get("per_class", {})
        for cls_name, cls_metrics in per_class.items():
            row[f"f1_{cls_name}"] = cls_metrics.get("f1")
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)

    logger.info("Model comparison (%d models):\n%s", len(df), df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Stratified GroupKFold CV evaluation (post-tuning)
# ---------------------------------------------------------------------------

def evaluate_model_cv(
    build_model_fn,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run stratified grouped k-fold CV with a model builder function.

    Unlike :func:`cross_validate_model`, this accepts a **factory function**
    that returns a fresh unfitted model for each fold, avoiding sklearn
    ``clone`` issues with custom wrappers.

    Parameters
    ----------
    build_model_fn : callable
        Zero-argument callable that returns a fresh unfitted pipeline
        (e.g. ``lambda: ImbPipeline([("smote", SMOTE()), ("clf", RF())])``).
    X, y, groups, n_splits, class_names
        Same as :func:`cross_validate_model`.

    Returns
    -------
    dict
        ``mean_balanced_accuracy``, ``std_balanced_accuracy``,
        ``mean_cohen_kappa``, ``std_cohen_kappa``, ``per_class_cv``
        (per-class mean/std recall and F1), and ``per_fold`` details.
    """
    X_arr = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
    y_arr = np.asarray(y)

    # Choose CV strategy — prefer StratifiedGroupKFold when groups exist
    if groups is not None:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            n_unique = len(np.unique(groups))
            actual_splits = min(n_splits, n_unique)
            cv = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=42)
            split_iter = list(cv.split(X_arr, y_arr, groups))
            logger.info("CV: StratifiedGroupKFold with %d splits", actual_splits)
        except ImportError:
            n_unique = len(np.unique(groups))
            actual_splits = min(n_splits, n_unique)
            cv = GroupKFold(n_splits=actual_splits)
            split_iter = list(cv.split(X_arr, y_arr, groups))
            logger.info("CV: GroupKFold with %d splits (StratifiedGroupKFold unavailable)", actual_splits)
    else:
        actual_splits = n_splits
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
        split_iter = list(cv.split(X_arr, y_arr))
        logger.info("CV: StratifiedKFold with %d splits", actual_splits)

    per_fold: List[Dict[str, Any]] = []
    bal_accs, kappas = [], []
    # Collect per-class metrics across folds
    per_class_folds: Dict[str, List[Dict[str, float]]] = {}

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        X_tr = X_arr.iloc[train_idx] if isinstance(X_arr, pd.DataFrame) else X_arr[train_idx]
        y_tr = y_arr[train_idx]
        X_val = X_arr.iloc[val_idx] if isinstance(X_arr, pd.DataFrame) else X_arr[val_idx]
        y_val = y_arr[val_idx]

        model = build_model_fn()
        model.fit(X_tr, y_tr)

        fold_metrics = evaluate_model(model, X_val, y_val, class_names)
        fold_metrics["fold"] = fold_idx
        per_fold.append(fold_metrics)

        bal_accs.append(fold_metrics["balanced_accuracy"])
        kappas.append(fold_metrics["cohen_kappa"])

        for cls_name, cls_m in fold_metrics.get("per_class", {}).items():
            per_class_folds.setdefault(cls_name, []).append(cls_m)

    # Aggregate per-class across folds
    per_class_cv: Dict[str, Dict[str, float]] = {}
    for cls_name, fold_list in per_class_folds.items():
        recalls = [m["recall"] for m in fold_list]
        f1s = [m["f1"] for m in fold_list]
        precisions = [m["precision"] for m in fold_list]
        per_class_cv[cls_name] = {
            "mean_recall": float(np.mean(recalls)),
            "std_recall": float(np.std(recalls)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "mean_precision": float(np.mean(precisions)),
            "std_precision": float(np.std(precisions)),
        }

    result = {
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs)),
        "mean_cohen_kappa": float(np.mean(kappas)),
        "std_cohen_kappa": float(np.std(kappas)),
        "n_splits": actual_splits,
        "per_class_cv": per_class_cv,
        "per_fold": per_fold,
    }

    logger.info(
        "CV evaluation (%d folds): bal_acc=%.4f±%.4f, kappa=%.4f±%.4f",
        actual_splits,
        result["mean_balanced_accuracy"],
        result["std_balanced_accuracy"],
        result["mean_cohen_kappa"],
        result["std_cohen_kappa"],
    )
    for cls_name, cv_m in per_class_cv.items():
        logger.info(
            "  %s: recall=%.3f±%.3f, F1=%.3f±%.3f",
            cls_name, cv_m["mean_recall"], cv_m["std_recall"],
            cv_m["mean_f1"], cv_m["std_f1"],
        )
    return result


# ---------------------------------------------------------------------------
# Confusion matrix analysis
# ---------------------------------------------------------------------------

def analyze_class_confusion(
    metrics: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    top_n: int = 2,
) -> Dict[str, Dict]:
    """Identify the top confusion targets for each class.

    Parameters
    ----------
    metrics : dict
        Metrics dict from :func:`evaluate_model` (must contain
        ``confusion_matrix`` and ``per_class``).
    class_names : list[str], optional
        Class names in the same order as the confusion matrix rows/columns.
        If ``None``, derived from ``per_class`` keys.
    top_n : int
        Number of top confusion targets to report per class.

    Returns
    -------
    dict
        ``{class_name: {"recall": float, "top_confusions": [(target, count, pct), ...]}}``
    """
    cm = np.array(metrics.get("confusion_matrix", []))
    if cm.size == 0:
        return {}

    if class_names is None:
        class_names = list(metrics.get("per_class", {}).keys())

    report: Dict[str, Dict] = {}
    for i, cls in enumerate(class_names):
        if i >= cm.shape[0]:
            break
        row = cm[i]
        total = int(row.sum())
        if total == 0:
            continue
        correct = int(row[i])
        recall = correct / total

        # Find top confusion targets (excluding self)
        confusions = []
        for j, count in enumerate(row):
            if j != i and count > 0 and j < len(class_names):
                confusions.append((class_names[j], int(count), int(count) / total))
        confusions.sort(key=lambda x: x[1], reverse=True)

        report[cls] = {
            "recall": recall,
            "support": total,
            "correct": correct,
            "top_confusions": confusions[:top_n],
        }

        if confusions:
            targets = ", ".join(
                f"{t[0]}={t[1]} ({t[2]:.1%})" for t in confusions[:top_n]
            )
            logger.info(
                "  %s (recall=%.3f, n=%d): misclassified as → %s",
                cls, recall, total, targets,
            )

    return report
