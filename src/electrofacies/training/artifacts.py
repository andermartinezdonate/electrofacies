"""Model artifact persistence for the electrofacies prediction pipeline.

Saves and loads complete model bundles comprising the fitted pipeline,
feature transformer, OOD detector, configuration snapshot, and evaluation
metrics.  Bundles are organised in timestamped directories under the
project's ``artifacts/`` folder.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import yaml

logger = logging.getLogger(__name__)

# Bundle component filenames (constants for consistency)
_MODEL_FILE = "model.joblib"
_TRANSFORMER_FILE = "transformer.joblib"
_OOD_DETECTOR_FILE = "ood_detector.joblib"
_CONFIG_FILE = "config_snapshot.yaml"
_METRICS_FILE = "metrics.json"
_METADATA_FILE = "metadata.json"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model_bundle(
    model: Any,
    transformer: Any,
    ood_detector: Any,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: str | Path,
    tier_name: str,
    algorithm: str,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> str:
    """Persist a complete model bundle to disk.

    Creates a timestamped directory under *output_dir* containing all
    components needed to reconstruct the model at inference time:

    - ``model.joblib`` — the fitted imblearn/sklearn pipeline.
    - ``transformer.joblib`` — the fitted ``FaciesTransformer``.
    - ``ood_detector.joblib`` — fitted OOD detector.
    - ``config_snapshot.yaml`` — exact configuration used for training.
    - ``metrics.json`` — held-out-set or CV evaluation metrics.
    - ``metadata.json`` — timestamp, tier, algorithm, feature list, class
      names.

    Parameters
    ----------
    model : estimator
        Fitted pipeline or estimator.
    transformer : object
        Fitted feature transformer (e.g. ``FaciesTransformer``).
    ood_detector : object
        Fitted out-of-distribution detector.
    config : dict
        Configuration dictionary used during training.
    metrics : dict
        Evaluation metrics dictionary (from :func:`evaluate_model`).
    output_dir : str or Path
        Base directory for artifacts (e.g. ``"artifacts"``).
    tier_name : str
        Tier identifier (e.g. ``"tier_1"``).
    algorithm : str
        Algorithm identifier (e.g. ``"random_forest"``).

    Returns
    -------
    str
        Absolute path to the created bundle directory.
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_name = f"{tier_name}_{algorithm}_{timestamp}"
    bundle_dir = output_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving model bundle to %s", bundle_dir)

    # 1. Model pipeline
    model_path = bundle_dir / _MODEL_FILE
    joblib.dump(model, model_path)
    logger.debug("Saved model -> %s", model_path)

    # 2. Transformer
    transformer_path = bundle_dir / _TRANSFORMER_FILE
    joblib.dump(transformer, transformer_path)
    logger.debug("Saved transformer -> %s", transformer_path)

    # 3. OOD detector
    ood_path = bundle_dir / _OOD_DETECTOR_FILE
    joblib.dump(ood_detector, ood_path)
    logger.debug("Saved OOD detector -> %s", ood_path)

    # 4. Config snapshot
    config_path = bundle_dir / _CONFIG_FILE
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=False)
    logger.debug("Saved config snapshot -> %s", config_path)

    # 5. Metrics
    metrics_path = bundle_dir / _METRICS_FILE
    _write_json(metrics, metrics_path)
    logger.debug("Saved metrics -> %s", metrics_path)

    # 6. Metadata
    if feature_names is None:
        feature_names = _extract_feature_names(model)
    if class_names is None:
        class_names = _extract_class_names(metrics)

    metadata = {
        "timestamp": timestamp,
        "tier": tier_name,
        "algorithm": algorithm,
        "feature_names": feature_names,
        "class_names": class_names,
        "bundle_format_version": "1.0",
    }
    metadata_path = bundle_dir / _METADATA_FILE
    _write_json(metadata, metadata_path)
    logger.debug("Saved metadata -> %s", metadata_path)

    bundle_abs = str(bundle_dir.resolve())
    logger.info("Bundle saved successfully: %s", bundle_abs)
    return bundle_abs


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_model_bundle(bundle_dir: str | Path) -> Dict[str, Any]:
    """Load all components of a saved model bundle.

    Parameters
    ----------
    bundle_dir : str or Path
        Path to a bundle directory previously created by
        :func:`save_model_bundle`.

    Returns
    -------
    dict
        Keys: ``model``, ``transformer``, ``ood_detector``, ``config``,
        ``metrics``, ``metadata``.

    Raises
    ------
    FileNotFoundError
        If *bundle_dir* does not exist or is missing required files.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    required_files = [_MODEL_FILE, _METADATA_FILE]
    for fname in required_files:
        if not (bundle_dir / fname).exists():
            raise FileNotFoundError(
                f"Required file '{fname}' missing from bundle: {bundle_dir}"
            )

    logger.info("Loading model bundle from %s", bundle_dir)

    result: Dict[str, Any] = {}

    # Model (required)
    result["model"] = joblib.load(bundle_dir / _MODEL_FILE)
    logger.debug("Loaded model")

    # Transformer (optional — may not exist for older bundles)
    transformer_path = bundle_dir / _TRANSFORMER_FILE
    if transformer_path.exists():
        result["transformer"] = joblib.load(transformer_path)
        logger.debug("Loaded transformer")
    else:
        result["transformer"] = None
        logger.warning("Transformer file not found in bundle — set to None.")

    # OOD detector (optional)
    ood_path = bundle_dir / _OOD_DETECTOR_FILE
    if ood_path.exists():
        result["ood_detector"] = joblib.load(ood_path)
        logger.debug("Loaded OOD detector")
    else:
        result["ood_detector"] = None
        logger.warning("OOD detector file not found in bundle — set to None.")

    # Config snapshot (optional)
    config_path = bundle_dir / _CONFIG_FILE
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            result["config"] = yaml.safe_load(fh)
        logger.debug("Loaded config snapshot")
    else:
        result["config"] = {}
        logger.warning("Config snapshot not found in bundle — set to empty dict.")

    # Metrics (optional)
    metrics_path = bundle_dir / _METRICS_FILE
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as fh:
            result["metrics"] = json.load(fh)
        logger.debug("Loaded metrics")
    else:
        result["metrics"] = {}
        logger.warning("Metrics file not found in bundle — set to empty dict.")

    # Metadata (required)
    with open(bundle_dir / _METADATA_FILE, "r", encoding="utf-8") as fh:
        result["metadata"] = json.load(fh)
    logger.debug("Loaded metadata")

    logger.info(
        "Bundle loaded: tier=%s, algorithm=%s, timestamp=%s",
        result["metadata"].get("tier", "?"),
        result["metadata"].get("algorithm", "?"),
        result["metadata"].get("timestamp", "?"),
    )
    return result


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def get_latest_bundle(
    artifacts_dir: str | Path,
    tier: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> str:
    """Find the most recent model bundle matching optional filters.

    Bundle directories are expected to follow the naming convention
    ``{tier}_{algorithm}_{timestamp}`` (as created by
    :func:`save_model_bundle`).

    Parameters
    ----------
    artifacts_dir : str or Path
        Root directory containing bundle sub-directories.
    tier : str or None, optional
        Filter by tier name (e.g. ``"tier_1"``).
    algorithm : str or None, optional
        Filter by algorithm (e.g. ``"random_forest"``).

    Returns
    -------
    str
        Absolute path to the most recently created bundle directory.

    Raises
    ------
    FileNotFoundError
        If no matching bundles are found.
    """
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    candidates: List[Path] = []
    for entry in artifacts_dir.iterdir():
        if not entry.is_dir():
            continue
        # Must contain metadata.json to be considered a valid bundle
        if not (entry / _METADATA_FILE).exists():
            continue

        dir_name = entry.name

        # Apply tier filter
        if tier is not None and not dir_name.startswith(f"{tier}_"):
            continue

        # Apply algorithm filter
        if algorithm is not None and f"_{algorithm}_" not in dir_name:
            continue

        candidates.append(entry)

    if not candidates:
        filters = []
        if tier:
            filters.append(f"tier={tier}")
        if algorithm:
            filters.append(f"algorithm={algorithm}")
        filter_str = ", ".join(filters) if filters else "none"
        raise FileNotFoundError(
            f"No matching bundles in {artifacts_dir} (filters: {filter_str})"
        )

    # Sort by directory name (which includes a timestamp) — newest last
    candidates.sort(key=lambda p: p.name)
    latest = candidates[-1]
    logger.info("Latest bundle matching filters: %s", latest)
    return str(latest.resolve())


# ===================================================================
# Internal helpers
# ===================================================================

def _write_json(data: Any, path: Path) -> None:
    """Write a JSON-serialisable object to *path*, converting numpy types."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Custom JSON serialiser for numpy/pandas types."""
    import numpy as _np

    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _extract_feature_names(model: Any) -> List[str]:
    """Best-effort extraction of feature names from a fitted pipeline."""
    # imblearn/sklearn Pipeline: try the last step's feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # Walk pipeline steps
    if hasattr(model, "named_steps"):
        for step_name in reversed(list(model.named_steps)):
            step = model.named_steps[step_name]
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    # XGBoost stores get_booster().feature_names
    if hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names
            if names:
                return list(names)
        except Exception:
            pass

    logger.debug("Could not extract feature names from model.")
    return []


def _extract_class_names(metrics: Dict[str, Any]) -> List[str]:
    """Extract class names from the per_class section of a metrics dict."""
    per_class = metrics.get("per_class", {})
    if per_class:
        return list(per_class.keys())
    return []
