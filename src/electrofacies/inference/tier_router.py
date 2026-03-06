"""Model tier routing for well-log inference.

Determines which model tier to use for a given well based on the available
canonical log curves and their data coverage.  Loads the corresponding
model bundles (one per algorithm) from the artifacts directory.

The tier system follows a priority scheme defined in *model_tiers.yaml*:
higher-priority tiers require more logs but produce better predictions.
The router selects the highest-priority tier whose required logs are **all**
present with sufficient non-null coverage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_tier_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load model tier definitions from *model_tiers.yaml*.

    Parameters
    ----------
    config_path : str or Path
        Absolute or relative path to the ``model_tiers.yaml`` file.

    Returns
    -------
    dict
        Parsed YAML contents.  The ``"tiers"`` key holds a mapping of tier
        names to their configuration dicts (each containing ``required_logs``,
        ``priority``, etc.).

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If the YAML is missing the required ``"tiers"`` key.
    """
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Model tiers config not found: {config_path}"
        )

    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh)

    if raw is None or "tiers" not in raw:
        raise ValueError(
            f"Invalid model_tiers config at {config_path}: "
            "missing 'tiers' key."
        )

    logger.info(
        "Loaded tier config from %s: %d tier(s) defined.",
        config_path,
        len(raw["tiers"]),
    )
    return raw


# ---------------------------------------------------------------------------
# Log availability detection
# ---------------------------------------------------------------------------

def determine_available_logs(
    df: pd.DataFrame,
    canonical_logs: List[str],
    min_coverage: float = 0.7,
) -> List[str]:
    """Identify canonical logs present in *df* with sufficient data coverage.

    A log is considered *available* only if:
    1. Its canonical name exists as a column in *df*.
    2. The fraction of non-null values in that column is at least
       *min_coverage*.

    Parameters
    ----------
    df : pd.DataFrame
        Well-log DataFrame with canonical column names (after mnemonic
        standardisation).
    canonical_logs : list of str
        Ordered list of canonical log mnemonics to check
        (e.g. ``["GR", "RESD", "RHOB", "NPHI", "DTC"]``).
    min_coverage : float, optional
        Minimum fraction of non-null values required for a log to be
        considered available.  Defaults to ``0.7`` (70 %).

    Returns
    -------
    list of str
        Canonical log names that are both present and have sufficient
        coverage, preserving the input order.
    """
    if df.empty:
        logger.warning("determine_available_logs received an empty DataFrame.")
        return []

    n_rows = len(df)
    available: List[str] = []

    for log_name in canonical_logs:
        if log_name not in df.columns:
            logger.debug("Log '%s' not present in DataFrame columns.", log_name)
            continue

        non_null_count = df[log_name].notna().sum()
        coverage = non_null_count / n_rows

        if coverage >= min_coverage:
            available.append(log_name)
            logger.debug(
                "Log '%s' available: coverage %.1f%% (%d / %d).",
                log_name,
                coverage * 100,
                non_null_count,
                n_rows,
            )
        else:
            logger.info(
                "Log '%s' present but insufficient coverage: "
                "%.1f%% < %.1f%% threshold (%d / %d non-null).",
                log_name,
                coverage * 100,
                min_coverage * 100,
                non_null_count,
                n_rows,
            )

    logger.info(
        "Available canonical logs (%d / %d checked): %s",
        len(available),
        len(canonical_logs),
        available,
    )
    return available


# ---------------------------------------------------------------------------
# Tier selection
# ---------------------------------------------------------------------------

def select_best_tier(
    available_logs: List[str],
    tiers_config: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Select the highest-priority tier whose required logs are all available.

    Tiers are evaluated in ascending order of their ``priority`` value
    (priority 1 is the best / first tried).  The first tier whose
    ``required_logs`` are a subset of *available_logs* is returned.

    Parameters
    ----------
    available_logs : list of str
        Canonical log names determined to be available for the well.
    tiers_config : dict
        The ``"tiers"`` mapping from the parsed *model_tiers.yaml*, i.e.
        ``tier_config["tiers"]``.

    Returns
    -------
    tuple of (str, dict) or None
        ``(tier_name, tier_dict)`` for the best matching tier, or ``None``
        if no tier can be satisfied by the available logs.
    """
    available_set = set(available_logs)

    # Sort tiers by priority (ascending -- lower number = higher priority).
    sorted_tiers = sorted(
        tiers_config.items(),
        key=lambda item: item[1].get("priority", 999),
    )

    for tier_name, tier_dict in sorted_tiers:
        required = tier_dict.get("required_logs", [])
        if set(required).issubset(available_set):
            logger.info(
                "Selected tier '%s' (%s, priority %d) -- "
                "required logs %s are all available.",
                tier_name,
                tier_dict.get("name", ""),
                tier_dict.get("priority", -1),
                required,
            )
            return tier_name, tier_dict

    logger.warning(
        "No model tier can be satisfied.  Available logs: %s.  "
        "Tier requirements: %s",
        available_logs,
        {
            name: cfg.get("required_logs", [])
            for name, cfg in sorted_tiers
        },
    )
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tier_models(
    artifacts_dir: Union[str, Path],
    tier_name: str,
) -> Dict[str, Any]:
    """Load all algorithm model bundles for the specified tier.

    Expects the artifacts directory to contain sub-directories or bundle
    files following the naming convention used by
    :func:`electrofacies.training.artifacts.load_model_bundle`.

    The function attempts to import and call ``load_model_bundle`` for each
    known algorithm (currently ``random_forest`` and ``xgboost``).  Algorithms
    whose bundles are not found are logged as warnings but do not raise.

    Parameters
    ----------
    artifacts_dir : str or Path
        Root directory containing trained model bundles.
    tier_name : str
        Name of the tier (e.g. ``"tier_1"``).

    Returns
    -------
    dict
        ``{algorithm_name: model_bundle}`` for every algorithm whose bundle
        was successfully loaded.  May be empty if no bundles are found.

    Raises
    ------
    FileNotFoundError
        If *artifacts_dir* does not exist.
    """
    artifacts_dir = Path(artifacts_dir).resolve()
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(
            f"Artifacts directory not found: {artifacts_dir}"
        )

    # Lazy import to avoid hard circular dependency at module level.
    from electrofacies.training.artifacts import load_model_bundle

    algorithms = ["random_forest", "xgboost"]
    loaded: Dict[str, Any] = {}

    for algo in algorithms:
        # Try exact path first: artifacts/<tier>/<algo>
        bundle_dir = artifacts_dir / tier_name / algo
        if not bundle_dir.is_dir():
            # Try flat naming: artifacts/<tier>_<algo>
            bundle_dir = artifacts_dir / f"{tier_name}_{algo}"
        if not bundle_dir.is_dir():
            # Try timestamped naming: artifacts/<tier>_<algo>_<timestamp>
            # (created by save_model_bundle — pick the latest match)
            pattern = f"{tier_name}_{algo}_*"
            candidates = sorted(artifacts_dir.glob(pattern))
            candidates = [c for c in candidates if c.is_dir()]
            if candidates:
                bundle_dir = candidates[-1]  # latest by lexicographic sort
            else:
                logger.warning(
                    "No bundle directory found for tier '%s', algorithm '%s' "
                    "under %s.",
                    tier_name,
                    algo,
                    artifacts_dir,
                )
                continue

        try:
            bundle = load_model_bundle(bundle_dir)
            loaded[algo] = bundle
            logger.info(
                "Loaded model bundle for tier '%s', algorithm '%s' from %s.",
                tier_name,
                algo,
                bundle_dir,
            )
        except Exception:
            logger.exception(
                "Failed to load model bundle for tier '%s', algorithm '%s' "
                "from %s.",
                tier_name,
                algo,
                bundle_dir,
            )

    if not loaded:
        logger.error(
            "No model bundles could be loaded for tier '%s' in %s.",
            tier_name,
            artifacts_dir,
        )
    else:
        logger.info(
            "Loaded %d algorithm(s) for tier '%s': %s",
            len(loaded),
            tier_name,
            list(loaded.keys()),
        )

    return loaded
