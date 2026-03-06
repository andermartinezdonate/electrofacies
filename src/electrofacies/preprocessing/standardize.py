"""Mnemonic and facies label standardization.

Provides utilities to map vendor-specific well-log mnemonics to the canonical
internal names defined in :pymod:`electrofacies.io.schemas`, and to
canonicalize raw facies labels using the alias table in *facies_schema.yaml*.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import yaml

from electrofacies.io.schemas import CANONICAL_LOGS, DEPTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mnemonic helpers
# ---------------------------------------------------------------------------


def load_mnemonic_map(config_path: Union[str, Path]) -> Dict[str, str]:
    """Load *mnemonic_aliases.yaml* and build a reverse alias-to-canonical map.

    Parameters
    ----------
    config_path : str or Path
        Absolute or relative path to *mnemonic_aliases.yaml*.

    Returns
    -------
    dict
        ``{alias_upper: canonical_name}`` -- every alias (upper-cased) maps to
        the canonical mnemonic string.  The canonical name itself is also
        included as a key so that already-standard columns pass through.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    reverse_map: Dict[str, str] = {}
    for canonical_name, aliases in raw.items():
        # Skip non-mnemonic sections (e.g. 'display')
        if not isinstance(aliases, list):
            continue
        # Include the canonical name itself
        reverse_map[canonical_name.upper()] = canonical_name
        for alias in aliases:
            key = str(alias).strip().upper()
            if key in reverse_map and reverse_map[key] != canonical_name:
                logger.warning(
                    "Duplicate alias '%s' maps to both '%s' and '%s'; "
                    "keeping first mapping to '%s'.",
                    key,
                    reverse_map[key],
                    canonical_name,
                    reverse_map[key],
                )
                continue
            reverse_map[key] = canonical_name

    logger.info(
        "Loaded mnemonic map with %d aliases across %d canonical names.",
        len(reverse_map),
        len({v for v in reverse_map.values()}),
    )
    return reverse_map


def standardize_columns(
    df: pd.DataFrame,
    mnemonic_map: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Rename DataFrame columns to canonical log mnemonics.

    Matching is case-insensitive.  Columns that do not match any alias are
    left unchanged and recorded as ``'unmapped'`` in the report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw well-log DataFrame (depth in index or as a column).
    mnemonic_map : dict
        Alias-to-canonical mapping as returned by :func:`load_mnemonic_map`.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(renamed_df, mapping_report)`` where *mapping_report* is
        ``{original_col: canonical_name | 'unmapped'}``.
    """
    if df.empty:
        logger.warning("standardize_columns received an empty DataFrame.")
        return df.copy(), {col: "unmapped" for col in df.columns}

    rename_map: Dict[str, str] = {}
    mapping_report: Dict[str, str] = {}
    seen_canonical: Dict[str, str] = {}  # canonical -> first original col

    for col in df.columns:
        key = str(col).strip().upper()
        canonical = mnemonic_map.get(key)
        if canonical is not None:
            if canonical in seen_canonical:
                logger.warning(
                    "Multiple columns map to '%s': '%s' and '%s'. "
                    "Keeping first ('%s').",
                    canonical,
                    seen_canonical[canonical],
                    col,
                    seen_canonical[canonical],
                )
                mapping_report[col] = "unmapped"
                continue
            rename_map[col] = canonical
            mapping_report[col] = canonical
            seen_canonical[canonical] = col
        else:
            mapping_report[col] = "unmapped"

    renamed_df = df.rename(columns=rename_map)

    n_mapped = sum(1 for v in mapping_report.values() if v != "unmapped")
    logger.info(
        "Standardized %d / %d columns to canonical names.",
        n_mapped,
        len(df.columns),
    )
    return renamed_df, mapping_report


# ---------------------------------------------------------------------------
# Facies label canonicalization
# ---------------------------------------------------------------------------

# Pre-compiled regex for normalizing label strings
_NORMALIZE_RE = re.compile(r"[\s\-]+")


def _normalize_label(raw: str) -> str:
    """Lower-case, collapse whitespace/hyphens/underscores to single ``_``."""
    text = str(raw).strip().lower()
    # Replace hyphens and whitespace with underscores
    text = _NORMALIZE_RE.sub("_", text)
    # Collapse repeated underscores
    text = re.sub(r"_+", "_", text)
    # Strip leading/trailing underscores
    text = text.strip("_")
    return text


def _load_facies_aliases(facies_config: Dict[str, Any]) -> Dict[str, str]:
    """Build ``{normalized_alias: canonical_name}`` from *facies_schema.yaml*."""
    label_aliases: Dict[str, list] = facies_config.get("label_aliases", {})
    lookup: Dict[str, str] = {}
    for canonical_name, aliases in label_aliases.items():
        norm_canonical = _normalize_label(canonical_name)
        lookup[norm_canonical] = canonical_name
        if aliases is None:
            continue
        for alias in aliases:
            norm_alias = _normalize_label(alias)
            if norm_alias in lookup and lookup[norm_alias] != canonical_name:
                logger.warning(
                    "Facies alias '%s' maps to both '%s' and '%s'; "
                    "keeping '%s'.",
                    norm_alias,
                    lookup[norm_alias],
                    canonical_name,
                    lookup[norm_alias],
                )
                continue
            lookup[norm_alias] = canonical_name
    return lookup


def _heuristic_facies_match(
    normalized: str,
    facies_names: list[str],
) -> Optional[str]:
    """Attempt a best-effort match for an unknown label.

    Heuristics applied (in order):
    1. Check if *normalized* starts with or is contained in any canonical name.
    2. Check if any canonical name starts with *normalized*.
    3. Token-overlap scoring: pick the canonical name sharing the most tokens.

    Returns ``None`` if no plausible match is found.
    """
    if not normalized:
        return None

    # 1. Substring containment
    for name in facies_names:
        norm_name = _normalize_label(name)
        if normalized in norm_name or norm_name in normalized:
            logger.info(
                "Heuristic facies match: '%s' -> '%s' (substring).",
                normalized,
                name,
            )
            return name

    # 2. Prefix match
    for name in facies_names:
        norm_name = _normalize_label(name)
        if norm_name.startswith(normalized) or normalized.startswith(norm_name):
            logger.info(
                "Heuristic facies match: '%s' -> '%s' (prefix).",
                normalized,
                name,
            )
            return name

    # 3. Token overlap
    tokens_input = set(normalized.split("_"))
    best_name: Optional[str] = None
    best_overlap = 0
    for name in facies_names:
        tokens_name = set(_normalize_label(name).split("_"))
        overlap = len(tokens_input & tokens_name)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
    if best_overlap > 0:
        logger.info(
            "Heuristic facies match: '%s' -> '%s' (token overlap=%d).",
            normalized,
            best_name,
            best_overlap,
        )
        return best_name

    return None


def canonicalize_facies_labels(
    series: pd.Series,
    facies_config: Dict[str, Any],
) -> pd.Series:
    """Map raw facies labels to canonical names using the alias table.

    Parameters
    ----------
    series : pd.Series
        Series of raw facies label strings.
    facies_config : dict
        Parsed contents of *facies_schema.yaml*.

    Returns
    -------
    pd.Series
        Series with labels replaced by their canonical equivalents.
        Labels that cannot be matched (even heuristically) are left as-is
        and a warning is logged.
    """
    if series.empty:
        logger.warning("canonicalize_facies_labels received an empty Series.")
        return series.copy()

    alias_lookup = _load_facies_aliases(facies_config)

    # All canonical facies names defined in the schema
    facies_names: list[str] = list(facies_config.get("facies", {}).keys())
    # Also include excluded labels
    excluded: list[str] = facies_config.get("excluded_labels", [])
    all_known_names = facies_names + excluded

    cache: Dict[str, str] = {}
    unmatched: set[str] = set()

    def _resolve(raw_val: Any) -> Any:
        if pd.isna(raw_val):
            return raw_val

        raw_str = str(raw_val)
        if raw_str in cache:
            return cache[raw_str]

        normalized = _normalize_label(raw_str)

        # Direct alias lookup
        canonical = alias_lookup.get(normalized)
        if canonical is not None:
            cache[raw_str] = canonical
            return canonical

        # Heuristic fallback
        heuristic = _heuristic_facies_match(normalized, all_known_names)
        if heuristic is not None:
            cache[raw_str] = heuristic
            return heuristic

        # Unresolvable
        if raw_str not in unmatched:
            logger.warning(
                "Facies label '%s' (normalized: '%s') could not be mapped. "
                "Leaving as-is.",
                raw_str,
                normalized,
            )
            unmatched.add(raw_str)
        cache[raw_str] = raw_str
        return raw_str

    result = series.map(_resolve)

    n_unique_mapped = len(cache) - len(unmatched)
    logger.info(
        "Canonicalized facies labels: %d unique values resolved, "
        "%d unmatched.",
        n_unique_mapped,
        len(unmatched),
    )
    return result
