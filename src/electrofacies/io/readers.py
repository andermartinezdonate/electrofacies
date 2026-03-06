"""Well-data readers for the electrofacies pipeline.

Provides functions for reading LAS files, CSV well exports, and Excel/CSV
training data.  Every reader returns a normalised dict (or DataFrame) so
that downstream code never needs to worry about file format.

All functions use the ``logging`` module for diagnostics; nothing is printed
to stdout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import lasio
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Metres-to-feet conversion factor.
_M_TO_FT: float = 1.0 / 0.3048

# Heuristic tokens that indicate *metre* depth units in LAS headers.
_METRE_TOKENS = {"m", "meter", "meters", "metre", "metres", "m.", "mtr"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_depth_unit(las: lasio.LASFile) -> str:
    """Return ``'ft'`` or ``'m'`` based on the LAS file header.

    The function inspects, in order:
    1. The ``STRT`` parameter's ``unit`` field.
    2. The ``STEP`` parameter's ``unit`` field.
    3. Well-parameter items named ``STRT`` / ``STOP``.

    Falls back to ``'ft'`` when the unit is ambiguous.
    """
    for param_name in ("STRT", "STOP", "STEP"):
        try:
            unit_str = las.well[param_name].unit
            if unit_str and unit_str.strip().lower() in _METRE_TOKENS:
                return "m"
            if unit_str and unit_str.strip().lower() in {"ft", "f", "feet", "foot", "ft."}:
                return "ft"
        except (KeyError, AttributeError):
            continue
    logger.debug("Could not determine depth unit from header; assuming feet.")
    return "ft"


def _extract_header_metadata(las: lasio.LASFile) -> Dict:
    """Pull commonly useful header fields from a parsed LAS file."""
    meta: Dict = {}

    def _safe(section: str, key: str) -> Optional[str]:
        try:
            val = las.well[key].value
            return str(val).strip() if val else None
        except (KeyError, AttributeError):
            return None

    meta["well_name"] = _safe("well", "WELL") or _safe("well", "WN") or ""
    meta["uwi"] = _safe("well", "UWI") or _safe("well", "API") or ""
    meta["kb"] = _safe("well", "EKBE") or _safe("well", "EKB") or _safe("well", "KB") or ""
    meta["company"] = _safe("well", "COMP") or ""
    meta["field"] = _safe("well", "FLD") or ""
    meta["location"] = _safe("well", "LOC") or ""
    meta["county"] = _safe("well", "CNTY") or ""
    meta["state"] = _safe("well", "STAT") or ""
    meta["country"] = _safe("well", "CTRY") or ""
    meta["date"] = _safe("well", "DATE") or ""
    meta["service_company"] = _safe("well", "SRVC") or ""

    # Latitude / longitude (non-standard but common).
    meta["latitude"] = _safe("well", "LATI") or _safe("well", "LAT") or ""
    meta["longitude"] = _safe("well", "LONG") or _safe("well", "LON") or ""

    # Depth range from header.
    try:
        meta["strt"] = float(las.well["STRT"].value)
    except (KeyError, AttributeError, ValueError, TypeError):
        meta["strt"] = None
    try:
        meta["stop"] = float(las.well["STOP"].value)
    except (KeyError, AttributeError, ValueError, TypeError):
        meta["stop"] = None

    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_las(path: Union[str, Path]) -> Dict:
    """Read a LAS file and return normalised well data.

    Parameters
    ----------
    path : str or Path
        Path to the ``.las`` file.

    Returns
    -------
    dict
        Keys:
        - ``'metadata'`` : dict of header fields.
        - ``'curves'``   : ``pd.DataFrame`` indexed by depth (feet).
        - ``'depth_unit'``: always ``'ft'`` after conversion.
        - ``'path'``     : absolute path string.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LAS file not found: {path}")

    logger.info("Reading LAS file: %s", path)

    try:
        las = lasio.read(str(path), ignore_header_errors=True)
    except Exception:
        logger.exception("lasio failed to parse %s", path)
        raise

    # Build curves DataFrame.
    depth_unit = _detect_depth_unit(las)
    df = las.df()  # indexed by depth

    # lasio may return the index as the depth mnemonic; ensure it is named
    # DEPTH internally.
    df.index.name = "DEPTH"

    # Convert metres to feet if necessary.
    if depth_unit == "m":
        logger.info("Converting depth index from metres to feet for %s", path.name)
        df.index = df.index * _M_TO_FT

    # Strip leading/trailing whitespace from column names.
    df.columns = [c.strip() for c in df.columns]

    # Extract metadata.
    metadata = _extract_header_metadata(las)
    metadata["num_curves"] = len(df.columns)
    metadata["curve_names"] = list(df.columns)
    metadata["depth_range"] = (float(df.index.min()), float(df.index.max()))
    metadata["num_samples"] = len(df)

    return {
        "metadata": metadata,
        "curves": df,
        "depth_unit": "ft",
        "path": str(path),
    }


def read_csv_well(path: Union[str, Path]) -> Dict:
    """Read a CSV-formatted well file and return normalised well data.

    The CSV must have a column whose name contains ``'depth'`` (case-
    insensitive) or is named ``'DEPT'``.  That column becomes the DataFrame
    index.

    Parameters
    ----------
    path : str or Path
        Path to the ``.csv`` file.

    Returns
    -------
    dict
        Same structure as :func:`read_las`.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV well file not found: {path}")

    logger.info("Reading CSV well file: %s", path)

    try:
        df = pd.read_csv(path)
    except Exception:
        logger.exception("Failed to parse CSV %s", path)
        raise

    # Identify the depth column.
    depth_col = None
    for col in df.columns:
        if col.strip().upper() in ("DEPTH", "DEPT", "MD"):
            depth_col = col
            break
    if depth_col is None:
        for col in df.columns:
            if "depth" in col.lower():
                depth_col = col
                break
    if depth_col is None:
        raise ValueError(
            f"No depth column found in CSV {path}. "
            f"Columns present: {list(df.columns)}"
        )

    df = df.set_index(depth_col)
    df.index.name = "DEPTH"
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.dropna(subset=[], how="all")  # drop rows where all values are NaN
    df = df[df.index.notna()]

    # Strip whitespace from column names.
    df.columns = [c.strip() for c in df.columns]

    # Heuristic depth-unit detection: if max depth < 2000, likely metres.
    depth_unit = "ft"
    if df.index.max() < 2000 and df.index.max() > 10:
        logger.warning(
            "Max depth %.1f in %s is low; assuming metres and converting to feet.",
            df.index.max(),
            path.name,
        )
        depth_unit = "m"
        df.index = df.index * _M_TO_FT

    # Derive a well name from the filename.
    well_name = path.stem

    metadata: Dict = {
        "well_name": well_name,
        "uwi": "",
        "num_curves": len(df.columns),
        "curve_names": list(df.columns),
        "depth_range": (float(df.index.min()), float(df.index.max())),
        "num_samples": len(df),
    }

    return {
        "metadata": metadata,
        "curves": df,
        "depth_unit": "ft",
        "path": str(path),
    }


def read_training_data(path: Union[str, Path]) -> pd.DataFrame:
    """Read the training data from an Excel or CSV file.

    Supports ``.xlsx``, ``.xls``, and ``.csv`` extensions.  Returns the raw
    DataFrame exactly as stored on disk (no renaming); downstream
    preprocessing is responsible for mapping columns to canonical names.

    Parameters
    ----------
    path : str or Path
        Path to the training file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not recognised.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training data file not found: {path}")

    ext = path.suffix.lower()
    logger.info("Reading training data from %s (format: %s)", path, ext)

    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, engine="openpyxl" if ext == ".xlsx" else None)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported training file extension '{ext}'. "
                "Expected .xlsx, .xls, or .csv."
            )
    except ValueError:
        raise
    except Exception:
        logger.exception("Failed to read training data from %s", path)
        raise

    logger.info(
        "Loaded training data: %d rows, %d columns (%s)",
        len(df),
        len(df.columns),
        ", ".join(df.columns[:8]),
    )
    return df


def scan_wells_folder(
    folder_path: Union[str, Path],
    extensions: Optional[List[str]] = None,
) -> List[Dict]:
    """Recursively scan a folder for well data files.

    Parameters
    ----------
    folder_path : str or Path
        Root directory to scan.
    extensions : list of str, optional
        File extensions to include (case-insensitive).  Defaults to
        ``['.las', '.LAS', '.csv']``.

    Returns
    -------
    list of dict
        Each dict contains ``'path'`` (str), ``'extension'`` (str), and
        ``'filename'`` (str).
    """
    if extensions is None:
        extensions = [".las", ".LAS", ".csv"]

    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        logger.warning("Wells folder does not exist: %s", folder)
        return []

    # Normalise extensions to lower-case for matching.
    ext_set = {e.lower() for e in extensions}

    results: List[Dict] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in ext_set:
            results.append(
                {
                    "path": str(p),
                    "extension": p.suffix,
                    "filename": p.name,
                }
            )

    logger.info(
        "Scanned %s: found %d well file(s) with extensions %s",
        folder,
        len(results),
        ext_set,
    )
    return results
