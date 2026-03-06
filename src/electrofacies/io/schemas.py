"""Canonical schemas, column constants, and dataclasses for the electrofacies pipeline.

Defines the authoritative internal naming conventions for well log mnemonics,
training column mappings, and data containers used throughout the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical log mnemonic constants
# ---------------------------------------------------------------------------
DEPTH: str = "DEPTH"
GR: str = "GR"
RESD: str = "RESD"
RHOB: str = "RHOB"
NPHI: str = "NPHI"
DTC: str = "DTC"
PE: str = "PE"
CALI: str = "CALI"
SP: str = "SP"

CANONICAL_LOGS: List[str] = [GR, RESD, RHOB, NPHI, DTC, PE, CALI, SP]
"""Ordered list of canonical log mnemonics (excludes DEPTH)."""

# ---------------------------------------------------------------------------
# Training column mapping
# ---------------------------------------------------------------------------
# Maps the raw column names present in the PDB03 training spreadsheet to
# their canonical internal names.
TRAINING_COLUMNS: Dict[str, str] = {
    "Depth Top (ft)": DEPTH,
    "GR (API)": GR,
    "RESD (ohm.m)": RESD,
    "RHOB (g/cm3)": RHOB,
    "NPHI (ft3/ft3)": NPHI,
    "Sonic (DTC us/ft)": DTC,
}

# The target and grouping columns in the training data.
TRAINING_TARGET: str = "Lithofacies"
TRAINING_FORMATION: str = "Formation"
TRAINING_DEPTH: str = "Depth Top (ft)"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WellData:
    """Container for a single well's data after reading and normalisation.

    Attributes
    ----------
    well_name : str
        Human-readable well name (from LAS header or filename).
    uwi : str
        Unique Well Identifier / API number.  Empty string if unavailable.
    depth_unit : str
        Unit of the depth index after loading (always ``"ft"`` after
        conversion).
    curves : pd.DataFrame
        Log curves indexed by depth.  Column names are canonical mnemonics
        where mapping succeeded, raw names otherwise.
    metadata : dict
        Additional header information (KB, location, etc.).
    source_path : str
        Absolute path to the original file on disk.
    """

    well_name: str
    uwi: str
    depth_unit: str
    curves: pd.DataFrame
    metadata: Dict = field(default_factory=dict)
    source_path: str = ""


@dataclass
class PredictionResult:
    """Container for facies predictions produced for a single well.

    Attributes
    ----------
    well_name : str
        Human-readable well name.
    uwi : str
        Unique Well Identifier / API number.
    predictions : pd.DataFrame
        DataFrame indexed by depth with at minimum a ``"FACIES"`` column and
        optionally per-class probability columns.
    tier_used : str
        Identifier of the model tier applied (e.g. ``"tier_1"``).
    algorithm : str
        Name of the algorithm used (e.g. ``"random_forest"``).
    qc_summary : dict
        Quality-control summary produced during inference (coverage, OOD
        fraction, confidence statistics, etc.).
    metadata : dict
        Any extra metadata to persist (run timestamp, config hash, etc.).
    """

    well_name: str
    uwi: str
    predictions: pd.DataFrame
    tier_used: str = ""
    algorithm: str = ""
    qc_summary: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
