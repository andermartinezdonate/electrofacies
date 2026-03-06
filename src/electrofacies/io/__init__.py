"""I/O sub-package for the electrofacies pipeline.

Re-exports the public API from readers, writers, and schemas so callers can
do::

    from electrofacies.io import read_las, WellData, write_predictions_csv
"""

from electrofacies.io.readers import (
    read_csv_well,
    read_las,
    read_training_data,
    scan_wells_folder,
)
from electrofacies.io.schemas import (
    CALI,
    CANONICAL_LOGS,
    DEPTH,
    DTC,
    GR,
    NPHI,
    PE,
    RESD,
    RHOB,
    SP,
    TRAINING_COLUMNS,
    TRAINING_DEPTH,
    TRAINING_FORMATION,
    TRAINING_TARGET,
    PredictionResult,
    WellData,
)
from electrofacies.io.writers import (
    write_batch_summary,
    write_predictions_csv,
    write_predictions_las,
    write_well_report,
)

__all__ = [
    # readers
    "read_las",
    "read_csv_well",
    "read_training_data",
    "scan_wells_folder",
    # writers
    "write_predictions_csv",
    "write_predictions_las",
    "write_well_report",
    "write_batch_summary",
    # schemas - constants
    "DEPTH",
    "GR",
    "RESD",
    "RHOB",
    "NPHI",
    "DTC",
    "PE",
    "CALI",
    "SP",
    "CANONICAL_LOGS",
    "TRAINING_COLUMNS",
    "TRAINING_TARGET",
    "TRAINING_FORMATION",
    "TRAINING_DEPTH",
    # schemas - dataclasses
    "WellData",
    "PredictionResult",
]
