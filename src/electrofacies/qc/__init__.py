"""Quality-control and confidence assessment for electrofacies predictions."""

from electrofacies.qc.confidence import (
    calibrate_probabilities,
    compute_confidence_scores,
    compute_entropy,
    compute_margin,
    compute_max_probability,
    flag_low_confidence,
)
from electrofacies.qc.ood import OODDetector
from electrofacies.qc.reports import (
    format_qc_status,
    generate_batch_report,
    generate_well_qc,
)

__all__ = [
    "compute_max_probability",
    "compute_entropy",
    "compute_margin",
    "compute_confidence_scores",
    "flag_low_confidence",
    "calibrate_probabilities",
    "OODDetector",
    "generate_well_qc",
    "generate_batch_report",
    "format_qc_status",
]
