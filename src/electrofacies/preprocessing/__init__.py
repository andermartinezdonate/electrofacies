"""Preprocessing sub-package for the electrofacies pipeline.

Exports the main public functions and classes from each module for
convenient access::

    from electrofacies.preprocessing import (
        load_mnemonic_map,
        standardize_columns,
        canonicalize_facies_labels,
        validate_well,
        engineer_features,
        FaciesTransformer,
        handle_missing,
        winsorize,
    )
"""

from electrofacies.preprocessing.standardize import (
    canonicalize_facies_labels,
    load_mnemonic_map,
    standardize_columns,
)
from electrofacies.preprocessing.validate import (
    check_null_coverage,
    detect_flatlines,
    detect_washouts,
    load_physical_ranges,
    validate_ranges,
    validate_well,
)
from electrofacies.preprocessing.features import (
    build_feature_columns,
    compute_training_stats,
    engineer_features,
)
from electrofacies.preprocessing.transform import (
    FaciesTransformer,
    handle_missing,
    winsorize,
)

__all__ = [
    # standardize
    "load_mnemonic_map",
    "standardize_columns",
    "canonicalize_facies_labels",
    # validate
    "load_physical_ranges",
    "validate_ranges",
    "detect_washouts",
    "detect_flatlines",
    "check_null_coverage",
    "validate_well",
    # features
    "compute_training_stats",
    "build_feature_columns",
    "engineer_features",
    # transform
    "FaciesTransformer",
    "winsorize",
    "handle_missing",
]
