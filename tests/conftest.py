"""Shared test fixtures for the electrofacies test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_well_df():
    """A small synthetic well DataFrame with canonical column names."""
    np.random.seed(42)
    n = 100
    depth = np.arange(5000, 5000 + n * 0.5, 0.5)
    return pd.DataFrame({
        "DEPTH": depth,
        "GR": np.random.uniform(20, 180, n),
        "RESD": np.random.lognormal(2, 1, n),
        "RHOB": np.random.uniform(2.1, 2.75, n),
        "NPHI": np.random.uniform(0.02, 0.35, n),
        "DTC": np.random.uniform(55, 120, n),
    })


@pytest.fixture
def sample_training_df(sample_well_df):
    """Training DataFrame with facies labels."""
    df = sample_well_df.copy()
    classes = [
        "massive_sandstone", "structured_sandstone", "sandy_siltstone",
        "siltstone", "calciturbidite", "clast_supported_conglomerate",
    ]
    df["Lithofacies"] = np.random.choice(classes, len(df))
    df["Formation"] = "DMG"
    df = df.rename(columns={
        "DEPTH": "Depth Top (ft)",
        "GR": "GR (API)",
        "RESD": "RESD (ohm.m)",
        "RHOB": "RHOB (g/cm3)",
        "NPHI": "NPHI (ft3/ft3)",
        "DTC": "Sonic (DTC us/ft)",
    })
    return df


@pytest.fixture
def facies_config():
    """Minimal facies config dict."""
    return {
        "facies": {
            "massive_sandstone": {"code": 0, "color": "#f3e6b3"},
            "structured_sandstone": {"code": 1, "color": "#e6b657"},
            "sandy_siltstone": {"code": 2, "color": "#a7663a"},
            "siltstone": {"code": 3, "color": "#505660"},
            "calciturbidite": {"code": 4, "color": "#2ca25f"},
            "clast_supported_conglomerate": {"code": 5, "color": "#7e3fa0"},
        },
        "excluded_labels": ["missing_strata"],
        "label_aliases": {
            "massive_sandstone": ["massive_sandstone", "massive"],
            "structured_sandstone": ["structured_sandstone", "structured"],
            "sandy_siltstone": ["sandy_siltstone"],
            "siltstone": ["siltstone", "silt"],
            "calciturbidite": ["calciturbidite"],
            "clast_supported_conglomerate": ["clast_supported_conglomerate"],
            "missing_strata": ["missing_strata", "gap"],
        },
    }


@pytest.fixture
def configs_dir():
    """Path to the project configs directory."""
    from pathlib import Path
    return Path(__file__).parent.parent / "configs"
