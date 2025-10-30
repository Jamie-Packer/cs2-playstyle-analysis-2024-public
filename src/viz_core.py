"""
Shared visualization utilities used across both Matplotlib and Plotly plotting modules.
"""
from __future__ import annotations
from typing import List

# ========================================================== Common Definitions ==========================================

# Short axis labels 
FEATURE_LABELS_SHORT = {
    "tapd": "TAPD",
    "oap": "OAP",
    "podt": "PODT",
    "pokt": "POKT",
    "adnt_rank": "ADNT",
    "adat_rank": "ADAT",
}

# Full definitions 
FEATURE_LEGEND_FULL = {
    "TAPD": "Time Alive Per Death",
    "OAP":  "Opening Attempt Percentage",
    "PODT": "Proportion of Deaths Traded",
    "POKT": "Proportion of Kills which were Trades",
    "ADNT": "Average Distance from Nearest Teammate",
    "ADAT": "Average Distance from Average Teammate",
}


def _default_features_base() -> List[str]:
    """Return standard feature ordering."""
    return list(FEATURE_LABELS_SHORT.keys())


def _feature_spec(side: str, features=None):
    """
    Return a list of (column_name, pretty_label, xlim) for the six side-specific features.
    """
    s = str(side).lower()
    if s not in {"t", "ct"}:
        raise ValueError("side must be 't' or 'ct'")

    all_feats = ["oap", "podt", "pokt", "tapd", "adnt_rank", "adat_rank"]
    feats = list(features) if features is not None else all_feats

    spec = {
        "oap":       (f"oap_{s}",       "Opening Attempt % (OAP)",                   (0, 45)),
        "podt":      (f"podt_{s}",      "Proportion of Deaths Traded % (PODT)",      (0, 40)),
        "pokt":      (f"pokt_{s}",      "Proportion of Kills that are Trades % (POKT)",       (0, 40)),
        "tapd":      (f"tapd_{s}",      "Time Alive Per Death (TAPD) (s)",           None),
        "adnt_rank": (f"adnt_rank_{s}", "Distance to Nearest Teammate (ADNT) – rank",(0.2, 1.0)),
        "adat_rank": (f"adat_rank_{s}", "Distance from Average Teammate (ADAT) – rank",(0.2, 1.0)),
    }

    out = []
    for f in feats:
        if f not in spec:
            raise KeyError(f"Unknown feature '{f}'. Choose from {list(spec)}.")
        out.append(spec[f])
    return out