"""Causal features for the machine-learning SOC estimators.

A BMS only knows the present and the past, so every feature is computed from
the current sample and a backward-looking window:

* V, I, T                      — the instantaneous measurements
* V_avg_w, I_avg_w             — causal moving averages over w = 10, 60, 300, 900 s

The averaged voltage/current give the model a short memory of the load
history (the same idea as the RC branch of the equivalent-circuit model)
without any recurrent architecture.  No feature uses the Ah counter, so the
estimators need neither an initial SOC nor a drift-free current integral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS_S = [10, 60, 300, 900]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs columns V, I, T sampled at 1 Hz (one row per second)."""
    out = pd.DataFrame({"V": df["V"].to_numpy(float), "I": df["I"].to_numpy(float), "T": df["T"].to_numpy(float)})
    for w in WINDOWS_S:
        out[f"V_avg_{w}"] = out["V"].rolling(w, min_periods=1).mean()
        out[f"I_avg_{w}"] = out["I"].rolling(w, min_periods=1).mean()
    return out


FEATURE_NAMES = list(make_features(pd.DataFrame({"V": [4.0], "I": [0.0], "T": [25.0]})).columns)
