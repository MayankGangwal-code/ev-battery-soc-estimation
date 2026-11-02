"""Error metrics (all in percentage points of SOC)."""
from __future__ import annotations

import numpy as np


def metrics(soc_true, soc_est, skip_s: int = 0) -> dict:
    """RMSE / MAE / max error in % SOC, optionally ignoring the first ``skip_s`` samples."""
    e = 100.0 * (np.asarray(soc_est, float) - np.asarray(soc_true, float))
    e = e[skip_s:] if skip_s < len(e) else e[-1:]
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "max_abs": float(np.max(np.abs(e))),
        "final_err": float(e[-1]),
    }


def convergence_time_s(soc_true, soc_est, tol: float = 0.02) -> int:
    """First time (s) after which the estimate stays within +-tol of the truth."""
    ae = np.abs(np.asarray(soc_est, float) - np.asarray(soc_true, float))
    bad = np.where(ae > tol)[0]
    return int(bad[-1] + 1) if len(bad) else 0


def settling_time_s(soc_true, soc_est, tol: float = 0.02, hold_s: int = 60) -> int:
    """Time (s) until the estimate first enters the +-tol band and stays inside it
    for at least ``hold_s`` consecutive seconds (1 Hz data).  Returns the length
    of the record if that never happens."""
    inside = np.abs(np.asarray(soc_est, float) - np.asarray(soc_true, float)) <= tol
    n = len(inside)
    run = 0
    for k in range(n):
        run = run + 1 if inside[k] else 0
        if run >= hold_s:
            return int(k - hold_s + 1)
    return int(n)
