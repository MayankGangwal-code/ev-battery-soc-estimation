"""Coulomb counting (ampere-hour integration) — the textbook SOC estimator.

    SOC(t) = SOC(0) + (1 / (3600 C)) * integral of I dt        (I > 0 = charging)

It is exact with a perfect current sensor and a known starting point, and it
drifts without bound otherwise, because every sensor error is integrated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import C_REF_AH


@dataclass
class SensorErrors:
    """Imperfections of a real BMS current measurement."""

    offset_a: float = 0.0  # constant bias in amps (e.g. Hall-effect sensor offset)
    gain: float = 1.0  # multiplicative gain error (1.02 = reads 2 % high)
    noise_a: float = 0.0  # white noise standard deviation in amps
    seed: int = 0

    def apply(self, i: np.ndarray) -> np.ndarray:
        i = self.gain * np.asarray(i, float) + self.offset_a
        if self.noise_a > 0:
            i = i + np.random.default_rng(self.seed).normal(0, self.noise_a, size=len(i))
        return i


def coulomb_counting(t_s, i_a, soc0: float, capacity_ah: float = C_REF_AH, errors: SensorErrors | None = None) -> np.ndarray:
    """SOC trajectory from current integration.

    Parameters
    ----------
    t_s : time stamps in seconds (need not be uniform)
    i_a : current in amps, positive when charging
    soc0 : assumed initial SOC
    capacity_ah : capacity used by the estimator (may differ from the truth)
    errors : optional current-sensor errors to apply before integrating
    """
    t_s = np.asarray(t_s, float)
    i_meas = errors.apply(i_a) if errors else np.asarray(i_a, float)
    dt = np.diff(t_s, prepend=t_s[0])
    if len(dt) > 1:
        dt[0] = dt[1]  # each sample is the mean over the interval that ends at t_s
    return soc0 + np.cumsum(i_meas * dt) / (3600.0 * capacity_ah)
