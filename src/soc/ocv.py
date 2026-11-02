"""Open-circuit voltage (OCV) versus state of charge."""
from __future__ import annotations

import numpy as np
import pandas as pd


class OCVTable:
    """Piecewise-linear OCV(SOC) with its slope and the inverse lookup.

    Outside the measured SOC range the curve is extrapolated linearly with
    the slope of the nearest segment, so the EKF never sees a flat (zero
    slope) region that would make it blind.
    """

    def __init__(self, soc: np.ndarray, ocv: np.ndarray):
        soc, ocv = np.asarray(soc, float), np.asarray(ocv, float)
        order = np.argsort(soc)
        self.soc, self.ocv = soc[order], ocv[order]
        if np.any(np.diff(self.ocv) <= 0):
            # make strictly increasing so that the inverse is well defined
            self.ocv = np.maximum.accumulate(self.ocv + 1e-6 * np.arange(len(self.ocv)))
        self._slope = np.gradient(self.ocv, self.soc)

    @classmethod
    def from_table(cls, table: pd.DataFrame, column: str = "ocv") -> "OCVTable":
        return cls(table["soc"].to_numpy(), table[column].to_numpy())

    def __call__(self, soc) -> np.ndarray:
        soc = np.asarray(soc, float)
        lo, hi = self.soc[0], self.soc[-1]
        v = np.interp(np.clip(soc, lo, hi), self.soc, self.ocv)
        v = np.where(soc < lo, self.ocv[0] + self._slope[0] * (soc - lo), v)
        v = np.where(soc > hi, self.ocv[-1] + self._slope[-1] * (soc - hi), v)
        return v

    def slope(self, soc) -> np.ndarray:
        """dOCV/dSOC in volts per unit SOC (always > 0)."""
        soc = np.asarray(soc, float)
        s = np.interp(np.clip(soc, self.soc[0], self.soc[-1]), self.soc, self._slope)
        return np.maximum(s, 1e-3)

    def inverse(self, voltage) -> np.ndarray:
        """SOC whose OCV equals the given voltage (the naive 'voltage gauge')."""
        voltage = np.asarray(voltage, float)
        lo, hi = self.ocv[0], self.ocv[-1]
        s = np.interp(np.clip(voltage, lo, hi), self.ocv, self.soc)
        s = np.where(voltage < lo, self.soc[0] + (voltage - lo) / self._slope[0], s)
        s = np.where(voltage > hi, self.soc[-1] + (voltage - hi) / self._slope[-1], s)
        return s


class OCVTableSet:
    """OCV tables measured at several temperatures; ``at(T)`` interpolates
    between the two nearest ones on a common SOC grid (the BMS looks up the
    table for the measured cell temperature)."""

    GRID = np.round(np.arange(0.0, 1.0001, 0.005), 3)

    def __init__(self, tables: dict[float, OCVTable]):
        self.temps = np.array(sorted(tables), float)
        self._ocv = np.array([tables[t](self.GRID) for t in self.temps])  # (n_temps, n_grid)

    def _row(self, temp_c: float) -> np.ndarray:
        t = float(np.clip(temp_c, self.temps[0], self.temps[-1]))
        j = int(np.searchsorted(self.temps, t, side="right"))
        if j == 0 or j >= len(self.temps) or self.temps[j - 1] == t:
            return self._ocv[min(max(j - 1, 0), len(self.temps) - 1)]
        w = (t - self.temps[j - 1]) / (self.temps[j] - self.temps[j - 1])
        return (1 - w) * self._ocv[j - 1] + w * self._ocv[j]

    def at(self, temp_c: float) -> OCVTable:
        return OCVTable(self.GRID, self._row(temp_c))

    def tables_for(self, temp_c) -> list[OCVTable]:
        """One OCVTable per sample of ``temp_c`` (cached per 0.5 °C step)."""
        temps = np.round(np.asarray(temp_c, float) * 2) / 2
        cache: dict[float, OCVTable] = {}
        return [cache.setdefault(float(t), OCVTable(self.GRID, self._row(float(t)))) for t in temps]
