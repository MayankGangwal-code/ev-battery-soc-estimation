r"""First-order (1-RC) Thevenin equivalent-circuit model of the cell.

              R1
         ┌───/\/\/───┐
  ──R0───┤           ├───┬── +  terminal
         └────||─────┘   │
               C1        │
   OCV(SOC) ─────────────┘

    V_terminal = OCV(SOC) + R0 * I + V1
    dV1/dt     = -V1 / (R1 C1) + I / C1          (I > 0 = charging)

R0 is the instantaneous ohmic resistance, the R1-C1 pair reproduces the slow
voltage relaxation (charge-transfer and diffusion) with time constant
tau = R1 C1.  All three parameters grow strongly as the cell gets colder, so
the BMS keeps a table of them versus temperature (``ECMTable``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from . import C_REF_AH
from .ocv import OCVTable


@dataclass
class ECMParams:
    R0: float  # ohm
    R1: float  # ohm
    tau: float  # seconds (= R1 * C1)

    @property
    def C1(self) -> float:
        return self.tau / self.R1

    def as_dict(self) -> dict:
        return {**asdict(self), "C1": self.C1}


class ECMTable:
    """ECM parameters as a function of cell temperature.

    Parameters are fitted at a few temperatures; in between they are
    interpolated in log space (resistance roughly follows an Arrhenius law),
    and clamped outside the fitted range.
    """

    def __init__(self, temps_c, params: list[ECMParams]):
        order = np.argsort(temps_c)
        self.temps = np.asarray(temps_c, float)[order]
        self.params = [params[k] for k in order]
        self._log = np.log(np.array([[p.R0, p.R1, p.tau] for p in self.params]))

    @classmethod
    def from_dict(cls, d: dict) -> "ECMTable":
        temps = [v["fit_temp_C"] for v in d.values()]
        params = [ECMParams(R0=v["R0"], R1=v["R1"], tau=v["tau"]) for v in d.values()]
        return cls(temps, params)

    def at(self, temp_c: float) -> ECMParams:
        t = float(np.clip(temp_c, self.temps[0], self.temps[-1]))
        r0, r1, tau = np.exp([np.interp(t, self.temps, self._log[:, k]) for k in range(3)])
        return ECMParams(R0=float(r0), R1=float(r1), tau=float(tau))

    def arrays(self, temp_c) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised R0, R1, tau for an array of temperatures."""
        t = np.clip(np.asarray(temp_c, float), self.temps[0], self.temps[-1])
        return tuple(np.exp(np.interp(t, self.temps, self._log[:, k])) for k in range(3))


def rc_voltage(t_s, i_a, R1: float, tau: float, v1_0: float = 0.0) -> np.ndarray:
    """Voltage across the RC pair, by exact discretisation (zero-order hold)."""
    t_s, i_a = np.asarray(t_s, float), np.asarray(i_a, float)
    dt = np.diff(t_s, prepend=t_s[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    a = np.exp(-dt / tau)
    if np.allclose(dt, dt[0]):  # uniform sampling: use a fast IIR filter
        from scipy.signal import lfilter

        b0 = R1 * (1.0 - a[0])
        zi = np.array([a[0] * v1_0])  # initial state so that the first output uses v1_0
        v1, _ = lfilter([b0], [1.0, -a[0]], i_a, zi=zi)
        return v1
    v1 = np.empty(len(i_a))
    v = v1_0
    for k in range(len(i_a)):
        v = a[k] * v + R1 * (1.0 - a[k]) * i_a[k]
        v1[k] = v
    return v1


def simulate_terminal_voltage(t_s, i_a, soc, params: ECMParams, ocv: OCVTable) -> np.ndarray:
    """Model terminal voltage given the (true) SOC trajectory."""
    return ocv(soc) + params.R0 * np.asarray(i_a, float) + rc_voltage(t_s, i_a, params.R1, params.tau)


def fit_ecm(t_s, i_a, v_meas, soc, ocv: OCVTable, r0_guess: float = 0.02) -> tuple[ECMParams, float]:
    """Least-squares fit of R0, R1, tau on a drive cycle with known SOC.

    Returns the parameters and the RMS voltage error of the fit in volts.
    """
    t_s, i_a, v_meas, soc = (np.asarray(x, float) for x in (t_s, i_a, v_meas, soc))
    residual_target = v_meas - ocv(soc)  # what R0*I + V1 has to explain

    def resid(p):
        R0, R1, log_tau = p
        return residual_target - (R0 * i_a + rc_voltage(t_s, i_a, R1, np.exp(log_tau)))

    x0 = [r0_guess, r0_guess, np.log(60.0)]
    res = least_squares(resid, x0, bounds=([1e-4, 1e-4, np.log(2.0)], [1.0, 2.0, np.log(3600.0)]), max_nfev=200)
    R0, R1, log_tau = res.x
    rms = float(np.sqrt(np.mean(res.fun**2)))
    return ECMParams(R0=float(R0), R1=float(R1), tau=float(np.exp(log_tau))), rms


def simulate_cell(t_s, i_a, soc0: float, params: ECMParams, ocv: OCVTable, capacity_ah: float = C_REF_AH, v_noise: float = 0.0, seed: int = 0):
    """Synthetic cell: returns (soc, v_terminal) for a current profile — used in tests."""
    t_s, i_a = np.asarray(t_s, float), np.asarray(i_a, float)
    dt = np.diff(t_s, prepend=t_s[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    soc = soc0 + np.cumsum(i_a * dt) / (3600.0 * capacity_ah)
    v = simulate_terminal_voltage(t_s, i_a, soc, params, ocv)
    if v_noise > 0:
        v = v + np.random.default_rng(seed).normal(0, v_noise, len(v))
    return soc, v
