"""Extended Kalman Filter for SOC, built on the 1-RC equivalent-circuit model.

State x = [SOC, V1].  Every second the filter

1. *predicts* the state with the model (Coulomb counting for SOC, the RC
   equation for V1) and grows its uncertainty P, then
2. *corrects* the state with the measured terminal voltage: the difference
   between the measured and the predicted voltage is multiplied by the Kalman
   gain K and added to the state.

Because the correction uses the voltage, the filter recovers from a wrong
initial SOC and does not drift with a biased current sensor — the two
weaknesses of plain Coulomb counting.

Reference: G. L. Plett, "Extended Kalman filtering for battery management
systems of LiPB-based HEV battery packs", J. Power Sources 134 (2004).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import C_REF_AH
from .ecm import ECMParams, ECMTable
from .ocv import OCVTable, OCVTableSet


@dataclass
class EKFSettings:
    # Defaults = best of the grid search in scripts/tune_ekf.py (training cycles only)
    q_soc: float = 1e-9  # process-noise variance of SOC per step (current-sensor error)
    q_v1: float = 1e-5  # process-noise variance of V1 per step (model error)
    r_v: float = 1e-5  # measurement-noise variance of the voltage ((3 mV)^2)
    p0_soc: float = 0.04  # initial SOC variance ((20 %)^2: we admit we do not know it)
    p0_v1: float = 1e-3


def ekf_soc(t_s, i_a, v_meas, params, ocv, soc0: float, temp_c=None, capacity_ah: float = C_REF_AH, settings: EKFSettings | None = None):
    """Run the EKF over a measurement sequence.

    Parameters
    ----------
    i_a : current *as measured by the BMS* (including any sensor error); the
          voltage feedback is what corrects for it
    params : an ``ECMParams`` (fixed model) or an ``ECMTable`` — in the latter
          case ``temp_c`` (measured cell temperature per sample) must be given and
          the model parameters follow the temperature
    ocv : an ``OCVTable`` (fixed) or an ``OCVTableSet`` (the table follows the
          measured temperature sample by sample; needs ``temp_c``)
    Returns ``(soc_hat, soc_std)``: the estimate and its 1-sigma uncertainty.
    """
    s = settings or EKFSettings()
    t_s, i_a, v_meas = (np.asarray(a, float) for a in (t_s, i_a, v_meas))
    n = len(t_s)
    dt = np.diff(t_s, prepend=t_s[0])
    if n > 1:
        dt[0] = dt[1]

    if isinstance(params, ECMTable):
        if temp_c is None:
            raise ValueError("temp_c is required when params is an ECMTable")
        R0, R1, tau = params.arrays(temp_c)
    else:
        R0, R1, tau = (np.full(n, params.R0), np.full(n, params.R1), np.full(n, params.tau))

    if isinstance(ocv, OCVTableSet):
        if temp_c is None:
            raise ValueError("temp_c is required when ocv is an OCVTableSet")
        ocv_k = ocv.tables_for(temp_c)
    else:
        ocv_k = [ocv] * n

    x = np.array([soc0, 0.0])
    P = np.diag([s.p0_soc, s.p0_v1])
    Q = np.diag([s.q_soc, s.q_v1])
    R = s.r_v
    eye = np.eye(2)
    soc_hat, soc_std = np.empty(n), np.empty(n)

    for k in range(n):
        # ---- predict -----------------------------------------------------
        a = np.exp(-dt[k] / tau[k])
        A = np.array([[1.0, 0.0], [0.0, a]])
        x = np.array([x[0] + dt[k] * i_a[k] / (3600.0 * capacity_ah), a * x[1] + R1[k] * (1.0 - a) * i_a[k]])
        P = A @ P @ A.T + Q
        # ---- correct with the voltage measurement --------------------------
        y_hat = float(ocv_k[k](x[0])) + R0[k] * i_a[k] + x[1]
        H = np.array([[float(ocv_k[k].slope(x[0])), 1.0]])
        S = float((H @ P @ H.T).item()) + R
        K = (P @ H.T) / S  # 2x1
        x = x + (K[:, 0] * (v_meas[k] - y_hat))
        x[0] = float(np.clip(x[0], 0.0, 1.0))  # SOC is physically bounded
        IKH = eye - K @ H
        P = IKH @ P @ IKH.T + K @ K.T * R  # Joseph form: keeps P symmetric positive
        soc_hat[k], soc_std[k] = x[0], np.sqrt(max(P[0, 0], 0.0))
    return soc_hat, soc_std
