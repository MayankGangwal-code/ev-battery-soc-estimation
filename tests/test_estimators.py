"""Unit tests for the estimators, run on a synthetic cell with known parameters.

Run with:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc import C_REF_AH  # noqa: E402
from soc.coulomb import SensorErrors, coulomb_counting  # noqa: E402
from soc.ecm import ECMParams, ECMTable, fit_ecm, rc_voltage, simulate_cell  # noqa: E402
from soc.ekf import ekf_soc  # noqa: E402
from soc.evaluate import convergence_time_s, metrics, settling_time_s  # noqa: E402
from soc.features import FEATURE_NAMES, WINDOWS_S, make_features  # noqa: E402
from soc.ocv import OCVTable, OCVTableSet  # noqa: E402


# --------------------------------------------------------------------------- #
# a synthetic NMC-like cell
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ocv():
    soc = np.linspace(0, 1, 21)
    return OCVTable(soc, 3.0 + 1.2 * soc - 0.4 * np.exp(-6 * soc) + 0.05 * np.sin(4 * soc))


@pytest.fixture(scope="module")
def params():
    return ECMParams(R0=0.02, R1=0.03, tau=60.0)


@pytest.fixture(scope="module")
def drive(ocv, params):
    """A 2 h random load profile (discharge with regen pulses) at 1 Hz."""
    rng = np.random.default_rng(3)
    n = 7200
    t = np.arange(1, n + 1, dtype=float)
    i = -np.abs(rng.normal(0.8, 0.8, n))  # discharge, ~0.9 A on average
    regen = rng.random(n) < 0.08
    i[regen] = rng.uniform(0.5, 3.0, regen.sum())  # some regen
    i = pd.Series(i).rolling(5, min_periods=1).mean().to_numpy()
    soc, v = simulate_cell(t, i, soc0=0.95, params=params, ocv=ocv, v_noise=0.003)
    return t, i, v, soc


# --------------------------------------------------------------------------- #
# OCV table
# --------------------------------------------------------------------------- #
def test_ocv_interpolation_and_inverse(ocv):
    s = np.array([0.1, 0.5, 0.9])
    assert np.allclose(ocv.inverse(ocv(s)), s, atol=1e-6)
    assert (ocv.slope(s) > 0).all()
    assert ocv(1.2) > ocv(1.0) > ocv(0.0) > ocv(-0.1)  # linear extrapolation keeps the order


def test_ocv_table_set_interpolates_between_temperatures():
    a = OCVTable([0, 1], [3.0, 4.0])
    b = OCVTable([0, 1], [3.2, 4.2])
    s = OCVTableSet({0.0: a, 20.0: b})
    assert np.isclose(s.at(10.0)(0.5), 3.6)
    assert np.isclose(s.at(-50.0)(0.5), 3.5)  # clamped below the range
    assert np.isclose(s.at(80.0)(0.5), 3.7)  # clamped above


# --------------------------------------------------------------------------- #
# Coulomb counting
# --------------------------------------------------------------------------- #
def test_coulomb_counting_constant_current():
    t = np.arange(1, 3601, dtype=float)  # one hour
    i = np.full(len(t), -1.39)  # 0.5 C discharge of a 2.78 Ah cell
    soc = coulomb_counting(t, i, soc0=1.0, capacity_ah=2.78)
    assert np.isclose(soc[-1], 0.5, atol=1e-6)


def test_coulomb_counting_ideal_matches_simulation(drive):
    t, i, v, soc = drive
    # the simulated cell started at 95 % before the first sample
    assert np.abs(coulomb_counting(t, i, soc0=0.95) - soc).max() < 1e-9


def test_coulomb_counting_drifts_with_offset(drive):
    t, i, v, soc = drive
    est = coulomb_counting(t, i, soc0=0.95, errors=SensorErrors(offset_a=0.05))
    expected_drift = 0.05 * 7200 / 3600 / C_REF_AH  # 3.6 % after 2 h
    assert np.isclose(est[-1] - soc[-1], expected_drift, atol=1e-6)


# --------------------------------------------------------------------------- #
# ECM
# --------------------------------------------------------------------------- #
def test_rc_voltage_steady_state_and_decay():
    t = np.arange(1, 601, dtype=float)
    i = np.full(600, 2.0)
    v1 = rc_voltage(t, i, R1=0.03, tau=60.0)
    assert np.isclose(v1[-1], 0.06, atol=1e-4)  # R1 * I after 10 tau
    assert np.isclose(v1[59], 0.06 * (1 - np.exp(-1)), atol=2e-3)  # 63 % after one tau
    # uniform (filter) and non-uniform (loop) code paths agree
    t2 = t.copy()
    t2[300] += 1e-6
    assert np.allclose(rc_voltage(t2, i, 0.03, 60.0), v1, atol=1e-6)


def test_fit_ecm_recovers_parameters(drive, ocv, params):
    t, i, v, soc = drive
    p, rms = fit_ecm(t, i, v, soc, ocv)
    assert abs(p.R0 - params.R0) < 0.002
    assert abs(p.R1 - params.R1) < 0.005
    assert abs(p.tau - params.tau) / params.tau < 0.25
    assert rms < 0.005


def test_ecm_table_interpolates_in_log_space():
    tab = ECMTable([0.0, 20.0], [ECMParams(0.04, 0.04, 100), ECMParams(0.01, 0.01, 25)])
    mid = tab.at(10.0)
    assert np.isclose(mid.R0, 0.02) and np.isclose(mid.tau, 50)  # geometric mean
    assert np.isclose(tab.at(-30).R0, 0.04) and np.isclose(tab.at(60).R0, 0.01)


# --------------------------------------------------------------------------- #
# EKF
# --------------------------------------------------------------------------- #
def test_ekf_converges_from_wrong_start_with_biased_sensor(drive, ocv, params):
    t, i, v, soc = drive
    i_meas = SensorErrors(offset_a=0.05, gain=1.02).apply(i)
    est, sd = ekf_soc(t, i_meas, v, params, ocv, soc0=soc[0] - 0.3)
    assert settling_time_s(soc, est, tol=0.02) < 600
    assert est.min() >= 0.0 and est.max() <= 1.0  # physically bounded
    m = metrics(soc, est, skip_s=600)
    assert m["rmse"] < 1.0 and m["max_abs"] < 2.5
    # the drifting Coulomb counter with the same sensor is far worse
    cc = coulomb_counting(t, i, soc0=soc[0] - 0.3, errors=SensorErrors(offset_a=0.05, gain=1.02))
    assert metrics(soc, cc, skip_s=600)["rmse"] > 10 * m["rmse"]
    assert (sd > 0).all() and sd[-1] < sd[0]


def test_ekf_accepts_temperature_tables(drive, ocv, params):
    t, i, v, soc = drive
    tab = ECMTable([0.0, 30.0], [params, params])
    ocvs = OCVTableSet({0.0: ocv, 30.0: ocv})
    temp = np.full(len(t), 15.0)
    a, _ = ekf_soc(t, i, v, params, ocv, soc0=soc[0])
    b, _ = ekf_soc(t, i, v, tab, ocvs, soc0=soc[0], temp_c=temp)
    assert np.allclose(a, b, atol=2e-3)
    with pytest.raises(ValueError):
        ekf_soc(t, i, v, tab, ocv, soc0=soc[0])


# --------------------------------------------------------------------------- #
# features / metrics
# --------------------------------------------------------------------------- #
def test_features_are_causal():
    df = pd.DataFrame({"V": np.linspace(4.0, 3.0, 100), "I": np.zeros(100), "T": np.full(100, 25.0)})
    f = make_features(df)
    assert list(f.columns) == FEATURE_NAMES and len(FEATURE_NAMES) == 3 + 2 * len(WINDOWS_S)
    # changing a future sample must not change earlier features
    df2 = df.copy()
    df2.loc[50, "V"] = 2.0
    f2 = make_features(df2)
    assert np.allclose(f.iloc[:50].to_numpy(), f2.iloc[:50].to_numpy())
    assert not np.allclose(f.iloc[50:60].to_numpy(), f2.iloc[50:60].to_numpy())


def test_metrics_units():
    truth = np.array([0.5, 0.5, 0.5])
    m = metrics(truth, np.array([0.5, 0.52, 0.47]))
    assert np.isclose(m["mae"], (0 + 2 + 3) / 3)
    assert np.isclose(m["max_abs"], 3.0) and np.isclose(m["final_err"], -3.0)
    assert convergence_time_s(truth, np.array([0.6, 0.55, 0.51]), tol=0.02) == 2
    est = np.array([0.6] * 5 + [0.51] * 70 + [0.6] * 3 + [0.5] * 100)
    assert settling_time_s(truth[:1].repeat(178), est, tol=0.02, hold_s=60) == 5
    assert settling_time_s(truth[:1].repeat(178), est, tol=0.02, hold_s=80) == 78
