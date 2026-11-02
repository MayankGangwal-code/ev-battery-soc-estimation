"""Tests on the processed data and the saved models (skipped until the
pipeline scripts have been run)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc import C_REF_AH, TEMP_ORDER, TEST_CYCLES, TRAIN_CYCLES  # noqa: E402
from soc.bms import CELL_MODEL_PATH, ML_MODELS_PATH, CellModel, estimate_all, load_ml_models  # noqa: E402
from soc.coulomb import SensorErrors  # noqa: E402
from soc.data import DATA_DIR, list_cycles, load_cycle, load_hppc, load_index  # noqa: E402
from soc.evaluate import metrics  # noqa: E402

needs_data = pytest.mark.skipif(not (DATA_DIR / "cycles_index.csv").exists(), reason="run scripts/prepare_data.py first")
needs_models = pytest.mark.skipif(not (CELL_MODEL_PATH.exists() and ML_MODELS_PATH.exists()), reason="run scripts/run_experiments.py first")


@needs_data
def test_index_and_splits():
    idx = load_index()
    valid = idx[idx["valid"]]
    assert set(valid["split"]) == {"train", "test"}
    assert set(valid[valid.split == "train"]["cycle"]) <= set(TRAIN_CYCLES)
    assert set(valid[valid.split == "test"]["cycle"]) <= set(TEST_CYCLES)
    assert len(list_cycles("test")) >= 20


@needs_data
@pytest.mark.parametrize("temp,cycle", [("25degC", "UDDS"), ("n20degC", "US06"), ("0degC", "LA92")])
def test_cycle_files_are_consistent(temp, cycle):
    d = load_cycle(temp, cycle)
    assert list(d.columns) == ["t_s", "V", "I", "T", "Ah", "SOC"]
    assert (np.diff(d["t_s"]) == 1).all() and not d.isna().any().any()
    assert np.allclose(d["SOC"], 1 + d["Ah"] / C_REF_AH, atol=1e-4)
    assert 0.97 < d["SOC"].iloc[0] <= 1.001  # starts fully charged
    # the cycler's Ah counter and the 1 Hz current agree (ideal Coulomb counting)
    cc = d["SOC"].iloc[0] - d["I"].iloc[0] / 3600 / C_REF_AH + np.cumsum(d["I"]) / 3600 / C_REF_AH
    assert np.abs(cc - d["SOC"]).max() < 0.005


@needs_data
def test_hppc_tables_physically_sensible():
    r0 = {t: load_hppc(t)["R0_ohm"].median() for t in TEMP_ORDER}
    assert r0["n20degC"] > r0["0degC"] > r0["25degC"] > 0.005  # colder = more resistive
    for t in TEMP_ORDER:
        h = load_hppc(t).dropna(subset=["ocv_rest"])
        assert (np.diff(h.sort_values("soc")["ocv_rest"]) > 0).all()  # OCV increases with SOC


@needs_models
def test_cell_model_roundtrip_and_estimators():
    cell = CellModel.from_json()
    assert cell.capacity_ah == C_REF_AH and len(cell.ecm.temps) == 6
    d = load_cycle("25degC", "LA92")
    est = estimate_all(d, cell, load_ml_models(), soc0_guess=d["SOC"].iloc[0] - 0.1, sensor=SensorErrors(offset_a=0.05, gain=1.02))
    truth = d["SOC"].to_numpy()
    for name, limit in [("EKF", 2.0), ("Neural Network", 2.0), ("Gradient Boosting", 3.0)]:
        assert metrics(truth, est[name])["rmse"] < limit, name
    assert metrics(truth, est["Coulomb counting"])["rmse"] > 5.0  # drifts with the biased sensor
    assert (est["EKF_std"] > 0).all()
