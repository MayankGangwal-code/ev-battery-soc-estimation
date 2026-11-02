"""The "BMS": everything needed to estimate SOC from a stream of (t, V, I, T).

``CellModel`` bundles the characterisation of the cell (OCV tables and
equivalent-circuit parameters at several temperatures) and is saved to
``models/cell_model.json`` by ``scripts/run_experiments.py``.  ``estimate_all``
runs every estimator on a measurement log, exactly as the app and the CLI do.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import C_REF_AH, __version__
from .coulomb import SensorErrors, coulomb_counting
from .ecm import ECMParams, ECMTable
from .ekf import EKFSettings, ekf_soc
from .features import make_features
from .ocv import OCVTable, OCVTableSet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CELL_MODEL_PATH = PROJECT_ROOT / "models" / "cell_model.json"
ML_MODELS_PATH = PROJECT_ROOT / "models" / "ml_models.joblib"


class CellModel:
    def __init__(self, ocv_tables: dict[float, OCVTable], ecm_table: ECMTable, capacity_ah: float = C_REF_AH, meta: dict | None = None):
        self.ocv_set = OCVTableSet(ocv_tables)
        self.ocv_tables = ocv_tables
        self.ecm = ecm_table
        self.capacity_ah = capacity_ah
        self.meta = meta or {}

    # ---- persistence ------------------------------------------------------
    def to_json(self, path: Path | str = CELL_MODEL_PATH) -> Path:
        d = {
            "package_version": __version__,
            "capacity_ah": self.capacity_ah,
            "ocv_tables": {str(t): {"soc": tab.soc.tolist(), "ocv": tab.ocv.tolist()} for t, tab in self.ocv_tables.items()},
            "ecm": [{"temp_C": float(t), "R0": p.R0, "R1": p.R1, "tau": p.tau, "C1": p.C1} for t, p in zip(self.ecm.temps, self.ecm.params)],
            "meta": self.meta,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=1))
        return path

    @classmethod
    def from_json(cls, path: Path | str = CELL_MODEL_PATH) -> "CellModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `python scripts/run_experiments.py` first")
        d = json.loads(path.read_text())
        ocv = {float(t): OCVTable(v["soc"], v["ocv"]) for t, v in d["ocv_tables"].items()}
        ecm = ECMTable([e["temp_C"] for e in d["ecm"]], [ECMParams(e["R0"], e["R1"], e["tau"]) for e in d["ecm"]])
        return cls(ocv, ecm, d["capacity_ah"], d.get("meta"))

    # ---- estimators ---------------------------------------------------------
    def ocv_lookup(self, v, temp_c) -> np.ndarray:
        """SOC read straight off the OCV table (ignores the load)."""
        return self.ocv_set.at(float(np.mean(temp_c))).inverse(v)

    def ekf(self, t_s, i_meas, v, temp_c, soc0: float, settings: EKFSettings | None = None):
        """EKF with OCV table and ECM parameters looked up from the measured temperature at every sample."""
        return ekf_soc(t_s, i_meas, v, self.ecm, self.ocv_set, soc0=soc0, temp_c=temp_c, capacity_ah=self.capacity_ah, settings=settings)


def load_ml_models(path: Path | str = ML_MODELS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python scripts/run_experiments.py` first")
    return joblib.load(path)["models"]


def estimate_all(df: pd.DataFrame, cell: CellModel, ml_models: dict | None = None, soc0_guess: float | None = None, sensor: SensorErrors | None = None) -> pd.DataFrame:
    """Run every estimator on a 1 Hz log with columns t_s, V, I, T.

    ``soc0_guess`` is what the BMS believes the SOC is at the start (default:
    the OCV lookup of the first sample, i.e. a rested cell).  ``sensor`` adds
    current-sensor errors to the *true* current column before estimation.
    Returns a DataFrame with one column per estimator.
    """
    t, v, i_true, temp = (df[c].to_numpy(float) for c in ("t_s", "V", "I", "T"))
    i_meas = sensor.apply(i_true) if sensor else i_true
    if soc0_guess is None:
        soc0_guess = float(np.clip(cell.ocv_lookup(v[:1], temp[:1])[0], 0, 1))
    out = pd.DataFrame({"t_s": t})
    out["Coulomb counting"] = coulomb_counting(t, i_meas, soc0=soc0_guess, capacity_ah=cell.capacity_ah)
    out["OCV lookup"] = cell.ocv_lookup(v, temp)
    ekf_hat, ekf_sd = cell.ekf(t, i_meas, v, temp, soc0=soc0_guess)
    out["EKF"], out["EKF_std"] = ekf_hat, ekf_sd
    if ml_models:
        feats = make_features(pd.DataFrame({"V": v, "I": i_meas, "T": temp}))
        for name, pipe in ml_models.items():
            out[name] = np.clip(pipe.predict(feats), 0.0, 1.0)
    return out
