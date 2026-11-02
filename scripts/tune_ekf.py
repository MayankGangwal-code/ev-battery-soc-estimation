"""Grid search of the EKF noise settings on the TRAINING cycles only.

The EKF has three tuning constants (``soc.ekf.EKFSettings``): the process-noise
variances q_soc and q_v1 and the voltage measurement-noise variance r_v.  This
script runs the filter with every combination on the Mixed training cycles
(Mixed1 excluded because it was used to fit the circuit model), under the same
realistic conditions as the evaluation, and prints the mean RMSE of each
combination.  The defaults in ``EKFSettings`` are the best combination found.

Needs models/cell_model.json (run scripts/run_experiments.py first).

Usage:  python scripts/tune_ekf.py            (~2 min)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc import TEMPERATURES  # noqa: E402
from soc.bms import CellModel  # noqa: E402
from soc.coulomb import SensorErrors  # noqa: E402
from soc.data import list_cycles, load_cycle  # noqa: E402
from soc.ekf import EKFSettings  # noqa: E402
from soc.evaluate import metrics  # noqa: E402

SOC0_ERROR, SENSOR, FIT_CYCLE = -0.10, SensorErrors(offset_a=0.05, gain=1.02), "Mixed1"
GRID = {"q_soc": [1e-11, 1e-10, 1e-9], "q_v1": [1e-7, 1e-6, 1e-5], "r_v": [1e-5, 1e-4, 1e-3]}


def main() -> None:
    cell = CellModel.from_json()
    cycles = [(t, c) for t, c in list_cycles("train") if c != FIT_CYCLE]
    data = [(TEMPERATURES[t], load_cycle(t, c)) for t, c in cycles]
    print(f"{len(data)} training cycles, {int(np.prod([len(v) for v in GRID.values()]))} settings")
    rows = []
    for q_soc, q_v1, r_v in itertools.product(*GRID.values()):
        st = EKFSettings(q_soc=q_soc, q_v1=q_v1, r_v=r_v)
        errs = []
        for temp_c, d in data:
            soc = d["SOC"].to_numpy()
            s_hat, _ = cell.ekf(d["t_s"].to_numpy(), SENSOR.apply(d["I"].to_numpy()), d["V"].to_numpy(), d["T"].to_numpy(), soc0=soc[0] + SOC0_ERROR, settings=st)
            errs.append({"temp": temp_c, "rmse": metrics(soc, s_hat)["rmse"], "rmse15": metrics(soc, s_hat, skip_s=900)["rmse"]})
        e = pd.DataFrame(errs)
        rows.append({"q_soc": q_soc, "q_v1": q_v1, "r_v": r_v, "rmse": e["rmse"].mean(), "rmse_after_15min": e["rmse15"].mean(),
                     "rmse_warm": e[e.temp >= 10]["rmse"].mean(), "rmse_cold": e[e.temp < 0]["rmse"].mean()})
        print(f"  q_soc={q_soc:.0e} q_v1={q_v1:.0e} r_v={r_v:.0e}: RMSE {rows[-1]['rmse']:.3f} %  (warm {rows[-1]['rmse_warm']:.2f}, cold {rows[-1]['rmse_cold']:.2f})")
    res = pd.DataFrame(rows).sort_values("rmse")
    out = ROOT / "reports" / "ekf_tuning.csv"
    res.round({"rmse": 4, "rmse_after_15min": 4, "rmse_warm": 4, "rmse_cold": 4}).to_csv(out, index=False, float_format="%.6g")
    best = res.iloc[0]
    print(f"\nbest: q_soc={best.q_soc:.0e} q_v1={best.q_v1:.0e} r_v={best.r_v:.0e} -> RMSE {best.rmse:.3f} %")
    d = EKFSettings()
    print(f"EKFSettings defaults: q_soc={d.q_soc:.0e} q_v1={d.q_v1:.0e} r_v={d.r_v:.0e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
