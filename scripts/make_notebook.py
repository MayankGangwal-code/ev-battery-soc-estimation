"""Generate notebooks/soc_walkthrough.ipynb (plain JSON, no jupyter needed to build it).

Usage:  python scripts/make_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "soc_walkthrough.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# EV battery SOC estimation — walkthrough

This notebook reproduces the project step by step on one drive cycle: look at the data, build the
estimators one by one (Coulomb counting → OCV lookup → equivalent-circuit model → EKF → machine learning)
and compare them. Run it from the repository root after `python scripts/run_experiments.py`.
""")

code("""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = Path.cwd() if (Path.cwd() / "src" / "soc").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from soc import C_REF_AH, TEMP_ORDER, TEMPERATURES
from soc.data import load_cycle, load_hppc, load_ocv_table, load_index
from soc.coulomb import coulomb_counting, SensorErrors
from soc.ocv import OCVTable
from soc.ecm import fit_ecm, simulate_terminal_voltage
from soc.ekf import ekf_soc
from soc.evaluate import metrics
from soc.plots import plot_soc_traces, plot_cycle_overview

idx = load_index()
idx[idx.valid].groupby(["temp_degC", "split"]).size().unstack()
""")

md("## 1. What the BMS measures\n\nA US06 cycle at 25 °C: voltage, current (charging positive — the spikes above zero are regenerative braking), cell temperature and the ground-truth SOC from the cycler's Ah counter.")

code("""
d = load_cycle("25degC", "US06")
plot_cycle_overview(d, "US06 at 25 °C")
""")

md("## 2. Coulomb counting\n\nExact with a perfect sensor and a known start; drifts with a sensor offset.")

code("""
d = load_cycle("25degC", "UDDS")
t, i, v, soc = d.t_s.to_numpy(), d.I.to_numpy(), d.V.to_numpy(), d.SOC.to_numpy()
ideal = coulomb_counting(t, i, soc0=soc[0])
drift = coulomb_counting(t, i, soc0=soc[0] - 0.10, errors=SensorErrors(offset_a=0.05, gain=1.02))
print("ideal   RMSE %.2f %%" % metrics(soc, ideal)["rmse"])
print("realistic (−10 %% start, +50 mA, +2 %%) RMSE %.2f %%, final error %+.1f %%" % (metrics(soc, drift)["rmse"], metrics(soc, drift)["final_err"]))
""")

md("## 3. The OCV–SOC table and the 'voltage gauge'\n\nThe estimators use the OCV curve from the slow C/20 discharge (the choice between the C/20 curves and the pulse-test rest voltages was made on the training cycles — see `reports/results.md`). Reading SOC off the curve while driving fails because of the I·R drop.")

code("""
hp = load_hppc("25degC").dropna(subset=["ocv_rest"])
tab = load_ocv_table("25degC")
ocv = OCVTable(tab.soc, tab.ocv_discharge)
lookup = ocv.inverse(v)
print("OCV lookup RMSE %.1f %%" % metrics(soc, lookup)["rmse"])
fig, ax = plt.subplots(figsize=(6, 3.5))
for temp in TEMP_ORDER:
    t_ = load_ocv_table(temp)
    ax.plot(100 * t_.soc, t_.ocv_discharge, label=f"{TEMPERATURES[temp]} °C")
ax.scatter(100 * hp.soc, hp.ocv_rest, color="k", s=12, zorder=5, label="pulse-test rests, 25 °C")
ax.set_xlabel("SOC (%)"); ax.set_ylabel("OCV (V)"); ax.legend(fontsize=8); fig
""")

md("## 4. Equivalent-circuit model\n\nFit R₀, R₁, τ of a 1-RC Thevenin model on a *training* cycle (Mixed1) by least squares, then check how well it reproduces the voltage of the test cycle.")

code("""
tr = load_cycle("25degC", "Mixed1")
params, rms = fit_ecm(tr.t_s, tr.I, tr.V, tr.SOC, ocv, r0_guess=float(hp.R0_ohm.median()))
print(f"R0 = {1000*params.R0:.1f} mΩ, R1 = {1000*params.R1:.1f} mΩ, tau = {params.tau:.0f} s   (fit RMS {1000*rms:.0f} mV)")
v_model = simulate_terminal_voltage(t, i, soc, params, ocv)
fig, ax = plt.subplots(figsize=(9, 3.2))
sl = slice(3000, 3600)
ax.plot(t[sl], v[sl], label="measured", lw=1); ax.plot(t[sl], v_model[sl], label="model", lw=1)
ax.set_xlabel("time (s)"); ax.set_ylabel("V"); ax.legend(); ax.set_title("UDDS at 25 °C, 10 minutes"); fig
""")

md("## 5. Extended Kalman Filter\n\nPredict with the model, correct with the measured voltage. Started 10 % too low, with the same biased current sensor as the drifting Coulomb counter.")

code("""
i_meas = SensorErrors(offset_a=0.05, gain=1.02).apply(i)
ekf, ekf_sd = ekf_soc(t, i_meas, v, params, ocv, soc0=soc[0] - 0.10)
print("EKF RMSE %.2f %% (max error %.1f %%)" % (metrics(soc, ekf)["rmse"], metrics(soc, ekf)["max_abs"]))
plot_soc_traces(t, soc, {"Coulomb counting (realistic)": drift, "OCV lookup": lookup, "EKF": ekf}, "UDDS at 25 °C")
""")

md("## 6. Machine-learning estimators\n\nThe trained models (gradient boosting and a neural network on V, I, T and their moving averages) are loaded from `models/`.")

code("""
from soc.bms import CellModel, load_ml_models, estimate_all
cell = CellModel.from_json(ROOT / "models" / "cell_model.json")
ml = load_ml_models(ROOT / "models" / "ml_models.joblib")
est = estimate_all(d, cell, ml, soc0_guess=soc[0] - 0.10, sensor=SensorErrors(offset_a=0.05, gain=1.02))
pd.DataFrame({n: metrics(soc, est[n]) for n in ["Coulomb counting", "OCV lookup", "EKF", "Gradient Boosting", "Neural Network"]}).T.round(2)
""")

md("## 7. The hard case: US06 at −20 °C\n\nHigh power and a cold cell: the internal resistance is 7× that at 40 °C and the OCV lookup is hopeless. Compare how the model-based and the learned estimators cope.")

code("""
d2 = load_cycle("n20degC", "US06")
est2 = estimate_all(d2, cell, ml, soc0_guess=d2.SOC.iloc[0] - 0.10, sensor=SensorErrors(offset_a=0.05, gain=1.02))
display(pd.DataFrame({n: metrics(d2.SOC, est2[n]) for n in ["Coulomb counting", "EKF", "Gradient Boosting", "Neural Network"]}).T.round(2))
plot_soc_traces(d2.t_s, d2.SOC, {n: est2[n].to_numpy() for n in ["Coulomb counting", "EKF", "Neural Network"]}, "US06 at −20 °C")
""")

md("""
## 8. Take-aways

* Coulomb counting is exact in the lab and drifts in the car: sensor offset and an unknown start are the problem, not the integral.
* The OCV lookup only works on a rested cell.
* The EKF uses the voltage to correct Coulomb counting: the first voltage sample removes the wrong start, and the error stays around 1 % with a worst case of a few percent; the linear 1-RC model is the limit at low temperature and low current.
* The neural network is almost as accurate on average and needs no initial SOC, at the price of a larger worst-case error and of only knowing the conditions it was trained on.
""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT, f"({len(cells)} cells)")
