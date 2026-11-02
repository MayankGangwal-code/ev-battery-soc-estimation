"""Fit, evaluate and report every SOC estimator.

Steps
-----
1. per temperature: OCV-SOC table and 1-RC ECM (R0, R1, tau fitted by least
   squares on the Mixed1 training cycle).  Four candidate OCV sources are
   compared on the *training* cycles (pulse-test rest voltages, C/20 discharge
   branch, C/20 charge branch, their average) and the best one is used.
2. machine-learning estimators trained on all Mixed (training) cycles
3. every estimator evaluated on the 22 held-out EPA drive cycles (UDDS, HWFET,
   LA92, US06 at six temperatures) under *realistic* conditions:
     - the BMS starts with a wrong initial SOC (10 % too low)
     - the current sensor reads 2 % high with a +50 mA offset
   Coulomb counting is also shown under *ideal* conditions (exact start, perfect sensor).
4. figures, reports/results.md, reports/metrics.json, models/cell_model.json,
   models/ml_models.joblib

Usage:  python scripts/run_experiments.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc import C_REF_AH, TEMP_ORDER, TEMPERATURES, TEST_CYCLES, __version__  # noqa: E402
from soc.coulomb import SensorErrors, coulomb_counting  # noqa: E402
from soc.data import load_capacity, list_cycles, load_cycle, load_hppc, load_index, load_ocv_table  # noqa: E402
from soc.bms import CellModel  # noqa: E402
from soc.ecm import ECMTable, fit_ecm  # noqa: E402
from soc.evaluate import metrics, settling_time_s  # noqa: E402
from soc.features import FEATURE_NAMES, make_features  # noqa: E402
from soc.models import RANDOM_STATE, build_models  # noqa: E402
from soc.ocv import OCVTable  # noqa: E402
from soc.plots import (  # noqa: E402
    plot_cc_drift,
    plot_convergence,
    plot_cycle_overview,
    plot_ocv_curves,
    plot_overall_bars,
    plot_resistance,
    plot_rmse_by_temperature,
    plot_soc_traces,
)

REPORTS, FIG, MODELS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models"

# realistic BMS conditions used for every "realistic" evaluation
SOC0_ERROR = -0.10  # the BMS believes the SOC is 10 % lower than it is
SENSOR = SensorErrors(offset_a=0.05, gain=1.02)  # +50 mA offset, reads 2 % high
SKIP_S = 900  # "after 15 min" metrics ignore the start-up transient
FIT_CYCLE = "Mixed1"
TRACE_CYCLES = [("25degC", "UDDS"), ("n20degC", "US06"), ("0degC", "LA92")]
OCV_SOURCES = ["pulse-test rest voltage", "C/20 discharge", "C/20 charge", "C/20 average"]


def md_table(df: pd.DataFrame, fmt="{:.2f}", index_name="") -> str:
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [("" if (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    t_start = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    index = load_index()
    train_cycles, test_cycles = list_cycles("train"), list_cycles("test")
    print(f"{len(train_cycles)} training cycles, {len(test_cycles)} test cycles")

    # ------------------------------------------------------------------ #
    # 1. OCV tables + ECM parameters per temperature
    # ------------------------------------------------------------------ #
    hppc = {t: load_hppc(t) for t in TEMP_ORDER}
    c20 = {t: load_ocv_table(t) for t in TEMP_ORDER}
    cap = load_capacity()
    c20_temp = {t: float(cap[(cap.temp == t) & (cap.test == "C/20 discharge")]["T_mean"].iloc[0]) for t in TEMP_ORDER}

    def ocv_source(temp: str, kind: str) -> OCVTable:
        hp = hppc[temp].dropna(subset=["ocv_rest"])
        tab = c20[temp]
        return {
            "pulse-test rest voltage": OCVTable(hp["soc"].to_numpy(), hp["ocv_rest"].to_numpy()),
            "C/20 discharge": OCVTable(tab["soc"].to_numpy(), tab["ocv_discharge"].to_numpy()),
            "C/20 charge": OCVTable(tab["soc"].to_numpy(), tab["ocv_charge"].to_numpy()),
            "C/20 average": OCVTable(tab["soc"].to_numpy(), tab["ocv"].to_numpy()),
        }[kind]

    def build_cell(kind: str):
        """Cell model for one OCV source: table + ECM refitted on Mixed1 at each temperature."""
        tabs, params, temps, info = {}, [], [], {}
        for temp in TEMP_ORDER:
            o = ocv_source(temp, kind)
            tr = load_cycle(temp, FIT_CYCLE)
            p, rms = fit_ecm(tr["t_s"], tr["I"], tr["V"], tr["SOC"], o, r0_guess=float(hppc[temp]["R0_ohm"].median()))
            fit_t = float(tr["T"].mean())
            key_t = c20_temp[temp] if kind.startswith("C/20") else float(hppc[temp]["T"].mean())
            tabs[key_t], info[temp] = o, {**p.as_dict(), "fit_rms_mV": 1000 * rms, "fit_cycle": FIT_CYCLE, "fit_temp_C": fit_t, "ocv_table_temp_C": key_t}
            params.append(p)
            temps.append(fit_t)
        return CellModel(tabs, ECMTable(temps, params), C_REF_AH, meta={"ocv_source": kind, "ecm_fit_cycle": FIT_CYCLE, "ecm_per_temperature": info}), info

    print("\n1a. choosing the OCV table on the training cycles (EKF, realistic conditions)")
    select_cycles = [(t, c) for t, c in train_cycles if c != FIT_CYCLE]  # Mixed1 is used for the ECM fit
    ocv_rows = []
    for kind in OCV_SOURCES:
        cell_k, _ = build_cell(kind)
        for temp, cyc in select_cycles:
            d = load_cycle(temp, cyc)
            soc = d["SOC"].to_numpy()
            s_hat, _ = cell_k.ekf(d["t_s"].to_numpy(), SENSOR.apply(d["I"].to_numpy()), d["V"].to_numpy(), d["T"].to_numpy(), soc0=soc[0] + SOC0_ERROR)
            ocv_rows.append({"ocv_source": kind, "temp_degC": TEMPERATURES[temp], "cycle": cyc, "rmse": metrics(soc, s_hat)["rmse"],
                             "bias_after_15min": float(np.mean(100 * (s_hat - soc)[SKIP_S:]))})
        print(f"   {kind:24s} mean RMSE {np.mean([r['rmse'] for r in ocv_rows if r['ocv_source'] == kind]):.2f} %")
    ocv_cmp = pd.DataFrame(ocv_rows)
    ocv_overall = ocv_cmp.groupby("ocv_source")["rmse"].mean().loc[OCV_SOURCES]
    ocv_by_temp = ocv_cmp.pivot_table(index="ocv_source", columns="temp_degC", values="rmse").loc[OCV_SOURCES]
    ocv_bias = ocv_cmp.pivot_table(index="ocv_source", columns="temp_degC", values="bias_after_15min").loc[OCV_SOURCES]
    OCV_CHOICE = str(ocv_overall.idxmin())
    print(f"   -> using the {OCV_CHOICE} table")

    print(f"\n1b. equivalent-circuit model per temperature (OCV: {OCV_CHOICE}, fit on {FIT_CYCLE})")
    cell, ecm = build_cell(OCV_CHOICE)
    for temp in TEMP_ORDER:
        e = ecm[temp]
        print(f"   {temp:8s} R0 = {1000 * e['R0']:5.1f} mΩ  R1 = {1000 * e['R1']:5.1f} mΩ  tau = {e['tau']:4.0f} s  (fit RMS {e['fit_rms_mV']:.0f} mV at {e['fit_temp_C']:.1f} °C, pulse-test R0 {1000 * hppc[temp]['R0_ohm'].median():.1f} mΩ)")
    cell.to_json(MODELS / "cell_model.json")

    # effective resistance versus current on the coldest training cycle: shows
    # why a linear model struggles at -20 degC
    d_cold = load_cycle("n20degC", FIT_CYCLE)
    o_cold = cell.ocv_set.at(float(d_cold["T"].mean()))
    i_cold, v_cold = d_cold["I"].to_numpy(), d_cold["V"].to_numpy()
    overpotential = o_cold(d_cold["SOC"].to_numpy()) - v_cold
    low = (i_cold <= -0.5) & (i_cold >= -1.5)
    high = (i_cold <= -5.5) & (i_cold >= -6.5)
    r_eff_low = 1000 * float(np.median(overpotential[low] / np.abs(i_cold[low])))
    r_eff_high = 1000 * float(np.median(overpotential[high] / np.abs(i_cold[high])))
    print(f"   -20 °C effective resistance (OCV - V)/|I|: {r_eff_low:.0f} mΩ at ~1 A, {r_eff_high:.0f} mΩ at ~6 A")

    # ------------------------------------------------------------------ #
    # 2. machine-learning estimators
    # ------------------------------------------------------------------ #
    print("\n2. training the ML estimators on the Mixed cycles")
    X_parts, y_parts = [], []
    for temp, cyc in train_cycles:
        d = load_cycle(temp, cyc)
        X_parts.append(make_features(d))
        y_parts.append(d["SOC"].to_numpy())
    X_tr, y_tr = pd.concat(X_parts, ignore_index=True), np.concatenate(y_parts)
    print(f"   {len(X_tr):,} training samples x {len(FEATURE_NAMES)} features")
    ml = build_models()
    ml_fit_s = {}
    for name, pipe in ml.items():
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        ml_fit_s[name] = time.time() - t0
        print(f"   {name:18s} fitted in {ml_fit_s[name]:.0f} s")
    joblib.dump({"models": ml, "feature_names": FEATURE_NAMES, "package_version": __version__, "n_train": int(len(X_tr))}, MODELS / "ml_models.joblib")

    # ------------------------------------------------------------------ #
    # 3. evaluation on the held-out cycles
    # ------------------------------------------------------------------ #
    print("\n3. evaluating on the held-out drive cycles")
    rows, traces = [], {}
    for temp, cyc in test_cycles:
        d = load_cycle(temp, cyc)
        t, i_true, v, soc = d["t_s"].to_numpy(), d["I"].to_numpy(), d["V"].to_numpy(), d["SOC"].to_numpy()
        i_meas = SENSOR.apply(i_true)
        soc0_wrong = soc[0] + SOC0_ERROR
        temp_c = d["T"].to_numpy()
        est = {
            "Coulomb counting (ideal)": coulomb_counting(t, i_true, soc0=soc[0]),
            "Coulomb counting (realistic)": coulomb_counting(t, i_true, soc0=soc0_wrong, errors=SENSOR),
            "OCV lookup": cell.ocv_lookup(v, temp_c),
        }
        ekf_hat, ekf_sd = cell.ekf(t, i_meas, v, temp_c, soc0=soc0_wrong)
        est["EKF"] = ekf_hat
        feats = make_features(pd.DataFrame({"V": v, "I": i_meas, "T": d["T"].to_numpy()}))
        for name, pipe in ml.items():
            est[name] = np.clip(pipe.predict(feats), 0.0, 1.0)
        for name, s in est.items():
            m = metrics(soc, s)
            m15 = metrics(soc, s, skip_s=SKIP_S)
            rows.append({"temp": temp, "temp_degC": TEMPERATURES[temp], "cycle": cyc, "hours": round(t[-1] / 3600, 2), "method": name,
                         **m, "rmse_after_15min": m15["rmse"], "max_abs_after_15min": m15["max_abs"],
                         "settle_s": settling_time_s(soc, s) if name == "EKF" else np.nan})
        if (temp, cyc) in TRACE_CYCLES:
            traces[(temp, cyc)] = (d, est, ekf_sd)
        print(f"   {temp:8s} {cyc:6s} ({t[-1] / 3600:.1f} h)  " + "  ".join(f"{n.split(' (')[0][:12]}={metrics(soc, s)['rmse']:.2f}" for n, s in est.items()))
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "per_cycle_metrics.csv", index=False)

    methods = list(est.keys())
    by_temp = res.pivot_table(index="method", columns="temp", values="rmse", aggfunc="mean").loc[methods, TEMP_ORDER]
    overall = res.groupby("method")[["rmse", "mae", "max_abs", "rmse_after_15min", "max_abs_after_15min"]].mean().loc[methods]
    overall["rmse_median"] = res.groupby("method")["rmse"].median().loc[methods]
    ekf_rows = res[res.method == "EKF"]
    warm, cold = ekf_rows[ekf_rows.temp_degC >= 10], ekf_rows[ekf_rows.temp_degC < 0]
    ekf_all = {"settle_median_s": float(ekf_rows["settle_s"].median()), "settle_max_s": float(ekf_rows["settle_s"].max()),
               "max_abs_after_15min": float(ekf_rows["max_abs_after_15min"].max())}
    ekf_warm = {"settle_median_s": float(warm["settle_s"].median()), "settle_max_s": float(warm["settle_s"].max()),
                "rmse": float(warm["rmse"].mean()), "max_abs_after_15min": float(warm["max_abs_after_15min"].max())}
    ekf_cold = {"rmse": float(cold["rmse"].mean()), "rmse_min": float(cold["rmse"].min()), "rmse_max": float(cold["rmse"].max()),
                "max_abs_after_15min": float(cold["max_abs_after_15min"].max())}

    # ------------------------------------------------------------------ #
    # 4. figures
    # ------------------------------------------------------------------ #
    print("\n4. figures")
    for (temp, cyc), (d, est, sd) in traces.items():
        plot_soc_traces(d["t_s"], d["SOC"], est, f"{cyc} drive cycle at {TEMPERATURES[temp]} °C — all estimators (realistic sensor, wrong initial SOC)",
                        FIG / f"traces_{cyc}_{temp}.png")
    plot_rmse_by_temperature(by_temp, FIG / "rmse_by_temperature.png", ymax=float(min(30, by_temp.max().max() * 1.15)))
    plot_overall_bars(overall["rmse"], FIG / "rmse_overall.png", title=f"Overall RMSE on the {len(test_cycles)} test cycles (% SOC)")
    plot_ocv_curves(c20, hppc, FIG / "ocv_curves.png", used=OCV_CHOICE)
    plot_resistance(hppc, ecm, FIG / "internal_resistance.png")

    # EKF convergence from several wrong starts (UDDS at 25 °C)
    d = load_cycle("25degC", "UDDS")
    t, v, soc = d["t_s"].to_numpy(), d["V"].to_numpy(), d["SOC"].to_numpy()
    i_meas = SENSOR.apply(d["I"].to_numpy())
    starts = {f"EKF, started at {100 * (soc[0] + e):.0f} %": soc[0] + e for e in (-0.50, -0.30, -0.10)}
    runs = {k: cell.ekf(t, i_meas, v, d["T"].to_numpy(), soc0=s0) for k, s0 in starts.items()}
    plot_convergence(t, soc, runs, starts, FIG / "ekf_convergence.png")

    # Coulomb-counting drift for several sensor offsets (UDDS at 25 °C, 4.4 h)
    drift = {f"{int(1000 * o)} mA": coulomb_counting(t, d["I"].to_numpy(), soc0=soc[0], errors=SensorErrors(offset_a=o)) for o in (0.0, 0.02, 0.05, 0.10)}
    plot_cc_drift(t, soc, drift, FIG / "coulomb_counting_drift.png")
    plot_cycle_overview(load_cycle("25degC", "US06"), "US06 drive cycle at 25 °C — what the BMS measures", FIG / "cycle_overview_US06_25degC.png")

    # ------------------------------------------------------------------ #
    # 5. metrics.json + results.md
    # ------------------------------------------------------------------ #
    summary = {
        "package_version": __version__, "C_REF_AH": C_REF_AH, "n_train_cycles": len(train_cycles), "n_test_cycles": len(test_cycles),
        "n_train_samples": int(len(X_tr)), "features": FEATURE_NAMES, "realistic_conditions": {"soc0_error": SOC0_ERROR, "current_offset_A": SENSOR.offset_a, "current_gain": SENSOR.gain},
        "ecm": ecm, "overall": overall.round(3).to_dict(orient="index"), "rmse_by_temperature": by_temp.round(3).to_dict(orient="index"),
        "ekf_all": ekf_all, "ekf_warm_10_to_40C": ekf_warm, "ekf_cold_below_0C": ekf_cold, "ml_fit_seconds": {k: round(v, 1) for k, v in ml_fit_s.items()},
        "ocv_source_used": OCV_CHOICE, "ocv_source_comparison_train_rmse": ocv_overall.round(3).to_dict(),
        "ocv_source_comparison_train_rmse_by_temp": ocv_by_temp.round(3).to_dict(orient="index"),
        "ocv_source_comparison_train_bias_by_temp": ocv_bias.round(3).to_dict(orient="index"),
        "n_ocv_selection_cycles": len(select_cycles),
        "effective_resistance_n20degC_mOhm": {"at_1A": round(r_eff_low, 1), "at_6A": round(r_eff_high, 1)},
        "worst_single_error_after_15min": res.groupby("method")["max_abs_after_15min"].max().round(3).to_dict(),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    ov = overall[["rmse", "mae", "rmse_after_15min", "max_abs_after_15min"]].copy()
    ov["worst"] = res.groupby("method")["max_abs_after_15min"].max().loc[methods]
    ov.columns = ["RMSE (%)", "MAE (%)", "RMSE after 15 min (%)", "mean per-cycle max |error| after 15 min (%)", "worst error after 15 min (%)"]
    ocv_show = pd.DataFrame({"mean RMSE (%)": ocv_overall.round(2)})
    ocv_bt = ocv_by_temp.copy(); ocv_bt.columns = [f"{int(c)} °C" for c in ocv_bt.columns]
    ocv_bs = ocv_bias.copy(); ocv_bs.columns = [f"{int(c)} °C" for c in ocv_bs.columns]
    ocv_show = pd.concat([ocv_show, ocv_bt.round(2)], axis=1)
    bt = by_temp.copy()
    bt.columns = [f"{TEMPERATURES[t]} °C" for t in bt.columns]
    ecm_tab = pd.DataFrame({f"{TEMPERATURES[t]} °C": {"R0 (mΩ)": 1000 * ecm[t]["R0"], "R1 (mΩ)": 1000 * ecm[t]["R1"], "τ = R1C1 (s)": ecm[t]["tau"], "fit RMS (mV)": ecm[t]["fit_rms_mV"],
                                                     "pulse-test R0 (mΩ)": 1000 * hppc[t]["R0_ohm"].median()} for t in TEMP_ORDER})
    per_cycle = res[res.method.isin(["Coulomb counting (realistic)", "EKF", "Neural Network", "Gradient Boosting"])].pivot_table(index=["temp_degC", "cycle"], columns="method", values="rmse")[["Coulomb counting (realistic)", "EKF", "Gradient Boosting", "Neural Network"]]
    per_cycle.index = [f"{int(a)} °C, {b}" for a, b in per_cycle.index]

    report = f"""# Results

Generated by `scripts/run_experiments.py` (random_state = {RANDOM_STATE}, runtime {summary['runtime_s']} s).

**Data.** LG 18650HG2 cell, {len(train_cycles)} training drive cycles (random mixes) and **{len(test_cycles)} held-out
EPA drive cycles** (UDDS, HWFET, LA92, US06) at −20, −10, 0, 10, 25 and 40 °C, resampled to 1 Hz.
SOC ground truth = Ah counted by the laboratory cycler from a full charge, relative to the measured
C/20 capacity of {C_REF_AH} Ah.

**Realistic BMS conditions** (applied to every estimator except the "ideal" Coulomb counter): the BMS
starts believing the SOC is **10 % lower** than it is, and the current sensor reads **2 % high with a
+50 mA offset**.

## 1. Overall accuracy on the {len(test_cycles)} test cycles

"After 15 min" columns ignore the first 15 minutes of each cycle, i.e. the start-up transient of the
estimators that begin with the wrong initial SOC.

{md_table(ov, index_name="Estimator")}

![overall](figures/rmse_overall.png)

* Ideal Coulomb counting is essentially exact — the problem is never the method, it is the sensor and
  the unknown starting point: with realistic errors the same integral drifts to a mean RMSE of
  {overall.loc['Coulomb counting (realistic)', 'rmse']:.1f} %.
* Reading SOC off the OCV curve from the terminal voltage alone fails under load (voltage sag), with
  the error exploding at low temperature where the internal resistance is largest.
* The **EKF** (equivalent-circuit model + Kalman filter) uses the voltage to *correct* Coulomb counting.
  From the 10 % wrong start the estimate is back inside ±2 % of the truth (and stays there for at least a
  minute) within {ekf_all['settle_max_s'] + 1:.0f} s on every test cycle: each cycle starts from a fully charged cell, where the OCV
  curve is steep and a single voltage reading is very informative. At 10–40 °C the filter then holds an
  RMSE of {ekf_warm['rmse']:.2f} % despite the biased current sensor, with a worst error of {ekf_warm['max_abs_after_15min']:.1f} % after the first
  15 min (the slow drift on the long, low-current UDDS cycles). Below 0 °C the linear 1-RC model describes the cell less well — on the −20 °C
  training cycle the effective resistance (OCV − V)/|I| is ≈{r_eff_low:.0f} mΩ at 1 A but only ≈{r_eff_high:.0f} mΩ at 6 A, a
  non-linearity a fixed R₀ cannot capture — and the per-cycle RMSE is {ekf_cold['rmse_min']:.1f}–{ekf_cold['rmse_max']:.1f} %
  (worst error {ekf_cold['max_abs_after_15min']:.1f} % after the first 15 min).
* The **ML estimators** need neither an initial SOC nor a current integral; the neural network reaches
  {overall.loc['Neural Network', 'rmse']:.2f} % RMSE and gradient boosting {overall.loc['Gradient Boosting', 'rmse']:.2f} %.

## 2. RMSE by temperature (mean over the test cycles at each temperature, % SOC)

{md_table(bt, index_name="Estimator")}

![bytemp](figures/rmse_by_temperature.png)

## 3. What the estimators do over a drive cycle

![t1](figures/traces_UDDS_25degC.png)

![t2](figures/traces_US06_n20degC.png)

![t3](figures/traces_LA92_0degC.png)

### Coulomb counting drifts with any current offset

![drift](figures/coulomb_counting_drift.png)

### The EKF recovers from a wrong initial SOC (UDDS at 25 °C, shaded = ±2σ of the filter's own uncertainty)

![conv](figures/ekf_convergence.png)

## 4. Cell characterisation used by the model-based estimators

### Equivalent-circuit parameters (1-RC Thevenin model, fitted on the Mixed1 cycle at each temperature)

{md_table(ecm_tab, fmt="{:.1f}", index_name="")}

![res](figures/internal_resistance.png)

### OCV–SOC tables — which curve should the BMS store?

A Li-ion cell shows hysteresis: at the same SOC the OCV measured while charging is higher than while
discharging (right panel of the figure). Four candidate tables were compared on the {len(select_cycles)} training
cycles (EKF, same realistic conditions, ECM refitted for each table); the best one — **{OCV_CHOICE}** — is
the table used everywhere else in this report.

{md_table(ocv_show, index_name="OCV table (RMSE % on training cycles)")}

Mean signed EKF error after 15 min (% SOC), training cycles — a wrong curve shows up as a bias:

{md_table(ocv_bs, index_name="OCV table")}

The charge curve and the charge/discharge average both bias the filter (the cell is almost always
discharging while driving). The pulse-test rest voltages and the C/20 discharge curve agree within
≈10–20 mV above 35 % SOC and give similar results at 25–40 °C; the C/20 curve wins clearly at low
temperature, most likely because the cold pulse test has few SOC steps (9 at −20 °C versus 180 points
on the C/20 curve) and a 1 h rest is not long enough for a cold cell to reach equilibrium.

![ocv](figures/ocv_curves.png)

## 5. Per-cycle RMSE (% SOC)

{md_table(per_cycle, index_name="Test cycle")}

## 6. What the BMS sees

![ov](figures/cycle_overview_US06_25degC.png)
"""
    (REPORTS / "results.md").write_text(report)
    print(f"\nwrote {REPORTS / 'results.md'}, {REPORTS / 'metrics.json'}, {MODELS / 'cell_model.json'}, {MODELS / 'ml_models.joblib'} and {len(list(FIG.glob('*.png')))} figures")
    print("\noverall RMSE (%):")
    print(overall["rmse"].round(2).to_string())
    print(f"\ndone in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
