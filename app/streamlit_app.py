"""Interactive demo: watch every SOC estimator work through a real EV drive cycle.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc import C_REF_AH, CYCLE_DESCRIPTIONS, TEMP_ORDER, TEMPERATURES  # noqa: E402
from soc.bms import CellModel, estimate_all, load_ml_models  # noqa: E402
from soc.coulomb import SensorErrors  # noqa: E402
from soc.data import list_cycles, load_cycle, load_ocv_table  # noqa: E402
from soc.evaluate import metrics, settling_time_s  # noqa: E402
from soc.plots import plot_soc_traces  # noqa: E402

st.set_page_config(page_title="EV battery SOC estimation", page_icon="🔋", layout="wide")


@st.cache_resource
def get_models():
    return CellModel.from_json(), load_ml_models()


@st.cache_data
def get_cycle(temp: str, cycle: str) -> pd.DataFrame:
    return load_cycle(temp, cycle)


try:
    cell, ml_models = get_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.title("🔋 EV Battery State-of-Charge Estimation")
st.caption(
    "LG 18650HG2 cell on real EPA drive cycles at −20 … 40 °C. Compare the textbook methods "
    "(Coulomb counting, OCV lookup, Extended Kalman Filter on a Thevenin model) with machine learning, "
    "under realistic sensor errors and a wrong starting point."
)

# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Drive cycle")
    source = st.radio("Data", ["Dataset test cycle", "Upload my own log"])
    if source == "Dataset test cycle":
        temp = st.selectbox("Cell temperature", TEMP_ORDER, index=4, format_func=lambda t: f"{TEMPERATURES[t]} °C")
        cycles = [c for t, c in list_cycles("test", [temp])]
        cycle = st.selectbox("Cycle", cycles, format_func=lambda c: f"{c} — {CYCLE_DESCRIPTIONS.get(c, '')}")
        df = get_cycle(temp, cycle)
        truth = df["SOC"].to_numpy()
    else:
        up = st.file_uploader("CSV sampled at 1 Hz with columns t_s, V, I, T", type="csv")
        if up is None:
            st.info("Upload a log, or switch to a dataset cycle.")
            st.stop()
        df = pd.read_csv(up)
        missing = [c for c in ("t_s", "V", "I", "T") if c not in df.columns]
        if missing:
            st.error(f"missing columns: {missing}")
            st.stop()
        truth = df["SOC"].to_numpy() if "SOC" in df.columns else None
        temp, cycle = None, "your log"

    st.header("BMS imperfections")
    soc0_err = st.slider("Initial SOC error assumed by the BMS (%)", -50, 10, -10, 5)
    offset_ma = st.slider("Current-sensor offset (mA)", 0, 200, 50, 10)
    gain_pct = st.slider("Current-sensor gain error (%)", -5.0, 5.0, 2.0, 0.5)
    use_ml = st.checkbox("Include machine-learning estimators", value=True)

sensor = SensorErrors(offset_a=offset_ma / 1000.0, gain=1 + gain_pct / 100.0)
start_true = float(truth[0]) if truth is not None else float(np.clip(cell.ocv_lookup(df["V"].to_numpy()[:1], df["T"].to_numpy()[:1])[0], 0, 1))
soc0 = float(np.clip(start_true + soc0_err / 100.0, -0.05, 1.05))

est = estimate_all(df, cell, ml_models if use_ml else None, soc0_guess=soc0, sensor=sensor)
names = [c for c in est.columns if c not in ("t_s", "EKF_std")]

# ----------------------------------------------------------------------------
# headline
# ----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration", f"{df['t_s'].iloc[-1] / 3600:.2f} h")
c2.metric("Mean cell temperature", f"{df['T'].mean():.1f} °C")
c3.metric("Charge removed", f"{-(df['I'].to_numpy()[1:] * np.diff(df['t_s'].to_numpy())).sum() / 3600:.2f} Ah of {C_REF_AH} Ah")
c4.metric("BMS start assumption", f"{100 * soc0:.0f} % SOC", delta=f"{soc0_err:+d} % vs truth" if truth is not None else None, delta_color="off")

if truth is not None:
    rows = []
    for n in names:
        m = metrics(truth, est[n])
        rows.append({"estimator": n, "RMSE (%)": m["rmse"], "max |error| (%)": m["max_abs"], "final error (%)": m["final_err"],
                     "settles within ±2 % after (s)": settling_time_s(truth, est[n]) if n == "EKF" else None})
    tab = pd.DataFrame(rows).set_index("estimator").round(2)
    st.subheader("Accuracy on this cycle")
    st.dataframe(tab)
else:
    st.subheader("Final estimates")
    st.dataframe(pd.DataFrame({n: [f"{100 * est[n].iloc[-1]:.1f} %"] for n in names}, index=["SOC at end of log"]))

# ----------------------------------------------------------------------------
# traces
# ----------------------------------------------------------------------------
st.subheader("SOC over the cycle")
plot_est = {n: est[n].to_numpy() for n in names}
if truth is None:
    truth_plot = est["EKF"].to_numpy()
    st.caption("No ground-truth SOC in the uploaded log: the EKF estimate is drawn as the reference line.")
else:
    truth_plot = truth
title = f"{cycle} at {TEMPERATURES[temp]} °C" if temp else "your log"
fig = plot_soc_traces(df["t_s"].to_numpy(), truth_plot, plot_est, title, show_error=truth is not None)
st.pyplot(fig)

# ----------------------------------------------------------------------------
# cell model
# ----------------------------------------------------------------------------
with st.expander("The cell model behind the EKF"):
    cols = st.columns(2)
    ecm_tab = pd.DataFrame(
        {f"{t:.0f} °C": {"R0 (mΩ)": 1000 * p.R0, "R1 (mΩ)": 1000 * p.R1, "τ = R1·C1 (s)": p.tau} for t, p in zip(cell.ecm.temps, cell.ecm.params)}
    ).round(1)
    cols[0].markdown("**1-RC Thevenin parameters** (fitted on the Mixed1 cycle at each temperature, interpolated by the measured temperature)")
    cols[0].dataframe(ecm_tab)
    ocv_tab = pd.concat({f"{TEMPERATURES[t]} °C": load_ocv_table(t).set_index("soc")["ocv_discharge"] for t in TEMP_ORDER}, axis=1)
    cols[1].markdown(f"**OCV–SOC table** ({cell.meta.get('ocv_source', 'C/20 discharge')} curve at each temperature — chosen on the training cycles)")
    cols[1].line_chart(ocv_tab, height=260)

with st.expander("How the estimators work"):
    st.markdown(
        """
| Estimator | Idea | Needs | Weakness |
|---|---|---|---|
| **Coulomb counting** | integrate the current: SOC = SOC₀ + ∫I dt / (3600·C) | initial SOC, accurate current sensor | every sensor error is integrated → drifts |
| **OCV lookup** | read SOC off the open-circuit-voltage curve | a rested cell | under load the terminal voltage ≠ OCV (I·R drop), useless while driving |
| **EKF** | predict with Coulomb counting + Thevenin model, correct with the measured voltage | OCV table, R₀/R₁/C₁ per temperature | a single R₀ cannot follow the current- and SOC-dependent resistance, worst at low temperature and low current |
| **Gradient boosting / neural network** | learn SOC from V, I, T and their recent averages | training data from a lab | only as good as the data it has seen (temperatures, ageing) |

The ground truth comes from the laboratory cycler's Ah counter (SOC = 1 + Ah / 2.78 Ah).
Realistic BMS imperfections are injected with the sliders: a wrong starting SOC and a biased current sensor.
"""
    )
