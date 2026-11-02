# Resume lines for this project

All numbers are from `reports/results.md`.

## Project title (as it should appear)

**EV Battery State-of-Charge Estimation — Kalman Filter vs. Machine Learning** — Python, NumPy/SciPy,
Scikit-learn, Streamlit · [GitHub](https://github.com/MayankGangwal-code/ev-battery-soc-estimation)

## Version A — 3 bullets (recommended)

* Built a BMS-style SOC estimation pipeline for an LG 18650 NMC cell on 62 real EV drive cycles
  (UDDS, HWFET, LA92, US06 and random mixes) at −20 … 40 °C: Coulomb counting, OCV lookup, and an Extended
  Kalman Filter on a 1-RC Thevenin model (R₀, R₁, C₁ fitted by least squares on drive-cycle data at each
  temperature; OCV table from C/20 and pulse tests).
* Benchmarked against gradient boosting and a neural network on causal V/I/T features under realistic
  conditions (10 % initial-SOC error, biased current sensor) on 22 unseen cycles: Coulomb counting drifted
  to 9.1 % RMSE, the EKF reached 0.93 % RMSE with a worst error of 2.8 %, and the neural network 1.07 % RMSE.
* Shipped a Streamlit app and CLI that run all estimators on any BMS log with adjustable sensor errors;
  18 unit tests including EKF validation on a synthetic cell; reproducible report with all figures.

## Version B — 2 bullets (tight)

* Estimated EV battery SOC from real drive-cycle data (−20 … 40 °C) with Coulomb counting, an Extended
  Kalman Filter on a fitted Thevenin model, and ML (gradient boosting, neural network); EKF 0.93 % and
  neural network 1.07 % RMSE on 22 unseen cycles vs. 9.1 % for drifting Coulomb counting.
* Built the full BMS workflow: OCV/pulse-test characterisation, least-squares model fitting, causal feature
  design, leakage-free evaluation on 22 held-out cycles, Streamlit demo with injectable sensor errors.

## Version C — one line

* **EV battery SOC estimation** — EKF on a Thevenin cell model vs. neural network on 62 real drive cycles
  at −20 … 40 °C; 0.93 % / 1.07 % RMSE vs. 9 % for Coulomb counting; Streamlit app.

## Skills line additions

Battery / BMS: SOC estimation, Coulomb counting, OCV characterisation, HPPC, equivalent-circuit modelling,
Extended Kalman Filter · Signal processing & control: state estimation, least-squares identification ·
ML: feature engineering, gradient boosting, neural networks, leakage-free validation.

## Where to put it

For **Vecmocon, EV, BMS and power-electronics roles** put this project first. For PSU / core-power roles
put the transformer DGA project first and this one second. The two together cover power systems,
machines and e-mobility.
