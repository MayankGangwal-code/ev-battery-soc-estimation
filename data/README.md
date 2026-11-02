# Data

## Source

**LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script**
Philip Kollmeyer, Carlos Vidal, Mina Naguib, Michael Skells — McMaster University, Hamilton, Canada.
Mendeley Data, V3 (2020), https://doi.org/10.17632/cp3473x7xv.3 — licence CC BY 4.0.

A brand-new 3 Ah LG HG2 cell was tested in a thermal chamber with a 75 A / 5 V Digatron Firing Circuits
battery tester (voltage and current accuracy 0.1 % of full scale). At each of six ambient temperatures
(40, 25, 10, 0, −10, −20 °C) the test sequence was: HPPC pulse test, C/20 discharge and charge, 0.5C / 1C / 2C
discharge tests, the four EPA drive cycles (UDDS, HWFET, LA92, US06) and eight random mixes of them. Drive
cycles use the power profile of a single cell in a compact EV and run until 95 % of the 1C capacity at that
temperature has been discharged. Below 10 °C the regenerative (charging) current limit was reduced.

The raw files (~600 MB of 10 Hz cycler exports) are **not** included; this folder holds the processed data
written by `scripts/prepare_data.py`. To regenerate it, download the dataset, unzip it, and run
`python scripts/prepare_data.py --raw <folder containing 25degC/, n20degC/, ...>`.

## Files

| Path | Content |
|---|---|
| `cycles/<temp>/<cycle>.csv` | drive cycle at 1 Hz: `t_s` (s, end of each 1-s bin), `V` (V), `I` (A, charging positive), `T` (°C, cell surface), `Ah` (cycler charge counter since full charge), `SOC` (= 1 + Ah/2.78) |
| `cycles_index.csv` | one row per raw drive-cycle file: temperature, cycle, test id, train/test split, validity decision and reason, duration, Ah discharged, SOC at the end, start/end voltage, mean temperature, current range, longest logger gap, interpolated seconds |
| `ocv/<temp>.csv` | OCV–SOC table from the C/20 test on a 0.5 % grid: discharge branch (the table used by the estimators), charge branch (Ah scaled to end at 100 %) and their average |
| `hppc/<temp>.csv` | pulse test, one row per SOC level: `soc`, `ocv_rest` (voltage after the 1 h rest), `T`, `rest_s`, `I_pulse`, `R0_ohm` (ΔV/I at the first 0.1 s sample), `R10_ohm` (after 10 s) |
| `capacity.csv` | measured capacities: C/20 discharge and 1C tests per temperature (with the mean cell temperature of each test) |

Temperature folders: `n20degC` = −20 °C, `n10degC` = −10 °C, `0degC`, `10degC`, `25degC`, `40degC`.

## Processing decisions

* **1 Hz resampling.** V, I and T are averaged over each second; Ah is the counter value at the end of the
  second, so the SOC label is consistent with the integral of the averaged current (ideal Coulomb counting
  reproduces the label with 0.06 % RMSE; worst single-sample difference 0.27 %).
* **Logger gaps.** Gaps of a few seconds (24 s in total over all 62 cycles) are filled by linear interpolation; files with a
  gap longer than 60 s are excluded (`10degC/LA92`: 155 s, `n10degC/Mixed4`: 416 s).
* **Aborted tests** with less than 1 Ah throughput are excluded (`25degC/HWFET` — empty, `40degC/Mixed4–8` —
  stopped after 0.19 Ah at 23 °C).
* **SOC reference.** C_REF = 2.78 Ah, the charge removed by the C/20 discharge at 25 °C from a full CC-CV
  charge to 2.8 V. SOC = 100 % at the start of every cycle (the cell was fully charged before each one).
  Each drive cycle was run until 95 % of the 1C capacity at that temperature had been discharged, so with
  this absolute definition the cycles end at ≈ 7 % SOC at 25 °C and ≈ 43 % at −20 °C (where the 1C capacity
  is only 1.67 Ah because the voltage reaches the cut-off early at that rate).
* **HPPC Ah.** The cycler's counter is reset inside the HPPC files, so the SOC of the pulse-test levels is
  obtained by integrating the measured current from the start of the test (full charge).
* **Capacity test anomalies** are kept for transparency in `capacity.csv`: the 1C test at "40 °C" (test 557)
  and the first one at "10 °C" (test 571) were run with the cell at ≈ 5 °C and ≈ 1 °C respectively. They
  are not used by the estimators.

## Split

| Split | Cycles | Use |
|---|---|---|
| train | Mixed1–Mixed8 at every temperature (40 valid cycles, 261,003 samples) | ECM parameter fit (Mixed1), EKF tuning, ML training |
| test | UDDS, HWFET, LA92, US06 at every temperature (22 valid cycles) | all reported results |
