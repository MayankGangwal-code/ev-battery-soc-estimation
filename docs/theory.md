# Theory cheat-sheet — Li-ion cells, BMS and SOC estimation

## 1. The cell

| Quantity | LG 18650HG2 (this project) |
|---|---|
| Chemistry | NMC cathode / graphite anode |
| Nominal capacity | 3.0 Ah (measured 2.78 Ah at C/20, 2.73 Ah at 1C, 25 °C) |
| Voltage window | 2.5 V – 4.2 V on the datasheet; 2.8 V lower limit in this dataset (nominal 3.6 V) |
| Max continuous discharge | 20 A (≈ 6.7C) |
| Format | 18650 cylindrical: 18 mm diameter, 65 mm long, ≈ 47 g |

**C-rate**: current / capacity. 1C = 3 A empties the cell in 1 h; C/20 = 0.15 A takes 20 h.
**CC-CV charging**: constant current to 4.2 V, then constant voltage until I < 50 mA.

## 2. Definitions

* **SOC** = Q_remaining / Q_full. Here SOC = 1 + Ah/2.78 with Ah the charge counted since full charge
  (negative while discharging). Plett's definition: Q_full is the charge removable at a slow rate at 25 °C.
* **DOD** = 1 − SOC. **SOH** = Q_full(now) / Q_full(new) (capacity fade) or R(now)/R(new) (resistance growth).
* **OCV**: open-circuit (equilibrium) voltage, a monotonic function of SOC, weakly temperature-dependent,
  with charge/discharge hysteresis.
* **Overpotential**: terminal voltage minus OCV under load — ohmic (R₀·I), charge-transfer and diffusion.

## 3. Why SOC cannot be measured

Under load: V = OCV(SOC) − I·R₀ − V_polarisation (discharge). R₀ and the polarisation depend on
temperature (×7 from 40 °C to −20 °C), current (Butler–Volmer: non-linear), SOC and age. So the
voltage alone gives SOC only after a long rest; while driving an estimator is needed.

## 4. Estimators

### Coulomb counting
SOC(t) = SOC₀ + (1 / 3600·C) ∫₀ᵗ I dt.  Errors: SOC₀ unknown; current offset δ → drift δ·t/(3600 C);
gain error ε → error ε·(Ah throughput)/C; capacity C wrong (ageing). Never self-correcting.

### OCV lookup
SOC = OCV⁻¹(V). Valid only at rest (or after compensating I·R). Used by BMSs to initialise SOC after
parking.

### Equivalent-circuit model (ECM), 1-RC Thevenin
```
V = OCV(SOC) + R₀·I + V₁          (I > 0 charging)
dV₁/dt = −V₁/(R₁C₁) + I/C₁       τ = R₁C₁
discrete (step Δt):  V₁[k+1] = a·V₁[k] + R₁(1−a)·I[k],   a = exp(−Δt/τ)
                     SOC[k+1] = SOC[k] + Δt·I[k]/(3600 C)
```
Fitted values (this cell): 25 °C R₀ = 19 mΩ, R₁ = 20 mΩ, τ = 27 s; −20 °C R₀ = 94 mΩ, R₁ = 97 mΩ, τ = 39 s.
Parameters are stored in tables versus temperature (and, in production BMSs, versus SOC and current).

### Extended Kalman Filter (Plett 2004)
State x = [SOC, V₁], input I, measurement V.
```
predict:  x⁻ = f(x, I)                 P⁻ = A P Aᵀ + Q
correct:  ŷ = OCV(SOC⁻) + R₀ I + V₁⁻    H = [dOCV/dSOC, 1]
          K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹
          x = x⁻ + K (V_meas − ŷ)       P = (I − K H) P⁻
```
Q = process noise (model + current-sensor error), R = voltage-noise variance, P = state covariance
(its √P₁₁ is the filter's own SOC uncertainty). The EKF linearises OCV(SOC) at each step; when the OCV
curve is flat (dOCV/dSOC small) the voltage carries little information and the estimate relies on
Coulomb counting.

### Data-driven estimators
Regression from causal features (V, I, T, moving averages) to SOC. No initial SOC, no drift; accuracy
limited to the operating conditions seen in training. Sequence models (LSTM/GRU) add memory.

## 5. Characterisation tests (how a BMS is calibrated)

| Test | What it gives |
|---|---|
| C/20 (or C/30) discharge + charge | capacity; OCV curves (the I·R drop is only ≈ 0.15 A·R; charge and discharge branches differ by the hysteresis — store the branch for the direction the cell mostly operates in) |
| HPPC pulse test (10 s pulses at each 5–10 % SOC after 1 h rest) | OCV at rest; R₀ = ΔV/ΔI at the first sample; polarisation from the 10 s drop |
| Capacity tests at several C-rates and temperatures | rate and temperature capability (Peukert-like behaviour) |
| Drive cycles | validation of the model and estimators; training data for ML |

## 6. Numbers worth remembering (this project)

* R₀ (0.1 s pulse): 10 mΩ (40 °C), 12 mΩ (25 °C), 31 mΩ (0 °C), 73 mΩ (−20 °C).
* 1C capacity: 2.73 Ah at 25 °C, 1.67 Ah at −20 °C (61 %).
* Drive-cycle currents: peaks −18 A (6C) discharge, +6 A (2C) regen at ≥ 10 °C.
* Coulomb-counting drift: 50 mA offset → 1.8 % SOC per hour.
* EKF settling from a 10 % wrong start: the first voltage sample brings it inside ±2 % on every test cycle
  (the cycles start from a fully charged cell, where the OCV curve is steep).
* RMSE on 22 unseen cycles: Coulomb counting (realistic) 9.1 %, OCV lookup 17.3 %, EKF 0.93 % (0.63 % at
  10–40 °C, 1.3 % below 0 °C, worst error 2.8 %), gradient boosting 2.07 %, neural network 1.07 % (worst 7.1 %).
* OCV table choice (training cycles, EKF RMSE): C/20 discharge 1.05 %, charge/discharge average 2.53 %,
  pulse-test rest voltages 2.74 %, C/20 charge 6.23 %.
* Effective resistance (OCV − V)/|I| on the −20 °C training cycle: ≈260 mΩ at 1 A, ≈123 mΩ at 6 A.

## 7. Standard drive cycles

* **UDDS** — Urban Dynamometer Driving Schedule (city, 1369 s, max 91 km/h).
* **HWFET** — Highway Fuel Economy Test (765 s, steady ≈ 77 km/h).
* **LA92** — "Unified" cycle, more aggressive city driving than UDDS.
* **US06** — high speed / high acceleration supplement (max 129 km/h) — the hardest for the battery.
The dataset repeats each profile, scaled to one cell of a compact EV pack, until 95 % of the 1C capacity
at that temperature is discharged.
