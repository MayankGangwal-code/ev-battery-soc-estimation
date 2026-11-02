# Interview preparation — EV Battery State-of-Charge Estimation

Everything here is true of the project in this repository. Numbers come from `reports/results.md`.

---

## 1. The 30-second pitch

> "State of charge is the fuel gauge of an electric vehicle, but you can't measure it directly — a battery
> management system has to *estimate* it from the voltage, current and temperature of the cells.
>
> I took a real EV cell dataset — an LG 18650 cell driven through standard drive cycles like UDDS and
> US06 at temperatures from −20 °C to +40 °C — and built the estimators a real BMS uses: **Coulomb
> counting**, the **open-circuit-voltage lookup**, and an **Extended Kalman Filter** on a Thevenin
> equivalent-circuit model of the cell. Then I trained a **neural network** on the same data.
>
> Under realistic conditions — a wrong starting SOC and a biased current sensor — Coulomb counting drifts to
> 9 % error, while the Kalman filter and the neural network both stay around 1 % across all temperatures —
> the Kalman filter slightly better on average and with a much smaller worst-case error. I also built an app
> where you can inject sensor errors and watch each estimator recover — or fail."

## 2. The 2-minute version

1. **Problem.** SOC is not measurable; it is inferred. Errors in SOC mean a wrong range display, wrong
   charging limits, over-discharge (damage) or wasted capacity (safety margin). Specs for a production BMS
   are typically 2–5 % error.
2. **Data.** The LG 18650HG2 dataset from McMaster University: a 3 Ah NMC cell in a thermal chamber,
   11–12 drive cycles at each of six temperatures (62 usable cycles), plus characterisation tests — a C/20
   discharge and charge, and a pulse (HPPC) test. Ground truth SOC comes from the laboratory cycler's Ah counter.
3. **Methods.** (a) Coulomb counting — integrate current; (b) OCV lookup — invert the OCV–SOC curve;
   (c) EKF — a 1-RC Thevenin model (R₀, R₁, C₁ per temperature, fitted by least squares on one training
   drive cycle) and a Kalman filter that predicts SOC with Coulomb counting and corrects it with the measured
   voltage; the OCV table and the filter's noise settings were chosen on the training cycles only;
   (d) gradient boosting and a neural network on V, I, T and their moving averages, trained on the "mixed" cycles.
4. **Evaluation.** 22 EPA drive cycles never used for fitting or training, at all six temperatures, with
   a 10 % initial-SOC error and a current sensor reading 2 % high with a 50 mA offset.
5. **Results.** Coulomb counting: 0.06 % ideal, 9.1 % realistic. OCV lookup: 17 % (33 % at −20 °C).
   EKF: 0.93 % overall — 0.63 % at 10–40 °C, 1.3 % below 0 °C — worst error 2.8 %, and the first voltage
   sample already removes the 10 % initial error. Neural network: 1.07 % overall, 1.7 % at −20 °C, worst
   error 7 %.
6. **Lessons.** Sensor errors, not the algorithm, limit Coulomb counting; the EKF needs the OCV curve in
   the *discharge* direction (hysteresis); a linear circuit model is good above 0 °C and marginal below, where
   the resistance depends strongly on current — exactly where the neural network earns its place.

---

## 3. Likely questions and short answers

### A. Battery basics

**Q1. What is SOC? What is SOH?**
SOC = remaining charge / full capacity, in %. SOH = present full capacity (or resistance) relative to
the new cell — ageing. I estimated SOC; SOH would be the next step.

**Q2. Why can't you just measure SOC from the voltage?**
Under load the terminal voltage = OCV − I·R − polarisation; the drop depends on current, temperature
and history. Only after a long rest does the voltage equal the OCV. My OCV-lookup baseline shows the
consequence: 8 % error at 25 °C and 33 % at −20 °C while driving.

**Q3. What is OCV and how did you measure the OCV–SOC curve?**
Open-circuit voltage: the equilibrium voltage of the cell at a given SOC. Two ways in the dataset: a very
slow C/20 discharge and charge (the voltage drop is only R·0.15 A), and the voltage after a 1-hour rest at
each 5–10 % SOC step of the pulse test. I compared both on the training cycles: the C/20 discharge curve
gave the best Kalman filter (the cold pulse test has only 9 SOC steps and a 1 h rest is too short for a cold
cell to relax), so that is the table the estimators use; the rest voltages are shown for comparison.

**Q4. What is hysteresis in a Li-ion cell?**
The OCV during charge is slightly higher than during discharge at the same SOC (≈25–40 mV for this NMC
cell above 50 % SOC, over 60 mV below 30 %) because of the lithium-insertion thermodynamics. It matters:
on the training cycles the charge curve biased my EKF by up to 19 % and the charge/discharge average by up
to 4.5 % at −20 °C, so a BMS must store the curve for the direction the cell mostly operates in — discharge.

**Q5. What is C-rate?**
Current relative to capacity: 1C discharges the full capacity in one hour. For a 3 Ah cell, 1C = 3 A,
C/20 = 0.15 A, 6C = 18 A (the peak in the US06 cycle).

**Q6. Why does the cell perform worse at low temperature?**
Ion transport in the electrolyte and the charge-transfer reaction slow down, so the internal resistance
rises — in the pulse test R₀ went from 10 mΩ at 40 °C to 73 mΩ at −20 °C. The I·R drop makes the voltage
hit the 2.8 V cut-off early, so the 1C capacity falls from 2.73 Ah at 25 °C to 1.67 Ah at −20 °C. The test
protocol discharges 95 % of that 1C capacity in each drive cycle, so at −20 °C the cycles end with ≈43 % of
the charge still inside the cell.

**Q7. What is the capacity of the cell and how did you define SOC = 0 %?**
Rated 3 Ah; measured 2.78 Ah by the C/20 discharge at 25 °C from 4.2 V to 2.8 V, which is my reference.
SOC = 1 + Ah/2.78, with Ah the charge counted by the cycler since the full charge. At 1C and 25 °C the cell
delivers 2.73 Ah, at 1C and −20 °C only 1.67 Ah.

**Q8. Why is the cell charged CC-CV?**
Constant current until 4.2 V, then constant voltage until the current falls to the cut-off (50 mA here).
The CV phase fills the last few percent without exceeding the voltage limit.

**Q9. What does a BMS do besides SOC?**
Cell-voltage and temperature monitoring, over-voltage/under-voltage/over-current/over-temperature
protection, cell balancing, SOH estimation, charging control, contactor control, communication with the
vehicle (CAN), and thermal-management requests.

**Q10. What is cell balancing?**
Cells in series drift apart in SOC. Passive balancing bleeds the fullest cells through resistors; active
balancing moves charge between cells. Without it the pack is limited by its weakest cell.

### B. The estimators

**Q11. Explain Coulomb counting. Why isn't it enough?**
SOC(t) = SOC₀ + (1/3600 C) ∫ I dt. It is exact with a perfect sensor and a known start (0.06 % in my test).
But every sensor error is integrated: a 50 mA offset adds 1.8 % SOC per hour, a 2 % gain error adds 2 % of
whatever was discharged, and a wrong SOC₀ is never corrected. Result: 9 % RMSE in my realistic test.

**Q12. How does a real BMS get SOC₀?**
From the OCV after a long rest (e.g. the car has been parked), and from the EKF while driving.

**Q13. What is the Thevenin / equivalent-circuit model?**
A voltage source OCV(SOC) in series with an ohmic resistance R₀ and one (or more) parallel R₁‖C₁ pairs:
V = OCV(SOC) + R₀·I + V₁, dV₁/dt = −V₁/(R₁C₁) + I/C₁. R₀ is the instantaneous drop, R₁C₁ the slow
relaxation (time constant τ ≈ 20–60 s here). I fitted R₀, R₁, τ by least squares on one drive cycle per
temperature; at 25 °C: R₀ = 19 mΩ, R₁ = 20 mΩ, τ = 27 s; at −20 °C: R₀ = 94 mΩ, R₁ = 97 mΩ, τ = 39 s.

**Q14. How does the Kalman filter estimate SOC?**
Two steps every second. *Predict*: propagate the state [SOC, V₁] with the model (Coulomb counting for SOC)
and grow its uncertainty P. *Correct*: compare the measured voltage with the model's prediction
OCV(SOC) + R₀I + V₁; multiply the difference by the Kalman gain K and add it to the state. K weighs the
model uncertainty against the voltage-sensor noise. Because OCV depends on SOC, a voltage error gets
translated into an SOC correction — that is what removes the drift and the wrong start.

**Q15. Why "extended"?**
The ordinary Kalman filter needs a linear model; OCV(SOC) is non-linear. The EKF linearises it at each
step with the slope dOCV/dSOC (the Jacobian H = [dOCV/dSOC, 1]).

**Q16. What are Q and R in your filter?**
Q is the process-noise covariance (how much I trust the model: current-sensor error for SOC, model error
for V₁); R is the measurement-noise variance of the voltage. Larger R → the filter trusts the voltage less
and corrects more slowly. I chose them by a 27-point grid search on the training cycles only
(`scripts/tune_ekf.py`): q_SOC = 1e-9, q_V1 = 1e-5, R = 1e-5 V² (≈3 mV).

**Q17. How fast does the EKF converge?**
In my tests almost instantly: from a start 10 % too low, the very first voltage sample brings the estimate
inside ±2 % on every test cycle, because each cycle starts from a fully charged cell, where the OCV curve is
steep and one voltage reading is very informative. Starting 30 or 50 % wrong at 25 °C gives the same
picture (see the convergence figure). It would be slower on the flat part of the OCV curve, e.g. starting
in the middle of a drive at 50 % SOC — there the filter has to rely on Coulomb counting for longer.

**Q18. Where does the EKF fail and why?**
Its worst errors (up to 2.8 %) are a slow drift on the long, low-current UDDS cycles, and its error is
larger below 0 °C (1.3 % vs 0.6 % RMSE). The reason is the model: the overpotential is non-linear in
current — on the −20 °C training cycle the effective resistance (OCV − V)/|I| is ≈260 mΩ at 1 A but only
≈123 mΩ at 6 A (Butler–Volmer kinetics) — and it depends on SOC. A single R₀ cannot represent that. The
fixes are a 2-RC model with current- and SOC-dependent parameter tables, or the neural network.

**Q19. What inputs did the neural network get, and why not the Ah counter?**
V, I, T and causal moving averages of V and I over 10 s, 1 min, 5 min and 15 min — 11 features, all
computable in real time from the past. No Ah counter, so the network needs no initial SOC and cannot drift.
A 64-64 feed-forward network trained on 261,003 samples from the mixed cycles.

**Q20. How did you avoid data leakage?**
Train on the 40 "Mixed" cycles, test only on the 22 standard EPA cycles; the ECM parameters and EKF tuning
used training cycles only; all features are causal (a test in the repo checks that changing a future sample
cannot change earlier features).

**Q21. Why not an LSTM?**
It is the natural next step (Chemali et al. 2018 reported ~1 % with an LSTM on similar data). The
moving-average features give the feed-forward network a short memory of the load history, which was
enough to reach 1.07 %. An LSTM would be my first extension.

**Q22. Which estimator would you ship?**
The EKF as the primary estimator — physically interpretable, a built-in uncertainty estimate, the smallest
worst-case error (2.8 %) and no dependence on training data — with the neural network as an independent
second opinion, and a consistency check between the two to flag sensor faults or an aged cell.

**Q23. How would you handle an aged cell?**
The capacity C in Coulomb counting and the OCV table shift with ageing. A BMS estimates SOH (capacity and
resistance) online — e.g. by comparing Ah throughput between two rested OCV readings — and updates C and
the ECM tables. Not covered in this project.

### C. Fundamentals they may jump to

**Q24. Li-ion cell construction?** Graphite anode, metal-oxide cathode (NMC here), separator, organic
electrolyte with LiPF₆, current collectors (Cu anode, Al cathode). Lithium ions shuttle between the
electrodes; electrons flow through the external circuit.

**Q25. NMC vs LFP?** NMC: higher energy density, 3.6–3.7 V nominal, sloped OCV curve (easy SOC estimation
from voltage). LFP: 3.2 V nominal, very flat OCV curve (hard for SOC — Coulomb counting dominates), safer,
longer life, cheaper.

**Q26. Why 2.8 V and 4.2 V limits?** Above 4.2 V the cathode degrades and lithium can plate on the anode.
The datasheet discharge cut-off is 2.5 V (the dataset used 2.8 V): below it there is almost no usable energy
left, and a deep over-discharge well below that (towards 1.5 V) dissolves the copper current collector and
permanently damages the cell.

**Q27. What is internal resistance made of?** Ohmic (electrolyte, contacts, collectors — instantaneous,
R₀), charge-transfer (the electrochemical reaction, ms–s), and diffusion (lithium moving in the solid
particles, seconds–minutes). The RC branch of the model lumps the last two.

**Q28. Energy vs power density?** Energy (Wh/kg) sets range; power (W/kg) sets acceleration and fast
charging. An 18650HG2 is a "power" cell (20 A continuous discharge).

**Q29. What is thermal runaway?** A self-accelerating exothermic reaction (separator melts → internal
short → more heat → electrolyte decomposition). The separator melts at roughly 130–165 °C and full runaway
of an NMC cell typically sets in above ~150–200 °C; the BMS prevents it by limiting voltage, current and
temperature.

**Q30. Series vs parallel in a pack?** Series adds voltage (e.g. 96 cells × 3.7 V ≈ 355 V), parallel adds
capacity. A 96s2p pack has 96 series groups of 2 parallel cells.

**Q31. What is regenerative braking from the battery's view?** A charging current pulse (positive in my
data, up to 6 A = 2C). At low temperature the charge current must be limited to avoid lithium plating —
the dataset reduced the regen limit below 10 °C for exactly that reason.

**Q32. What is the Kalman gain intuitively?** A number between 0 and 1 that says how much to believe the
new measurement versus the prediction: K ≈ 0 when the measurement is noisy or the model is confident,
K ≈ 1 when the model is uncertain (e.g. right after a wrong initialisation).

**Q33. Difference between EKF and UKF / particle filter?** EKF linearises analytically (fast, standard in
BMSs); the unscented KF propagates sigma points through the non-linearity (better for strong
non-linearity, ~3× cost); particle filters handle non-Gaussian cases at much higher cost.

**Q34. Why least squares for the model parameters?** With known SOC from the lab, V − OCV(SOC) must equal
R₀I + V₁; minimising the squared voltage error over a whole cycle gives R₀, R₁, τ. Fit RMS was 10 mV at
40 °C and 86 mV at −20 °C — a direct measure of how well the linear model fits.

**Q35. How would you implement the EKF on a microcontroller?** It is a 2-state filter: a handful of
multiplications per second, OCV and parameter look-up tables in flash, fixed-point arithmetic if needed —
negligible for any automotive MCU. The neural network (64-64) is also small: ~5,000 multiply-adds per second.

**Q36. What sampling rate did you use and why?** 1 Hz, the typical rate of a BMS SOC task; the lab data was
10 Hz and was averaged per second so the Ah integral is preserved.

**Q37. How did you validate your code?** 18 unit tests, including a synthetic cell with known R₀/R₁/τ: the
fit recovers the parameters, Coulomb counting matches the simulation exactly, and the EKF converges from a
30 % wrong start with a biased sensor.

**Q38. What's the accuracy of the ground truth itself?** The label is the cycler's Ah counter divided by a
fixed 2.78 Ah, but the cell's capacity measurably fades during the campaign (the 1C capacity at 25 °C dropped
1.2 % between two tests a few days apart) and the cycler's current has its own tolerance, so the labels are
only good to about a percent; differences between estimators below that are not meaningful.

**Q39. What about measuring SOC with impedance (EIS)?** Possible in the lab but impractical online; some
BMSs inject small AC signals to track resistance for SOH, not SOC.

**Q40. What would you do with more time?** LSTM/GRU sequence model; 2-RC model with current- and
SOC-dependent tables; SOH estimation; test on the Panasonic 18650PF dataset for generalisation; deploy the
EKF in C on an STM32 with a current shunt and compare with the lab reference.

---

## 4. Three things not to say

* Do not say the neural network "replaces" the Kalman filter. Say they complement each other.
* Do not quote "1 %" without context. Say **"0.93 % RMSE for the Kalman filter and 1.07 % for the neural
  network on 22 unseen drive cycles across −20 to 40 °C, with a worst error of 2.8 % for the filter and 7 %
  for the network."**
* Do not claim the model works for other cells or aged cells — it was fitted to one new LG HG2 cell.

## 5. If they ask you to draw something

Draw the **Thevenin model**: a voltage source labelled OCV(SOC), then R₀ in series, then R₁ in parallel
with C₁, then the terminals. Write V = OCV(SOC) + R₀I + V₁ next to it. Then draw the **EKF loop**: a box
"predict (model)" → box "correct (voltage)" → back, with "K = Kalman gain" on the arrow. Finally sketch the
**OCV curve**: voltage rising from 3.0 V at 0 % to 4.2 V at 100 %, steep at the ends, flatter in the middle
— and say "the flat part is where the voltage tells you the least, so there Coulomb counting carries the
estimate."
