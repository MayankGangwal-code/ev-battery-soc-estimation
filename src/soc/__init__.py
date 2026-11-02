"""
soc — State-of-charge estimation for a lithium-ion EV cell (LG 18650HG2).

Sub-modules
-----------
data      : load the processed drive cycles, OCV tables and HPPC results
ocv       : open-circuit-voltage table (interpolation, slope, inverse lookup)
coulomb   : Coulomb counting with optional current-sensor errors
ecm       : 1-RC Thevenin equivalent-circuit model and parameter fitting
ekf       : Extended Kalman Filter SOC estimator built on the ECM
features  : causal features for the machine-learning estimators
models    : scikit-learn regressors used in the comparison
evaluate  : error metrics
plots     : figures for the report

Sign convention (same as the dataset): current I > 0 charges the cell,
I < 0 discharges it. SOC is a fraction in [0, 1].
"""

__version__ = "1.0.0"

# Reference capacity: charge removed by a C/20 discharge from 4.2 V (CC-CV,
# 50 mA cut-off) to 2.8 V at 25 degC — measured 2.78 Ah (nominal rating 3.0 Ah).
# SOC = 1 + Ah / C_REF, where Ah is the charge throughput since the full charge
# (negative while discharging).
C_REF_AH = 2.78

CELL = {
    "model": "LG 18650HG2",
    "chemistry": "NMC / graphite",
    "nominal_capacity_ah": 3.0,
    "nominal_voltage_v": 3.6,
    "v_max": 4.2,
    "v_min": 2.8,
}

# Temperature folders in the raw dataset -> nominal ambient temperature (degC)
TEMPERATURES = {"n20degC": -20, "n10degC": -10, "0degC": 0, "10degC": 10, "25degC": 25, "40degC": 40}
TEMP_ORDER = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]

# Drive cycles: the eight random mixes are used for training / parameter
# fitting, the four standard EPA cycles are kept for testing.
TRAIN_CYCLES = [f"Mixed{i}" for i in range(1, 9)]
TEST_CYCLES = ["UDDS", "HWFET", "LA92", "US06"]

CYCLE_DESCRIPTIONS = {
    "UDDS": "Urban Dynamometer Driving Schedule — city driving, stop-and-go",
    "HWFET": "Highway Fuel Economy Test — steady highway driving",
    "LA92": "LA92 'Unified' cycle — aggressive city driving",
    "US06": "US06 — high speed, high acceleration",
    "Mixed": "random mix of UDDS, HWFET, LA92 and US06",
}

# Sampling rate of the processed cycles
SAMPLE_PERIOD_S = 1.0
