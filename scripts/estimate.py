"""Estimate SOC from a BMS log on the command line.

The log is a CSV sampled at 1 Hz with columns ``t_s, V, I, T``
(seconds, cell volts, amps with charging positive, cell temperature in °C).

    python scripts/estimate.py my_log.csv                     # writes my_log_soc.csv
    python scripts/estimate.py my_log.csv --soc0 0.8          # tell the BMS where it starts
    python scripts/estimate.py --demo 25degC UDDS             # run on a dataset cycle, with metrics
    python scripts/estimate.py --demo n20degC US06 --offset 0.05 --gain 1.02 --soc0-error -0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc.bms import CellModel, estimate_all, load_ml_models  # noqa: E402
from soc.coulomb import SensorErrors  # noqa: E402
from soc.data import load_cycle  # noqa: E402
from soc.evaluate import metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="SOC estimation: Coulomb counting, OCV lookup, EKF, gradient boosting, neural network")
    ap.add_argument("csv", nargs="?", help="1 Hz log with columns t_s, V, I, T")
    ap.add_argument("--demo", nargs=2, metavar=("TEMP", "CYCLE"), help="use a dataset cycle, e.g. 25degC UDDS")
    ap.add_argument("--soc0", type=float, default=None, help="SOC the BMS assumes at the start (default: from the first voltage)")
    ap.add_argument("--soc0-error", type=float, default=0.0, help="demo only: offset added to the true initial SOC")
    ap.add_argument("--offset", type=float, default=0.0, help="current-sensor offset in A added before estimation")
    ap.add_argument("--gain", type=float, default=1.0, help="current-sensor gain error (1.02 = reads 2 %% high)")
    ap.add_argument("--no-ml", action="store_true", help="skip the machine-learning estimators")
    ap.add_argument("--out", default=None, help="output CSV (default: <input>_soc.csv)")
    args = ap.parse_args()

    cell = CellModel.from_json()
    ml = None if args.no_ml else load_ml_models()
    sensor = SensorErrors(offset_a=args.offset, gain=args.gain) if (args.offset or args.gain != 1.0) else None

    if args.demo:
        temp, cyc = args.demo
        df = load_cycle(temp, cyc)
        soc0 = args.soc0 if args.soc0 is not None else float(df["SOC"].iloc[0] + args.soc0_error)
        truth = df["SOC"].to_numpy()
        out_path = Path(args.out) if args.out else ROOT / "reports" / f"demo_{temp}_{cyc}_soc.csv"
    else:
        if not args.csv:
            ap.error("give a CSV file or use --demo TEMP CYCLE")
        df = pd.read_csv(args.csv)
        missing = [c for c in ("t_s", "V", "I", "T") if c not in df.columns]
        if missing:
            ap.error(f"missing columns in {args.csv}: {missing}")
        soc0, truth = args.soc0, None
        out_path = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_soc.csv")

    est = estimate_all(df, cell, ml, soc0_guess=soc0, sensor=sensor)
    if truth is not None:
        est["SOC_true"] = truth
    est.round(5).to_csv(out_path, index=False)

    print(f"{len(df)} samples ({df['t_s'].iloc[-1] / 3600:.2f} h), mean cell temperature {df['T'].mean():.1f} °C")
    print(f"initial SOC assumed by the BMS: {100 * (soc0 if soc0 is not None else est['OCV lookup'].iloc[0]):.1f} %")
    cols = [c for c in est.columns if c not in ("t_s", "EKF_std", "SOC_true")]
    print("\nfinal SOC estimate:")
    for c in cols:
        line = f"  {c:18s} {100 * est[c].iloc[-1]:6.1f} %"
        if truth is not None:
            m = metrics(truth, est[c])
            line += f"   (RMSE over the cycle {m['rmse']:.2f} %, max error {m['max_abs']:.1f} %)"
        print(line)
    if truth is not None:
        print(f"  {'true SOC':18s} {100 * truth[-1]:6.1f} %")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
