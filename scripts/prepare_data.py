"""Turn the raw LG 18650HG2 cycler files into the small, clean files used by the project.

Raw data: "LG 18650HG2 Li-ion Battery Data" (Kollmeyer, Vidal, Naguib, Skells — McMaster
University, 2020), https://data.mendeley.com/datasets/cp3473x7xv/3  (CC BY 4.0).
Point ``--raw`` at the folder that contains the temperature sub-folders
``n20degC, n10degC, 0degC, 10degC, 25degC, 40degC`` (the ``LG_HG2_data`` folder of the
download).

Outputs (all CSV, in ``data/``):

* ``cycles/<temp>/<cycle>.csv``  — drive cycles resampled to 1 Hz:
      t_s, V, I, T, Ah, SOC   (SOC = 1 + Ah / 2.78, the ground truth)
* ``ocv/<temp>.csv``             — OCV-SOC table from the C/20 discharge/charge test
* ``hppc/<temp>.csv``            — rest OCV points and pulse resistances from the HPPC test
* ``capacity.csv``               — measured capacities (C/20 and 1C) per temperature
* ``cycles_index.csv``           — one row per raw drive-cycle file, with the validity decision

Usage:  python scripts/prepare_data.py [--raw data/raw/LG_HG2_data]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc import C_REF_AH, TEMP_ORDER, TEMPERATURES, TEST_CYCLES, TRAIN_CYCLES  # noqa: E402

OUT = ROOT / "data"
CYCLE_NAMES = TRAIN_CYCLES + TEST_CYCLES


# --------------------------------------------------------------------------- #
# raw file reader
# --------------------------------------------------------------------------- #
def read_raw(path: Path) -> pd.DataFrame:
    """Read one Digatron CSV export -> columns t (s), V, I, T, Ah, step."""
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Time Stamp"))
    df = pd.read_csv(io.StringIO("".join(lines[hdr:])), skiprows=[1])
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    t = pd.to_timedelta(df["Prog Time"].astype(str)).dt.total_seconds().to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "t": t - t[0],
            "V": df["Voltage"].astype(float).to_numpy(),
            "I": df["Current"].astype(float).to_numpy(),
            "T": df["Temperature"].astype(float).to_numpy(),
            "Ah": df["Capacity"].astype(float).to_numpy(),
            "step": df["Step"].astype(int).to_numpy(),
        }
    )
    # drop blank / corrupt rows, and duplicated time stamps (keep the last)
    out = out.dropna(subset=["V", "I", "Ah"])
    out = out[~out["t"].duplicated(keep="last")].reset_index(drop=True)
    return out


def integrate_ah(t: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Trapezoidal charge throughput in Ah (positive while charging)."""
    t, i = np.asarray(t, dtype=float), np.asarray(i, dtype=float)
    inc = 0.5 * (i[1:] + i[:-1]) * np.diff(t) / 3600.0
    return np.concatenate([[0.0], np.cumsum(inc)])


def segments(mask: np.ndarray):
    """Yield (start, end) index pairs of contiguous True runs."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return
    d = np.diff(m.astype(int), prepend=0, append=0)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        yield s, e  # end exclusive


# --------------------------------------------------------------------------- #
# 1. drive cycles -> 1 Hz
# --------------------------------------------------------------------------- #
def resample_1hz(df: pd.DataFrame) -> pd.DataFrame:
    """Average V, I, T over each 1-second bin; Ah = last value in the bin.

    t_s marks the END of the bin, so SOC(t_s) is the state after the charge
    throughput of that second.
    """
    b = np.floor(df["t"].to_numpy()).astype(int)
    g = df.groupby(b)
    out = pd.DataFrame(
        {
            "t_s": g["t"].first().index.to_numpy() + 1,
            "V": g["V"].mean().to_numpy(),
            "I": g["I"].mean().to_numpy(),
            "T": g["T"].mean().to_numpy(),
            "Ah": g["Ah"].last().to_numpy(),
        }
    )
    # put the series on a strict 1 Hz grid: short logger gaps (a few seconds, rarely
    # minutes) are filled by linear interpolation of V, I, T and Ah, so that the
    # Ah label stays consistent with the integral of the interpolated current
    full = pd.RangeIndex(int(out["t_s"].iloc[0]), int(out["t_s"].iloc[-1]) + 1)
    out = out.set_index("t_s").reindex(full)
    n_filled = int(out["V"].isna().sum())
    out = out.interpolate(method="linear", limit_direction="both")
    out.index.name = "t_s"
    out = out.reset_index()
    out["SOC"] = 1.0 + out["Ah"] / C_REF_AH
    out.attrs["n_filled"] = n_filled
    return out


def process_cycles(raw: Path, log: list[str]) -> pd.DataFrame:
    rows = []
    for temp in TEMP_ORDER:
        for f in sorted((raw / temp).glob("*.csv")):
            parts = f.stem.split("_", 1)
            if len(parts) != 2 or parts[1] not in CYCLE_NAMES:
                continue
            test_id, cycle = parts
            df = read_raw(f)
            hours = df["t"].iloc[-1] / 3600
            ah_dis = -df["Ah"].min()
            t_mean = df["T"].mean()
            max_gap = float(np.diff(df["t"].to_numpy()).max())
            valid, reason = True, ""
            if ah_dis < 1.0:
                valid, reason = False, f"aborted test: only {ah_dis:.2f} Ah discharged"
            elif abs(t_mean - TEMPERATURES[temp]) > 6:
                valid, reason = False, f"mean temperature {t_mean:.1f} degC far from {TEMPERATURES[temp]} degC"
            elif max_gap > 60:
                valid, reason = False, f"logger gap of {max_gap:.0f} s in the recording"
            row = {
                "temp": temp, "temp_degC": TEMPERATURES[temp], "cycle": cycle, "test_id": test_id,
                "split": "train" if cycle in TRAIN_CYCLES else "test", "valid": valid, "reason": reason,
                "hours": round(hours, 2), "Ah_discharged": round(ah_dis, 3),
                "SOC_end": round(1 - ah_dis / C_REF_AH, 3), "V_start": round(df["V"].iloc[0], 3),
                "V_end": round(df["V"].iloc[-1], 3), "T_mean": round(t_mean, 1),
                "I_min": round(df["I"].min(), 2), "I_max": round(df["I"].max(), 2), "max_gap_s": round(max_gap, 1),
            }
            if valid:
                res = resample_1hz(df)
                row["n_rows"] = len(res)
                row["n_interpolated_s"] = res.attrs["n_filled"]
                d = OUT / "cycles" / temp
                d.mkdir(parents=True, exist_ok=True)
                res.round({"V": 4, "I": 4, "T": 2, "Ah": 5, "SOC": 5}).to_csv(d / f"{cycle}.csv", index=False)
            else:
                row["n_rows"] = 0
                row["n_interpolated_s"] = 0
                log.append(f"  excluded {temp}/{f.name}: {reason}")
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. OCV-SOC table from the C/20 test
# --------------------------------------------------------------------------- #
def ocv_table(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Average of the C/20 discharge and charge branches on a common SOC grid.

    The discharge branch starts at the full charge (SOC = 1).  The charge branch
    is mapped onto the same SOC axis by scaling its Ah so that it ends at SOC = 1
    (this removes the small coulombic-efficiency / CV-tail surplus).  Averaging
    the two branches cancels the ±R·I drop and most of the hysteresis.
    """
    i = df["I"].to_numpy()
    dis = max(segments(i < -0.1), key=lambda se: se[1] - se[0])
    chg = max(segments(i > 0.02), key=lambda se: se[1] - se[0])

    d = df.iloc[dis[0]:dis[1]]
    soc_d = 1.0 + (d["Ah"].to_numpy() - d["Ah"].iloc[0]) / C_REF_AH
    soc_end = soc_d[-1]

    c = df.iloc[chg[0]:chg[1]]
    ah_c = c["Ah"].to_numpy() - c["Ah"].iloc[0]
    soc_c = soc_end + ah_c / ah_c[-1] * (1.0 - soc_end)

    ocv_d = np.interp(grid, soc_d[::-1], d["V"].to_numpy()[::-1], left=np.nan, right=np.nan)
    ocv_c = np.interp(grid, soc_c, c["V"].to_numpy(), left=np.nan, right=np.nan)
    out = pd.DataFrame({"soc": grid, "ocv_discharge": ocv_d, "ocv_charge": ocv_c})
    out["ocv"] = out[["ocv_discharge", "ocv_charge"]].mean(axis=1)
    out = out.dropna(subset=["ocv"]).reset_index(drop=True)
    out.attrs["soc_end"] = float(soc_end)
    out.attrs["ah_discharge"] = float(d["Ah"].iloc[0] - d["Ah"].iloc[-1])
    out.attrs["ah_charge"] = float(ah_c[-1])
    return out


# --------------------------------------------------------------------------- #
# 3. HPPC: rest OCV points and pulse resistances
# --------------------------------------------------------------------------- #
def hppc_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per SOC level of the HPPC test.

    At each level the cell rested for 1 h (-> OCV = voltage at the end of the
    rest) and was then hit with a ~10 s discharge pulse (-> R0 from the voltage
    jump at the first sample, R10 from the drop after 10 s).
    """
    t, v, i, temp = df["t"].to_numpy(), df["V"].to_numpy(), df["I"].to_numpy(), df["T"].to_numpy()
    ah = integrate_ah(t, i)  # own integration: the cycler's counter resets inside this file
    soc = 1.0 + ah / C_REF_AH
    rest = np.abs(i) < 0.01
    pulse_segs = [(s, e) for s, e in segments(i < -0.3) if 8 <= t[e - 1] - t[s] <= 12]

    def pulse_after(idx):
        """R0 / R10 of the first ~10 s discharge pulse after sample idx."""
        for s, e in pulse_segs:
            if s > idx and rest[max(s - 50, 0):s].all():
                v_before = v[s - 1]
                return {"I_pulse": i[s:e].mean(), "R0_ohm": (v_before - v[s]) / abs(i[s]),
                        "R10_ohm": (v_before - v[e - 1]) / abs(i[s:e].mean())}
        return {"I_pulse": np.nan, "R0_ohm": np.nan, "R10_ohm": np.nan}

    # level 0: the test starts fully charged, after the rest that followed the charge
    rows = [{"soc": 1.0, "ocv_rest": v[0], "T": temp[0], "rest_s": np.nan, **pulse_after(0)}]
    last_soc = np.inf
    for s, e in segments(rest):
        if t[e - 1] - t[s] < 3000:
            continue
        if soc[e - 1] >= last_soc:
            break  # the recharge at the end of the file has started
        last_soc = soc[e - 1]
        rows.append({"soc": soc[e - 1], "ocv_rest": v[e - 1], "T": temp[s:e].mean(), "rest_s": t[e - 1] - t[s], **pulse_after(e - 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "LG_HG2_data"))
    args = ap.parse_args()
    raw = Path(args.raw)
    if not (raw / "25degC").exists():
        sys.exit(f"raw data not found under {raw}. Download it from https://data.mendeley.com/datasets/cp3473x7xv/3")

    log: list[str] = []
    print("1. drive cycles -> 1 Hz")
    index = process_cycles(raw, log)
    index.to_csv(OUT / "cycles_index.csv", index=False)
    print("\n".join(log))
    v = index[index["valid"]]
    print(f"   {len(v)} valid cycles of {len(index)} files; {int((v.split == 'train').sum())} train / {int((v.split == 'test').sum())} test")

    print("2. OCV tables from the C/20 tests")
    grid = np.round(np.arange(0.0, 1.0001, 0.005), 3)
    cap_rows = []
    (OUT / "ocv").mkdir(exist_ok=True)
    (OUT / "hppc").mkdir(exist_ok=True)
    for temp in TEMP_ORDER:
        f = next((raw / temp).glob("*C20DisCh.csv"))
        df = read_raw(f)
        tab = ocv_table(df, grid)
        tab.round(4).to_csv(OUT / "ocv" / f"{temp}.csv", index=False)
        cap_rows.append(
            {"temp": temp, "temp_degC": TEMPERATURES[temp], "test": "C/20 discharge", "file": f.name,
             "capacity_Ah": round(tab.attrs["ah_discharge"], 3), "T_mean": round(df["T"].mean(), 1)}
        )
        for g in sorted((raw / temp).glob("*Cap_1C.csv")):
            d = read_raw(g)
            cap_rows.append(
                {"temp": temp, "temp_degC": TEMPERATURES[temp], "test": "1C discharge", "file": g.name,
                 "capacity_Ah": round(-d["Ah"].min(), 3), "T_mean": round(d["T"].mean(), 1)}
            )
        print(f"   {temp:8s} C/20 capacity {tab.attrs['ah_discharge']:.3f} Ah, table covers SOC {tab.soc.min():.3f}-{tab.soc.max():.3f}")

        # HPPC
        h = next((raw / temp).glob("*HPPC.csv"))
        hp = hppc_table(read_raw(h))
        hp = hp[(hp["T"] - TEMPERATURES[temp]).abs() < 6]  # drop rests logged after the chamber was warmed up
        hp.round(5).to_csv(OUT / "hppc" / f"{temp}.csv", index=False)
        print(f"            HPPC: {len(hp)} SOC levels, R0 at ~90 % SOC = "
              f"{1000 * hp.loc[(hp.soc - 0.9).abs().idxmin(), 'R0_ohm']:.1f} mOhm")

    cap = pd.DataFrame(cap_rows)
    cap.to_csv(OUT / "capacity.csv", index=False)
    print(f"\nreference capacity C_REF = {C_REF_AH} Ah (C/20 at 25 degC). Files written under {OUT}")


if __name__ == "__main__":
    main()
