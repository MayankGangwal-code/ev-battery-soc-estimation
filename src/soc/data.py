"""Loaders for the processed data in ``data/``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import TEMP_ORDER, TEMPERATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _check(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python scripts/prepare_data.py` first")
    return path


def load_index(data_dir: Path | None = None) -> pd.DataFrame:
    """One row per raw drive-cycle file, with the validity decision and split."""
    return pd.read_csv(_check((data_dir or DATA_DIR) / "cycles_index.csv"))


def load_cycle(temp: str, cycle: str, data_dir: Path | None = None) -> pd.DataFrame:
    """A 1 Hz drive cycle: columns t_s, V, I, T, Ah, SOC."""
    return pd.read_csv(_check((data_dir or DATA_DIR) / "cycles" / temp / f"{cycle}.csv"))


def list_cycles(split: str | None = None, temps: list[str] | None = None, data_dir: Path | None = None) -> list[tuple[str, str]]:
    """(temp, cycle) pairs of the valid cycles, optionally filtered by split."""
    idx = load_index(data_dir)
    idx = idx[idx["valid"]]
    if split:
        idx = idx[idx["split"] == split]
    if temps:
        idx = idx[idx["temp"].isin(temps)]
    order = {t: i for i, t in enumerate(TEMP_ORDER)}
    idx = idx.assign(_o=idx["temp"].map(order)).sort_values(["_o", "cycle"])
    return list(zip(idx["temp"], idx["cycle"]))


def load_ocv_table(temp: str, data_dir: Path | None = None) -> pd.DataFrame:
    """OCV-SOC table (columns soc, ocv_discharge, ocv_charge, ocv) for one temperature."""
    return pd.read_csv(_check((data_dir or DATA_DIR) / "ocv" / f"{temp}.csv"))


def load_hppc(temp: str, data_dir: Path | None = None) -> pd.DataFrame:
    """HPPC results: rest OCV and pulse resistances per SOC level."""
    return pd.read_csv(_check((data_dir or DATA_DIR) / "hppc" / f"{temp}.csv"))


def load_capacity(data_dir: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(_check((data_dir or DATA_DIR) / "capacity.csv"))


def temp_label(temp: str) -> str:
    return f"{TEMPERATURES[temp]:+d} °C".replace("+", "") if TEMPERATURES[temp] >= 0 else f"{TEMPERATURES[temp]} °C"
