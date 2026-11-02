"""Figures for the report (matplotlib, PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import TEMP_ORDER, TEMPERATURES  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

METHOD_COLOURS = {
    "True SOC": INK,
    "Coulomb counting (ideal)": "#eda100",
    "Coulomb counting (realistic)": "#eb6834",
    "OCV lookup": "#e87ba4",
    "EKF": "#2a78d6",
    "Gradient Boosting": "#1baf7a",
    "Neural Network": "#6250d6",
}
METHOD_STYLES = {"Coulomb counting (ideal)": "--"}
# one-hue ramp for the six temperatures (cold = light, hot = dark)
TEMP_COLOURS = dict(zip(TEMP_ORDER, ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]))

plt.rcParams.update(
    {
        "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12, "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    }
)


def _save(fig, path):
    if path is None:
        return fig
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def temp_label(temp: str) -> str:
    return f"{TEMPERATURES[temp]} °C"


# --------------------------------------------------------------------------- #
def plot_soc_traces(t_s, soc_true, estimates: dict, title: str, path=None, show_error=True):
    """SOC versus time for the truth and several estimators, plus the error panel."""
    h = np.asarray(t_s, float) / 3600.0
    if show_error:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax2 = None
    ax.plot(h, 100 * np.asarray(soc_true), color=INK, lw=1.8, label="True SOC", zorder=5)
    for name, est in estimates.items():
        ax.plot(h, 100 * np.asarray(est), color=METHOD_COLOURS.get(name, MUTED), lw=1.3, ls=METHOD_STYLES.get(name, "-"), label=name)
        if ax2 is not None:
            ax2.plot(h, 100 * (np.asarray(est) - np.asarray(soc_true)), color=METHOD_COLOURS.get(name, MUTED), lw=1.1, ls=METHOD_STYLES.get(name, "-"))
    ax.set_ylabel("SOC (%)")
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    if ax2 is not None:
        ax2.axhline(0, color=AXIS, lw=0.8)
        ax2.set_ylabel("error (% SOC)")
        ax2.set_xlabel("time (h)")
        lim = min(25, max(5, 1.1 * np.nanmax([np.nanmax(np.abs(100 * (np.asarray(e) - np.asarray(soc_true)))) for e in estimates.values()])))
        ax2.set_ylim(-lim, lim)
    else:
        ax.set_xlabel("time (h)")
    fig.tight_layout()
    return _save(fig, path)


def plot_rmse_by_temperature(table: pd.DataFrame, path=None, title="RMSE of each estimator by temperature (test cycles)", ymax=None):
    """table: index = method, columns = temp folder names, values = RMSE in % SOC."""
    cols = [t for t in TEMP_ORDER if t in table.columns]
    methods = list(table.index)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(cols))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        vals = table.loc[m, cols].to_numpy(float)
        bars = ax.bar(x + (k - (len(methods) - 1) / 2) * (w + 0.01), vals, width=w, color=METHOD_COLOURS.get(m, MUTED), label=m)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, min(v, ymax or v) + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(x, [temp_label(t) for t in cols])
    ax.set_ylabel("RMSE (% SOC)")
    if ymax:
        ax.set_ylim(0, ymax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_title(title, loc="left", pad=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_ocv_curves(tables: dict, hppc: dict, path=None, used: str = "C/20 discharge"):
    """Left: the OCV-SOC table used by the estimators at every temperature.
    Right: hysteresis at 25 °C (C/20 charge vs discharge, their average, pulse-test rests)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for temp in TEMP_ORDER:
        t = tables[temp]
        ax.plot(100 * t["soc"], t["ocv_discharge"], lw=1.5, color=TEMP_COLOURS[temp], label=temp_label(temp))
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("OCV (V)")
    ax.set_title(f"OCV–SOC table used by the estimators ({used})", loc="left", pad=10)
    ax.legend(fontsize=8, title="cell temperature", title_fontsize=8)

    t = tables["25degC"]
    ax2.plot(100 * t["soc"], t["ocv_discharge"], color="#eb6834", lw=1.3, label="C/20 discharge (used)")
    ax2.plot(100 * t["soc"], t["ocv_charge"], color="#1baf7a", lw=1.3, label="C/20 charge")
    ax2.plot(100 * t["soc"], t["ocv"], color=INK, lw=1.3, ls="--", label="average")
    h = hppc["25degC"].dropna(subset=["ocv_rest"])
    ax2.scatter(100 * h["soc"], h["ocv_rest"], color="#2a78d6", s=28, zorder=5, label="1 h rest (pulse test)")
    ax2.set_xlabel("SOC (%)")
    ax2.set_ylabel("voltage (V)")
    ax2.set_title("Hysteresis at 25 °C: charge vs discharge curves", loc="left", pad=10)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_resistance(hppc: dict, ecm: dict, path=None):
    """Internal resistance versus temperature from the pulse test and the ECM fit."""
    temps = [TEMPERATURES[t] for t in TEMP_ORDER]
    r0 = [1000 * hppc[t]["R0_ohm"].median() for t in TEMP_ORDER]
    r10 = [1000 * hppc[t]["R10_ohm"].median() for t in TEMP_ORDER]
    rfit = [1000 * ecm[t]["R0"] for t in TEMP_ORDER]
    rtot = [1000 * (ecm[t]["R0"] + ecm[t]["R1"]) for t in TEMP_ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(temps, r0, marker="o", color="#2a78d6", lw=1.4, label="pulse test, 0.1 s (R₀)")
    ax.plot(temps, r10, marker="s", color="#eb6834", lw=1.4, label="pulse test, 10 s")
    ax.plot(temps, rfit, marker="^", color="#1baf7a", lw=1.4, ls="--", label="ECM fit R₀ (1 Hz model)")
    ax.plot(temps, rtot, marker="v", color="#6250d6", lw=1.4, ls="--", label="ECM fit R₀ + R₁")
    for x, y in zip(temps, r0):
        ax.text(x + 1.2, y * 0.80, f"{y:.0f} mΩ", ha="left", va="top", fontsize=8, color=INK_2)
    ax.set_ylim(6, 400)
    ax.set_yscale("log")
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("resistance (mΩ, log scale)")
    ax.set_title("Internal resistance grows ~7× from 40 °C to −20 °C", loc="left", pad=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_convergence(t_s, soc_true, runs: dict, starts: dict, path=None, title="EKF recovers from a wrong initial SOC within seconds", seconds=90):
    """runs: label -> (soc_hat, soc_std); starts: label -> initial guess."""
    n = int(seconds)
    t = np.concatenate([[0.0], np.asarray(t_s)[:n]])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, 100 * np.concatenate([[soc_true[0]], np.asarray(soc_true)[:n]]), color=INK, lw=1.8, label="True SOC", zorder=5)
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    for (label, (s, sd)), c in zip(runs.items(), colours):
        s = np.concatenate([[starts[label]], np.asarray(s)[:n]])
        sd = np.concatenate([[np.asarray(sd)[0]], np.asarray(sd)[:n]])
        ax.plot(t, 100 * s, color=c, lw=1.3, marker="o", ms=3, label=label)
        ax.fill_between(t, 100 * (s - 2 * sd), 100 * (s + 2 * sd), color=c, alpha=0.10, lw=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("SOC (%)")
    ax.set_ylim(40, 110)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_cc_drift(t_s, soc_true, runs: dict, path=None, title="Coulomb counting: a constant current-sensor offset drifts without bound"):
    h = np.asarray(t_s) / 3600.0
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colours = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e34948"]
    for (label, s), c in zip(runs.items(), colours):
        ax.plot(h, 100 * (np.asarray(s) - np.asarray(soc_true)), color=c, lw=1.4, label=label)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("SOC error (%)")
    ax.set_title(title, loc="left", pad=10)
    ax.legend(fontsize=8.5, title="current-sensor offset", title_fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_cycle_overview(df: pd.DataFrame, title: str, path=None):
    h = df["t_s"].to_numpy() / 3600.0
    fig, axes = plt.subplots(4, 1, figsize=(9, 7.2), sharex=True)
    for ax, col, lab, c in zip(axes, ["V", "I", "T", "SOC"], ["voltage (V)", "current (A)", "cell temperature (°C)", "SOC (%)"], ["#2a78d6", "#eb6834", "#1baf7a", INK]):
        y = df[col].to_numpy() * (100 if col == "SOC" else 1)
        ax.plot(h, y, color=c, lw=0.7 if col in ("V", "I") else 1.4)
        ax.set_ylabel(lab)
    axes[1].axhline(0, color=AXIS, lw=0.8)
    axes[-1].set_xlabel("time (h)")
    axes[0].set_title(title, loc="left", pad=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_overall_bars(summary: pd.Series, path=None, title="Overall RMSE on the test cycles (% SOC)"):
    s = summary.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(s) + 1.4))
    bars = ax.barh(s.index, s.values, color=[METHOD_COLOURS.get(m, MUTED) for m in s.index], height=0.6)
    for b, v in zip(bars, s.values):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, f"{v:.2f} %", va="center", ha="left", fontsize=9, color=INK_2)
    ax.set_xlim(0, s.max() * 1.18)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.grid(False)
    ax.set_title(title, loc="left", pad=10)
    fig.tight_layout()
    return _save(fig, path)
