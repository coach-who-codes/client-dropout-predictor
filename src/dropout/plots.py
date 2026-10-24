"""Report figures with one consistent, colour-blind-safe style."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical order: blue, orange, aqua
SEQ_LIGHT, SEQ_DARK = "#86b6ef", "#1c5cab"

PRETTY = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _fig(width: float, height: float, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def model_comparison(y: np.ndarray, oof: Dict[str, np.ndarray], path: Path) -> None:
    fig, (ax1, ax2) = _fig(10, 4.2, 2)
    for color, (name, proba) in zip(SERIES, oof.items()):
        fpr, tpr, _ = roc_curve(y, proba)
        prec, rec, _ = precision_recall_curve(y, proba)
        ax1.plot(fpr, tpr, color=color, lw=2, label=PRETTY.get(name, name))
        ax2.plot(rec, prec, color=color, lw=2, label=PRETTY.get(name, name))
    ax1.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--")
    ax1.text(0.62, 0.55, "random guess", color=INK_2, fontsize=8, rotation=33)
    ax2.axhline(y.mean(), color=INK_2, lw=1, ls="--")
    ax2.text(0.02, y.mean() + 0.02, f"base rate {y.mean():.0%}", color=INK_2, fontsize=8)
    _style(ax1, "ROC curve (5-fold out-of-fold)")
    _style(ax2, "Precision-recall curve")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate (recall)")
    ax2.set(xlabel="Recall: share of dropouts caught", ylabel="Precision", ylim=(0, 1.02))
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def savings_curve(thresholds: np.ndarray, savings_per_100: np.ndarray, chosen: float, path: Path) -> None:
    fig, ax = _fig(8, 3.6)
    ax.plot(thresholds, savings_per_100, color=SERIES[0], lw=2)
    best = savings_per_100[np.argmin(np.abs(thresholds - chosen))]
    ax.scatter([chosen], [best], s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(
        f"chosen threshold {chosen:.2f}\n= Rs {best:,.0f} saved per 100 clients",
        (chosen, best),
        xytext=(30, -45),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8},
    )
    ax.axhline(0, color=INK_2, lw=1)
    _style(ax, "Net rupees saved per 100 clients vs. risk threshold (cross-validation)")
    ax.set(xlabel="Contact every client with predicted risk >= threshold", ylabel="Rs saved per 100 clients")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def feature_importance(mean_abs_shap: pd.Series, labels: Dict[str, str], path: Path, top: int = 12) -> None:
    data = mean_abs_shap.sort_values(ascending=True).tail(top)
    fig, ax = _fig(8, 4.6)
    names = [labels.get(f, f) for f in data.index]
    ax.barh(names, data.values, color=SERIES[0], height=0.6)
    for y_pos, value in enumerate(data.values):
        ax.text(value, y_pos, f" {value:.2f}", va="center", fontsize=8, color=INK_2)
    _style(ax, "What drives dropout risk (mean |SHAP|, test set)")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average impact on predicted risk (log-odds)")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def rate_by_bucket(df: pd.DataFrame, column: str, bins, labels, target: str, title: str, xlabel: str, path: Path):
    buckets = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    rates = df.groupby(buckets, observed=False)[target].mean()
    counts = df.groupby(buckets, observed=False)[target].size()
    fig, ax = _fig(7, 3.6)
    ax.bar(rates.index.astype(str), rates.values, color=SERIES[0], width=0.6)
    for x, (rate, n) in enumerate(zip(rates.values, counts.values)):
        ax.text(x, rate + 0.01, f"{rate:.0%}\n(n={n})", ha="center", fontsize=8, color=INK_2)
    _style(ax, title)
    ax.grid(axis="x", visible=False)
    ax.set(xlabel=xlabel, ylabel="Dropout rate", ylim=(0, max(rates.values) * 1.3))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
