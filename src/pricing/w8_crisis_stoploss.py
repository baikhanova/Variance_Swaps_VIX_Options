# W8 Person 2 — Crisis Analysis & Stop-Loss
#
# Builds on Person 1's vrp_backtest.py.
# Covers: march 2020 P&L, max drawdown, stop-loss rule (VIX > 40),
# and comparison of key metrics before/after the stop-loss.

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import warnings
warnings.filterwarnings("ignore")

from math import sqrt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.pricing.vrp_backtest import (
    run_vrp_backtest,
    HESTON_PARAMS,
    HESTON_PHYSICAL,
    VEGA_TARGET,
    MONTHS_PER_YEAR,
    N_MONTHS_BACKTEST,
    TRANSACTION_COST,
    TRADING_DAYS_PER_MONTH,
    DELTA,
    compute_vega,
    heston_variance_strike,
)
from src.models.heston import HestonParams, simulate_heston_paths
from src.pricing.variance_mc import compute_realized_variance

COLORS = {
    "blue":  "#1a6faf",
    "red":   "#c0392b",
    "green": "#27ae60",
    "dark":  "#1a1a2e",
    "grey":  "#7f8c8d",
    "gold":  "#f39c12",
}
# crisis analysis helpers

def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    # drawdown at each point = (current - peak so far) / peak
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def crisis_summary(result) -> dict:
    # pull the key numbers we need for the crisis section
    df = result.monthly_pnl
    eq = result.equity_curve

    dd = compute_drawdown_series(eq)
    max_dd = float(dd.min())
    max_dd_date = dd.idxmin()

    worst_idx = df["net_pnl"].idxmin()
    worst_pnl = float(df["net_pnl"].min())

    # march 2020 is the main crisis event — covid crash, VIX hit ~85
    march_2020_mask = (df.index.year == 2020) & (df.index.month == 3)
    march_2020_pnl = float(df.loc[march_2020_mask, "net_pnl"].sum()) if march_2020_mask.any() else None

    # flag any month where the loss was more than 2 standard deviations
    monthly_returns = df["net_pnl"] / VEGA_TARGET
    vol = monthly_returns.std()
    crisis_mask = monthly_returns < -2 * vol
    crisis_months = df.index[crisis_mask].tolist()

    return {
        "max_drawdown": max_dd,
        "max_drawdown_date": max_dd_date,
        "worst_month_pnl": worst_pnl,
        "worst_month_date": worst_idx,
        "march_2020_pnl": march_2020_pnl,
        "crisis_months": crisis_months,
    }