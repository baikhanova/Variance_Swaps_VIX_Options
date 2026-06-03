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