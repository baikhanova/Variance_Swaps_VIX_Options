from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from math import sqrt

from src.pricing.vrp_backtest import run_vrp_backtest, MONTHS_PER_YEAR


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax()
    return float(drawdown.min())


def compute_metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()

    ann_return = (1 + returns).prod() ** (MONTHS_PER_YEAR / len(returns)) - 1 if len(returns) > 0 else np.nan
    ann_vol = returns.std(ddof=1) * sqrt(MONTHS_PER_YEAR) if len(returns) > 1 else np.nan
    sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan

    downside = returns[returns < 0]
    downside_vol = downside.std(ddof=1) * sqrt(MONTHS_PER_YEAR) if len(downside) > 1 else np.nan
    sortino = ann_return / downside_vol if pd.notna(downside_vol) and downside_vol > 0 else np.nan

    mdd = max_drawdown(returns) if len(returns) > 0 else np.nan
    calmar = ann_return / abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan

    var_5 = np.percentile(returns, 5) if len(returns) > 0 else np.nan
    cvar_5 = returns[returns <= var_5].mean() if len(returns) > 0 else np.nan

    skew = returns.skew() if len(returns) > 1 else np.nan
    kurt = returns.kurt() if len(returns) > 1 else np.nan

    return {
        "Ann.Return": ann_return,
        "Ann.Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": mdd,
        "Calmar": calmar,
        "VaR(5%)": var_5,
        "CVaR(5%)": cvar_5,
        "Skewness": skew,
        "Kurtosis": kurt,
        "N": len(returns),
    }


def load_backtest_df():
    result = run_vrp_backtest()
    df = result.monthly_pnl.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    rename_map = {
        "VIX": "vix",
        "K_var": "k_var",
        "RV": "rv",
        "VRP (vol pts)": "vrp_vol",
        "Net P&L ($)": "net_pnl",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "vrp_vol" not in df.columns and "VRP" in df.columns:
        df["vrp_vol"] = df["VRP"]

    if "net_pnl" not in df.columns and "Net P&L" in df.columns:
        df["net_pnl"] = df["Net P&L"]

    df["strategy_return"] = df["net_pnl"] / 10000.0

    # if later real SPX monthly returns are available, replace this
    if "spx_return" not in df.columns:
        df["spx_return"] = 0.0

    df["d_vix"] = df["vix"].diff()
    df["vix_lag1"] = df["vix"].shift(1)

    return df


def beta_decomposition(df: pd.DataFrame):
    reg = df[["strategy_return", "spx_return", "d_vix", "vix_lag1"]].dropna().copy()

    y = reg["strategy_return"].values
    X = np.column_stack([
        np.ones(len(reg)),
        reg["spx_return"].values,
        reg["d_vix"].values,
        reg["vix_lag1"].values,
    ])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta

    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    coef_df = pd.DataFrame({
        "Variable": ["alpha", "beta_SPX", "beta_dVIX", "beta_VIX_lag1"],
        "Coefficient": beta
    })

    summary = {
        "R2": r2,
        "alpha_monthly": beta[0],
        "alpha_annualized": beta[0] * MONTHS_PER_YEAR
    }

    return coef_df, summary


def build_regime_tables(df: pd.DataFrame):
    full = compute_metrics(df["strategy_return"])
    high = compute_metrics(df.loc[df["vrp_vol"] > 3, "strategy_return"])
    low = compute_metrics(df.loc[df["vrp_vol"] < 1, "strategy_return"])

    metrics_table = pd.DataFrame({
        "Full Period": full,
        "High VRP (>3)": high,
        "Low VRP (<1)": low,
    })

    regime_df = df.copy()
    regime_df["regime"] = np.where(
        regime_df["vrp_vol"] > 3, "High VRP",
        np.where(regime_df["vrp_vol"] < 1, "Low VRP", "Middle")
    )

    return metrics_table, regime_df


def run_metrics_analysis():
    df = load_backtest_df()
    betas, reg_summary = beta_decomposition(df)
    metrics_table, regime_df = build_regime_tables(df)

    os.makedirs("outputs/w8", exist_ok=True)
    betas.to_csv("outputs/w8/beta_decomposition.csv", index=False)
    metrics_table.to_csv("outputs/w8/vrp_metrics.csv")
    regime_df.to_csv("outputs/w8/vrp_regimes.csv")

    return df, betas, reg_summary, metrics_table, regime_df
