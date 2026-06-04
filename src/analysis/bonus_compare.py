from __future__ import annotations

import os
import sys
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pricing.vrp_backtest import MONTHS_PER_YEAR, run_vrp_backtest


def _pick_column(df: pd.DataFrame, candidates: list[str], new_name: str, required: bool = True):
    for c in candidates:
        if c in df.columns:
            df[new_name] = df[c]
            return

    for col in df.columns:
        low = str(col).lower()
        for c in candidates:
            if c.lower() in low:
                df[new_name] = df[col]
                return

    if required:
        raise KeyError(f"Could not find column for '{new_name}'. Available columns: {df.columns.tolist()}")


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax()
    return float(drawdown.min())


def compute_metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()

    if len(returns) == 0:
        return {
            "Ann.Return": np.nan,
            "Ann.Vol": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "MaxDrawdown": np.nan,
            "Calmar": np.nan,
            "VaR(5%)": np.nan,
            "CVaR(5%)": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
            "N": 0,
        }

    ann_return = (1 + returns).prod() ** (MONTHS_PER_YEAR / len(returns)) - 1
    ann_vol = returns.std(ddof=1) * sqrt(MONTHS_PER_YEAR) if len(returns) > 1 else np.nan
    sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan

    downside = returns[returns < 0]
    downside_vol = downside.std(ddof=1) * sqrt(MONTHS_PER_YEAR) if len(downside) > 1 else np.nan
    sortino = ann_return / downside_vol if pd.notna(downside_vol) and downside_vol > 0 else np.nan

    mdd = max_drawdown(returns)
    calmar = ann_return / abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan

    var_5 = np.percentile(returns, 5)
    cvar_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else np.nan

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


def load_df() -> pd.DataFrame:
    result = run_vrp_backtest()
    df = result.monthly_pnl.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    _pick_column(df, ["Net P&L ($)", "Net P&L", "net_pnl", "pnl"], "net_pnl", required=True)
    _pick_column(df, ["K_var", "k_var", "kvar"], "k_var", required=False)
    _pick_column(df, ["RV", "rv", "realized_var"], "rv", required=False)
    _pick_column(df, ["VIX", "vix", "vix_level", "model_vix", "iv", "iv_vol"], "vix", required=False)

    if "k_var" not in df.columns:
        raise KeyError(f"No k_var column found. Available columns: {df.columns.tolist()}")
    if "rv" not in df.columns:
        raise KeyError(f"No rv column found. Available columns: {df.columns.tolist()}")

    if "vix" not in df.columns:
        df["vix"] = 100 * np.sqrt(np.maximum(df["k_var"], 0))

    df["iv_vol"] = 100 * np.sqrt(np.maximum(df["k_var"], 0))
    df["rv_vol"] = 100 * np.sqrt(np.maximum(df["rv"], 0))

    return df


def build_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index.copy())

    # 1) short variance swap: already computed in backtest
    out["Variance Swap"] = df["net_pnl"] / 10000.0

    # 2) short VIX futures proxy:
    # approximate 1M short VIX future by shorting next-month VIX move
    vix_next = df["vix"].shift(-1)
    out["Short VIX Future"] = -(vix_next - df["vix"]) / df["vix"].clip(lower=1e-8)

    # 3) short straddle proxy:
    # simple delta-hedged proxy: implied vol minus realized vol, normalized by IV
    out["Short Straddle"] = (df["iv_vol"] - df["rv_vol"]) / df["iv_vol"].replace(0, np.nan)

    out = out.dropna()
    return out


def compare_metrics(ret_df: pd.DataFrame) -> pd.DataFrame:
    table = {}
    for col in ret_df.columns:
        table[col] = compute_metrics(ret_df[col])
    return pd.DataFrame(table)


def save_plots(ret_df: pd.DataFrame, metrics: pd.DataFrame):
    os.makedirs("outputs/w8", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # equity curves
    eq = 100 * (1 + ret_df).cumprod()
    for col in eq.columns:
        axes[0, 0].plot(eq.index, eq[col], lw=2, label=col)
    axes[0, 0].axhline(100, color="gray", ls="--", lw=1)
    axes[0, 0].set_title("Equity Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # annual returns
    ann_returns = metrics.loc["Ann.Return"]
    axes[0, 1].bar(ann_returns.index, ann_returns.values, color=["steelblue", "purple", "orange"], alpha=0.85)
    axes[0, 1].set_title("Annual Return")
    axes[0, 1].grid(alpha=0.3)

    # sharpe
    sharpe_vals = metrics.loc["Sharpe"]
    axes[1, 0].bar(sharpe_vals.index, sharpe_vals.values, color=["steelblue", "purple", "orange"], alpha=0.85)
    axes[1, 0].set_title("Sharpe Ratio")
    axes[1, 0].grid(alpha=0.3)

    # max drawdown
    mdd_vals = metrics.loc["MaxDrawdown"]
    axes[1, 1].bar(mdd_vals.index, mdd_vals.values, color=["steelblue", "purple", "orange"], alpha=0.85)
    axes[1, 1].set_title("Max Drawdown")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/w8/bonus_compare.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_bonus_compare():
    df = load_df()
    ret_df = build_strategy_returns(df)
    metrics = compare_metrics(ret_df)

    os.makedirs("outputs/w8", exist_ok=True)
    ret_df.to_csv("outputs/w8/bonus_returns.csv")
    metrics.to_csv("outputs/w8/bonus_metrics.csv")

    save_plots(ret_df, metrics)

    return df, ret_df, metrics
