from __future__ import annotations

import os
import sys
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pricing.vrp_backtest import MONTHS_PER_YEAR, run_vrp_backtest


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


def _pick_column(df: pd.DataFrame, candidates: list[str], new_name: str, required: bool = True):
    for c in candidates:
        if c in df.columns:
            df[new_name] = df[c]
            return

    # fallback: search by substring
    for col in df.columns:
        low = str(col).lower()
        for c in candidates:
            if c.lower() in low:
                df[new_name] = df[col]
                return

    if required:
        raise KeyError(f"Could not find column for '{new_name}'. Available columns: {df.columns.tolist()}")


def load_backtest_df() -> pd.DataFrame:
    result = run_vrp_backtest()
    df = result.monthly_pnl.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    # normalize most important columns
    _pick_column(df, ["VIX", "vix", "vix_level", "model_vix", "iv", "iv_vol"], "vix", required=False)
    _pick_column(df, ["K_var", "k_var", "kvar"], "k_var", required=False)
    _pick_column(df, ["RV", "rv", "realized_var"], "rv", required=False)
    _pick_column(df, ["VRP (vol pts)", "VRP", "vrp", "vrp_vol"], "vrp_vol", required=False)
    _pick_column(df, ["Net P&L ($)", "Net P&L", "net_pnl", "pnl"], "net_pnl", required=True)

    # if VIX column does not exist, build a proxy from K_var
    if "vix" not in df.columns:
        if "k_var" in df.columns:
            df["vix"] = 100 * np.sqrt(np.maximum(df["k_var"], 0))
        else:
            raise KeyError(f"No VIX-like column and no k_var to build proxy. Available columns: {df.columns.tolist()}")

    # if VRP missing, build it from K_var and RV
    if "vrp_vol" not in df.columns:
        if "k_var" in df.columns and "rv" in df.columns:
            df["vrp_vol"] = 100 * (df["k_var"] - df["rv"])
        else:
            raise KeyError(f"No VRP-like column and cannot build it. Available columns: {df.columns.tolist()}")

    df["strategy_return"] = df["net_pnl"] / 10000.0

    # placeholder SPX return if none available
    if "spx_return" not in df.columns:
        df["spx_return"] = 0.0

    df["d_vix"] = df["vix"].diff()
    df["vix_lag1"] = df["vix"].shift(1)

    return df


def beta_decomposition(df: pd.DataFrame):
    reg = df[["strategy_return", "spx_return", "d_vix", "vix_lag1"]].dropna().copy()

    y = reg["strategy_return"].values
    X = np.column_stack(
        [
            np.ones(len(reg)),
            reg["spx_return"].values,
            reg["d_vix"].values,
            reg["vix_lag1"].values,
        ]
    )

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta

    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    coef_df = pd.DataFrame(
        {
            "Variable": ["alpha", "beta_SPX", "beta_dVIX", "beta_VIX_lag1"],
            "Coefficient": beta,
        }
    )

    summary = {
        "R2": r2,
        "alpha_monthly": beta[0],
        "alpha_annualized": beta[0] * MONTHS_PER_YEAR,
    }

    return coef_df, summary


def build_regime_tables(df: pd.DataFrame):
    full = compute_metrics(df["strategy_return"])
    high = compute_metrics(df.loc[df["vrp_vol"] > 3, "strategy_return"])
    low = compute_metrics(df.loc[df["vrp_vol"] < 1, "strategy_return"])

    metrics_table = pd.DataFrame(
        {
            "Full Period": full,
            "High VRP (>3)": high,
            "Low VRP (<1)": low,
        }
    )

    regime_df = df.copy()
    regime_df["regime"] = np.where(
        regime_df["vrp_vol"] > 3,
        "High VRP",
        np.where(regime_df["vrp_vol"] < 1, "Low VRP", "Middle"),
    )

    return metrics_table, regime_df


def save_plots(df: pd.DataFrame, metrics_table: pd.DataFrame):
    os.makedirs("outputs/w8", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. equity curve
    equity = 100 * (1 + df["strategy_return"]).cumprod()
    axes[0, 0].plot(df.index, equity, color="steelblue", lw=2)
    axes[0, 0].axhline(100, color="gray", ls="--", lw=1)
    axes[0, 0].set_title("Equity Curve")
    axes[0, 0].grid(alpha=0.3)

    # 2. monthly pnl
    colors = ["seagreen" if x > 0 else "tomato" for x in df["net_pnl"]]
    axes[0, 1].bar(df.index, df["net_pnl"], color=colors, alpha=0.85)
    axes[0, 1].axhline(0, color="gray", lw=1)
    axes[0, 1].set_title("Monthly Net P&L")
    axes[0, 1].grid(alpha=0.3)

    # 3. vrp regimes
    axes[1, 0].plot(df.index, df["vrp_vol"], color="purple", lw=2)
    axes[1, 0].axhline(3, color="green", ls="--", lw=1, label="High VRP")
    axes[1, 0].axhline(1, color="orange", ls="--", lw=1, label="Low VRP")
    axes[1, 0].set_title("VRP Regimes")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. sharpe by regime
    sharpe_vals = [
        metrics_table.loc["Sharpe", "Full Period"],
        metrics_table.loc["Sharpe", "High VRP (>3)"],
        metrics_table.loc["Sharpe", "Low VRP (<1)"],
    ]
    labels = ["Full", "High VRP", "Low VRP"]
    axes[1, 1].bar(labels, sharpe_vals, color=["steelblue", "green", "red"], alpha=0.85)
    axes[1, 1].set_title("Sharpe by Regime")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/w8/w8_metrics_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_metrics_analysis():
    df = load_backtest_df()
    betas, reg_summary = beta_decomposition(df)
    metrics_table, regime_df = build_regime_tables(df)

    os.makedirs("outputs/w8", exist_ok=True)
    betas.to_csv("outputs/w8/beta_decomposition.csv", index=False)
    metrics_table.to_csv("outputs/w8/vrp_metrics.csv")
    regime_df.to_csv("outputs/w8/vrp_regimes.csv")

    save_plots(df, metrics_table)

    return df, betas, reg_summary, metrics_table, regime_df
