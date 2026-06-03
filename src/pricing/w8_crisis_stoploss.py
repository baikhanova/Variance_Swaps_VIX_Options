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
# stop-loss backtest

def run_vrp_backtest_with_stoploss(
    params_rn: HestonParams = HESTON_PARAMS,
    params_p: HestonParams = HESTON_PHYSICAL,
    n_months: int = N_MONTHS_BACKTEST,
    vix_threshold: float = 40.0,
    seed: int = 42,
) -> pd.DataFrame:
    # same as Person 1's backtest but with one extra rule:
    # if VIX at month-start exceeds vix_threshold, skip the trade entirely.
    # VIX is approximated as 100 * sqrt(v_t) from the simulated variance path.
    total_steps = n_months * TRADING_DAYS_PER_MONTH
    total_years = n_months * DELTA

    _, stock_paths, var_paths = simulate_heston_paths(
        params=params_p,
        maturity=total_years,
        steps=total_steps,
        paths=1,
        seed=seed,
    )
    stock = stock_paths[0]
    variance = var_paths[0]

    rows = []
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")

    for m in range(n_months):
        i0 = m * TRADING_DAYS_PER_MONTH
        i1 = i0 + TRADING_DAYS_PER_MONTH

        v_t = float(variance[i0])

        # VIX proxy: 100 * sqrt(v_t) — same convention as the rest of the project
        vix_proxy = 100.0 * sqrt(max(v_t, 0.0))

        params_m = HestonParams(
            s0=float(stock[i0]),
            v0=v_t,
            r=params_rn.r,
            kappa=params_rn.kappa,
            theta=params_rn.theta,
            sigma_v=params_rn.sigma_v,
            rho=params_rn.rho,
        )

        k_var = heston_variance_strike(params_m, DELTA)
        vega_unit = compute_vega(params_m, DELTA, notional=1.0)
        notional = min(VEGA_TARGET / max(vega_unit, 1e-8), 5_000_000.0)

        window = stock[i0:i1 + 1][np.newaxis, :]
        rv = float(compute_realized_variance(window, maturity=DELTA)[0])

        # stop-loss: skip the trade if VIX is too high
        if vix_proxy > vix_threshold:
            net_pnl = 0.0
            trade_taken = False
        else:
            gross_pnl = notional * (k_var - rv)
            tc = TRANSACTION_COST * notional
            net_pnl = gross_pnl - tc
            trade_taken = True

        rows.append({
            "date": dates[m],
            "vix_proxy": vix_proxy,
            "k_var": k_var,
            "rv": rv,
            "iv_vol": 100.0 * sqrt(max(k_var, 0)),
            "rv_vol": 100.0 * sqrt(max(rv, 0)),
            "notional": notional,
            "net_pnl": net_pnl,
            "trade_taken": trade_taken,
        })

    df = pd.DataFrame(rows).set_index("date")
    df["vrp_vol"] = df["iv_vol"] - df["rv_vol"]
    return df


def compare_stoploss_metrics(
    df_base: pd.DataFrame,
    df_sl: pd.DataFrame,
) -> dict:
    # compute sharpe, sortino, max drawdown, win rate for both strategies
    def metrics(df):
        r = df["net_pnl"] / VEGA_TARGET
        sharpe = r.mean() / r.std() * (MONTHS_PER_YEAR ** 0.5) if r.std() > 0 else 0.0
        cum = (1 + r).cumprod()
        max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
        win_rate = (r > 0).mean() * 100
        sortino_denom = r[r < 0].std()
        sortino = r.mean() / sortino_denom * (MONTHS_PER_YEAR ** 0.5) if sortino_denom > 0 else 0.0
        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_pnl": df["net_pnl"].sum(),
            "months_traded": int(df["trade_taken"].sum()) if "trade_taken" in df.columns else len(df),
        }

    return {
        "base": metrics(df_base),
        "stop_loss": metrics(df_sl),
    }


# plotting

def plot_crisis_analysis(result, crisis_stats: dict, save_path: str):
    # three panels: equity curve (with march 2020 shaded), monthly P&L bars, drawdown
    df = result.monthly_pnl
    eq = result.equity_curve
    dd = compute_drawdown_series(eq)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.patch.set_facecolor(COLORS["dark"])

    # --- top: equity curve ---
    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    ax1.plot(eq.index, eq.values, color=COLORS["blue"], lw=2.5, label="Strategy NAV")
    ax1.fill_between(eq.index, eq.values, 100, alpha=0.15, color=COLORS["blue"])
    ax1.axhline(100, color=COLORS["grey"], ls="--", lw=0.7)

    # shade March 2020
    march_mask = (eq.index.year == 2020) & (eq.index.month == 3)
    if march_mask.any():
        ax1.axvspan(
            eq.index[march_mask][0] - pd.Timedelta(days=15),
            eq.index[march_mask][0] + pd.Timedelta(days=15),
            color=COLORS["red"], alpha=0.25, label="Mar 2020 (Covid crash)"
        )

    ax1.set_title("Equity Curve — Short Variance Swap (2019–2024)", color="white", fontsize=11)
    ax1.set_ylabel("NAV (base=100)", color=COLORS["grey"])
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333")
    ax1.legend(facecolor="#111", labelcolor="white", fontsize=9)
    ax1.grid(alpha=0.15, color="#444")

    # --- middle: monthly P&L bars ---
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    bar_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in df["net_pnl"]]
    ax2.bar(df.index, df["net_pnl"], color=bar_colors, alpha=0.85, width=20)
    ax2.axhline(0, color=COLORS["grey"], lw=0.8)

    # mark the worst month
    worst_date = crisis_stats["worst_month_date"]
    worst_val = crisis_stats["worst_month_pnl"]
    ax2.annotate(
        f"Worst month\n${worst_val:,.0f}",
        xy=(worst_date, worst_val),
        xytext=(worst_date, worst_val * 0.6),
        color="white", fontsize=8,
        arrowprops=dict(arrowstyle="->", color=COLORS["gold"]),
        ha="center",
    )

    ax2.set_title("Monthly Net P&L", color="white", fontsize=11)
    ax2.set_ylabel("P&L ($)", color=COLORS["grey"])
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(alpha=0.15, color="#444")

    # --- bottom: drawdown ---
    ax3 = axes[2]
    ax3.set_facecolor("#0d1117")
    ax3.fill_between(dd.index, dd.values * 100, 0, color=COLORS["red"], alpha=0.6)
    ax3.plot(dd.index, dd.values * 100, color=COLORS["red"], lw=1.5)
    ax3.axhline(0, color=COLORS["grey"], lw=0.8)

    max_dd_date = crisis_stats["max_drawdown_date"]
    max_dd_val = crisis_stats["max_drawdown"] * 100
    ax3.annotate(
        f"Max DD\n{max_dd_val:.1f}%",
        xy=(max_dd_date, max_dd_val),
        xytext=(max_dd_date, max_dd_val - 5),
        color="white", fontsize=8,
        arrowprops=dict(arrowstyle="->", color=COLORS["gold"]),
        ha="center",
    )

    ax3.set_title("Drawdown from Peak", color="white", fontsize=11)
    ax3.set_ylabel("Drawdown (%)", color=COLORS["grey"])
    ax3.tick_params(colors="white")
    ax3.spines[:].set_color("#333")
    ax3.grid(alpha=0.15, color="#444")

    fig.suptitle(
        "Person 2 — Crisis Analysis: Short Variance Swap",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["dark"])
    plt.close()
    print(f"Saved: {save_path}")


def plot_stoploss_comparison(
    df_base: pd.DataFrame,
    df_sl: pd.DataFrame,
    metrics: dict,
    vix_threshold: float,
    save_path: str,
):
    # left panel: equity curves side by side; right panel: metrics bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(COLORS["dark"])

    # equity curves
    def equity_curve(df):
        r = df["net_pnl"] / VEGA_TARGET
        cum = (1 + r).cumprod()
        return cum / cum.iloc[0] * 100

    eq_base = equity_curve(df_base)
    eq_sl = equity_curve(df_sl)

    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    ax1.plot(eq_base.index, eq_base.values, color=COLORS["blue"], lw=2, label="No stop-loss")
    ax1.plot(eq_sl.index, eq_sl.values, color=COLORS["green"], lw=2, label=f"Stop-loss VIX>{vix_threshold:.0f}")
    ax1.axhline(100, color=COLORS["grey"], ls="--", lw=0.7)

    # shade months where stop-loss fired
    fired = df_sl[~df_sl["trade_taken"]].index
    for d in fired:
        ax1.axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15),
                    color=COLORS["gold"], alpha=0.15)

    ax1.set_title(f"Equity Curve: Base vs Stop-Loss (VIX > {vix_threshold:.0f})",
                  color="white", fontsize=11)
    ax1.set_ylabel("NAV (base=100)", color=COLORS["grey"])
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333")
    ax1.legend(facecolor="#111", labelcolor="white", fontsize=9)
    ax1.grid(alpha=0.15, color="#444")

    # metrics bar chart
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")

    labels = ["Sharpe", "Sortino", "Max DD (%)"]
    base_vals = [
        metrics["base"]["sharpe"],
        metrics["base"]["sortino"],
        abs(metrics["base"]["max_drawdown"]) * 100,
    ]
    sl_vals = [
        metrics["stop_loss"]["sharpe"],
        metrics["stop_loss"]["sortino"],
        abs(metrics["stop_loss"]["max_drawdown"]) * 100,
    ]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, base_vals, width, color=COLORS["blue"], alpha=0.85, label="No stop-loss")
    bars2 = ax2.bar(x + width / 2, sl_vals, width, color=COLORS["green"], alpha=0.85, label="Stop-loss")

    for bar, v in zip(list(bars1) + list(bars2), base_vals + sl_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}",
                 ha="center", color="white", fontsize=9)

    ax2.set_title("Risk Metrics Comparison", color="white", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="white")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.legend(facecolor="#111", labelcolor="white", fontsize=9)
    ax2.grid(alpha=0.15, color="#444", axis="y")

    fig.suptitle(
        f"Person 2 — Stop-Loss Analysis (threshold VIX = {vix_threshold:.0f})",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["dark"])
    plt.close()
    print(f"Saved: {save_path}")


def main(output_dir: str = "outputs/w8", vix_threshold: float = 40.0):
    os.makedirs(output_dir, exist_ok=True)

    print("W8 Person 2 — Crisis Analysis & Stop-Loss")

    result_base = run_vrp_backtest()
    df_base = result_base.monthly_pnl

    crisis = crisis_summary(result_base)

    print(f"Max Drawdown:    {crisis['max_drawdown']*100:.1f}%  ({crisis['max_drawdown_date'].strftime('%Y-%m')})")
    print(f"Worst month:     ${crisis['worst_month_pnl']:,.0f}  ({crisis['worst_month_date'].strftime('%Y-%m')})")
    if crisis["march_2020_pnl"] is not None:
        print(f"March 2020 P&L:  ${crisis['march_2020_pnl']:,.0f}")
    else:
        print("March 2020: not in simulation window")
    print(f"Crisis months (loss > 2σ): {len(crisis['crisis_months'])}")

    plot_crisis_analysis(
        result_base, crisis,
        save_path=os.path.join(output_dir, "w8_crisis_analysis.png"),
    )

    print(f"\nRunning stop-loss backtest (VIX threshold = {vix_threshold})...")
    df_sl = run_vrp_backtest_with_stoploss(vix_threshold=vix_threshold)

    trades_taken = df_sl["trade_taken"].sum()
    trades_skipped = (~df_sl["trade_taken"]).sum()
    print(f"Trades taken:   {trades_taken} / {len(df_sl)}")
    print(f"Trades skipped: {trades_skipped}")

    metrics = compare_stoploss_metrics(df_base, df_sl)

    print(f"\n{'Metric':<20} {'Base':>10} {'Stop-Loss':>10}")
    print(f"{'Sharpe':<20} {metrics['base']['sharpe']:>10.2f} {metrics['stop_loss']['sharpe']:>10.2f}")
    print(f"{'Sortino':<20} {metrics['base']['sortino']:>10.2f} {metrics['stop_loss']['sortino']:>10.2f}")
    print(f"{'Max Drawdown':<20} {metrics['base']['max_drawdown']*100:>9.1f}% {metrics['stop_loss']['max_drawdown']*100:>9.1f}%")
    print(f"{'Win Rate':<20} {metrics['base']['win_rate']:>9.0f}% {metrics['stop_loss']['win_rate']:>9.0f}%")
    print(f"{'Total P&L':<20} ${metrics['base']['total_pnl']:>9,.0f} ${metrics['stop_loss']['total_pnl']:>9,.0f}")

    plot_stoploss_comparison(
        df_base, df_sl, metrics, vix_threshold,
        save_path=os.path.join(output_dir, "w8_stoploss_comparison.png"),
    )

    summary = pd.DataFrame({
        "Metric": ["Sharpe", "Sortino", "Max Drawdown (%)", "Win Rate (%)", "Total P&L ($)", "Months Traded"],
        "Base": [
            round(metrics["base"]["sharpe"], 3),
            round(metrics["base"]["sortino"], 3),
            round(metrics["base"]["max_drawdown"] * 100, 2),
            round(metrics["base"]["win_rate"], 1),
            round(metrics["base"]["total_pnl"], 0),
            metrics["base"]["months_traded"],
        ],
        "Stop-Loss": [
            round(metrics["stop_loss"]["sharpe"], 3),
            round(metrics["stop_loss"]["sortino"], 3),
            round(metrics["stop_loss"]["max_drawdown"] * 100, 2),
            round(metrics["stop_loss"]["win_rate"], 1),
            round(metrics["stop_loss"]["total_pnl"], 0),
            metrics["stop_loss"]["months_traded"],
        ],
    })
    csv_path = os.path.join(output_dir, "w8_person2_metrics.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
