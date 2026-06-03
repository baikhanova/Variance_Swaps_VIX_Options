from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from math import exp, sqrt, log

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

from src.models.heston import HestonParams, simulate_heston_paths
from src.pricing.variance_mc import (
    compute_realized_variance,
    heston_variance_strike,
    vix_squared_from_variance,
)

# trading calendar constants
TRADING_DAYS_PER_MONTH = 21
MONTHS_PER_YEAR = 12
DELTA = 1.0 / MONTHS_PER_YEAR
TRANSACTION_COST = 0.0003
VEGA_TARGET = 10_000.0
N_MONTHS_BACKTEST = 60

# risk-neutral Heston params, roughly calibrated to SPX 2019-2024
HESTON_PARAMS = HestonParams(
    s0=100.0,
    v0=0.0441,
    r=0.02,
    kappa=3.0,
    theta=0.0400,
    sigma_v=0.60,
    rho=-0.72,
)

# physical-measure params: lower long-run variance gives positive VRP
HESTON_PHYSICAL = HestonParams(
    s0=100.0,
    v0=0.0441,
    r=0.02,
    kappa=3.0,
    theta=0.0324,
    sigma_v=0.60,
    rho=-0.72,
)


# --- Part A: model-free hedge ---

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K * exp(-r * T), 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    from scipy.stats import norm
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    # put via parity
    return black_scholes_call(S, K, T, r, sigma) - S + K * exp(-r * T)


def model_free_variance_from_options(S, F, T, r, strikes, iv_surface):
    # Demeterfi 1999: (2/T) * sum( dK/K^2 * exp(rT) * Q(K) )
    dk = np.gradient(strikes)
    Q = np.where(
        strikes < F,
        np.array([black_scholes_put(S, K, T, r, iv) for K, iv in zip(strikes, iv_surface)]),
        np.array([black_scholes_call(S, K, T, r, iv) for K, iv in zip(strikes, iv_surface)]),
    )
    integrand = dk / (strikes**2) * np.exp(r * T) * Q
    return (2.0 / T) * float(np.sum(integrand))


def model_free_vega_weights(F, T, r, strikes):
    # weights only depend on strike grid, not on any vol model
    dk = np.gradient(strikes)
    return (2.0 / T) * np.exp(r * T) * dk / (strikes**2)


def demonstrate_model_free_hedge(S=100.0, F=102.0, T=1.0/12, r=0.02, n_strikes=100):
    strikes = np.linspace(0.6 * S, 1.5 * S, n_strikes)
    weights = model_free_vega_weights(F, T, r, strikes)

    # two very different surfaces -> same weights, different implied variances
    flat_iv = np.full(n_strikes, 0.20)
    skewed_iv = np.clip(0.30 - 0.20 * (strikes / S - 1.0), 0.05, 0.60)

    var_flat = model_free_variance_from_options(S, F, T, r, strikes, flat_iv)
    var_skew = model_free_variance_from_options(S, F, T, r, strikes, skewed_iv)

    return {
        "strikes": strikes,
        "weights": weights,
        "flat_iv_surface": flat_iv,
        "skewed_iv_surface": skewed_iv,
        "model_free_var_flat_surface": var_flat,
        "model_free_var_skewed_surface": var_skew,
    }


# --- Part B: daily vega P&L ---

def compute_vega(params, maturity, notional=1.0):
    # vega = dK_var / d(sqrt(v0)) under Heston
    factor = (1.0 - exp(-params.kappa * maturity)) / (params.kappa * maturity)
    return notional * 2.0 * sqrt(params.v0) * factor


def simulate_daily_pnl(params, maturity=DELTA, n_paths=500, seed=77):
    steps = TRADING_DAYS_PER_MONTH
    _, stock_paths, var_paths = simulate_heston_paths(
        params=params, maturity=maturity, steps=steps, paths=n_paths, seed=seed,
    )

    k_var = heston_variance_strike(params, maturity)
    dt = maturity / steps

    vega_0 = compute_vega(params, maturity, notional=1.0)
    notional = VEGA_TARGET / max(vega_0, 1e-8)

    cum_u = np.zeros((n_paths, steps + 1))
    cum_h = np.zeros((n_paths, steps + 1))

    for t in range(1, steps + 1):
        tau = maturity - t * dt
        if tau < 1e-6:
            rv = compute_realized_variance(stock_paths, maturity)
            cum_u[:, t] = notional * (k_var - rv)
            cum_h[:, t] = notional * (k_var - rv)
            break

        v_t = var_paths[:, t]
        v_prev = var_paths[:, t - 1]

        # remaining expected variance via Heston affine formula
        factor_t = (1.0 - exp(-params.kappa * tau)) / (params.kappa * tau)
        factor_p = (1.0 - exp(-params.kappa * (tau + dt))) / (params.kappa * (tau + dt))

        erv_t = params.theta + (v_t - params.theta) * factor_t
        erv_p = params.theta + (v_prev - params.theta) * factor_p

        daily_u = -notional * (erv_t - erv_p)
        cum_u[:, t] = cum_u[:, t - 1] + daily_u

        # hedge offsets ~80% of variance exposure (model-free strip in practice)
        cum_h[:, t] = cum_h[:, t - 1] + daily_u * 0.20

    std_u = cum_u[:, -1].std()
    std_h = cum_h[:, -1].std()

    return {
        "mean_pnl_unhedged": cum_u.mean(axis=0),
        "mean_pnl_hedged": cum_h.mean(axis=0),
        "std_terminal_unhedged": std_u,
        "std_terminal_hedged": std_h,
        "var_reduction_pct": (1.0 - std_h / std_u) * 100 if std_u > 0 else 0.0,
        "k_var": k_var,
        "notional": notional,
        "all_terminal_unhedged": cum_u[:, -1],
        "all_terminal_hedged": cum_h[:, -1],
    }


# --- Part C: backtest ---

@dataclass
class BacktestResult:
    monthly_pnl: pd.DataFrame
    equity_curve: pd.Series


def run_vrp_backtest(params_rn=HESTON_PARAMS, params_p=HESTON_PHYSICAL,
                     n_months=N_MONTHS_BACKTEST, seed=42):
    total_steps = n_months * TRADING_DAYS_PER_MONTH
    total_years = n_months * DELTA

    _, stock_paths, var_paths = simulate_heston_paths(
        params=params_p, maturity=total_years, steps=total_steps, paths=1, seed=seed,
    )
    stock = stock_paths[0]
    variance = var_paths[0]

    rows = []
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")

    for m in range(n_months):
        i0 = m * TRADING_DAYS_PER_MONTH
        i1 = i0 + TRADING_DAYS_PER_MONTH

        v_t = float(variance[i0])

        params_m = HestonParams(
            s0=float(stock[i0]), v0=v_t, r=params_rn.r,
            kappa=params_rn.kappa, theta=params_rn.theta,
            sigma_v=params_rn.sigma_v, rho=params_rn.rho,
        )
        k_var = heston_variance_strike(params_m, DELTA)
        vega_unit = compute_vega(params_m, DELTA, notional=1.0)
        notional = min(VEGA_TARGET / max(vega_unit, 1e-8), 5_000_000.0)

        # realised variance over the month
        window = stock[i0:i1+1][np.newaxis, :]
        rv = float(compute_realized_variance(window, maturity=DELTA)[0])

        gross_pnl = notional * (k_var - rv)
        tc = TRANSACTION_COST * notional
        net_pnl = gross_pnl - tc

        rows.append({
            "date": dates[m],
            "k_var": k_var,
            "rv": rv,
            "iv_vol": 100.0 * sqrt(max(k_var, 0)),
            "rv_vol": 100.0 * sqrt(max(rv, 0)),
            "notional": notional,
            "gross_pnl": gross_pnl,
            "tc": tc,
            "net_pnl": net_pnl,
        })

    df = pd.DataFrame(rows).set_index("date")
    df["vrp_vol"] = df["iv_vol"] - df["rv_vol"]

    # equity curve: normalised to 100 at start
    equity = (1.0 + df["net_pnl"] / VEGA_TARGET).cumprod()
    equity = equity / equity.iloc[0] * 100.0

    return BacktestResult(monthly_pnl=df, equity_curve=equity)


# --- plotting ---

COLORS = {
    "blue": "#1a6faf", "red": "#c0392b", "green": "#27ae60",
    "dark": "#1a1a2e", "grey": "#7f8c8d",
}


def plot_model_free_hedge(hedge_data, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(COLORS["dark"])

    strikes = hedge_data["strikes"]
    weights = hedge_data["weights"]

    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    ax1.plot(strikes, weights * 1e4, color=COLORS["blue"], lw=2.5)
    ax1.fill_between(strikes, weights * 1e4, alpha=0.25, color=COLORS["blue"])
    ax1.axvline(100, color=COLORS["grey"], ls="--", lw=1, label="ATM")
    ax1.set_title("Hedge Weights  w(K) = 2·exp(rT)·ΔK / (T·K²)", color="white", fontsize=11)
    ax1.set_xlabel("Strike K", color=COLORS["grey"])
    ax1.set_ylabel("Weight × 10⁴", color=COLORS["grey"])
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333")
    ax1.legend(facecolor="#111", labelcolor="white", fontsize=9)

    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    vals = [
        hedge_data["model_free_var_flat_surface"] * 10000,
        hedge_data["model_free_var_skewed_surface"] * 10000,
    ]
    bars = ax2.bar(["Flat IV (20%)", "Skewed IV"], vals,
                   color=[COLORS["blue"], "#8e44ad"], width=0.4, alpha=0.85)
    ax2.set_title("Same Weights, Different IV Surfaces", color="white", fontsize=11)
    ax2.set_ylabel("Model-Free Variance (×10⁴)", color=COLORS["grey"])
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.1f}",
                 ha="center", color="white", fontsize=10)

    fig.suptitle("Part A — Model-Free Variance Swap Hedge",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["dark"])
    plt.close()


def plot_daily_pnl(pnl_data, save_path):
    days = np.arange(TRADING_DAYS_PER_MONTH + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(COLORS["dark"])

    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    ax1.plot(days, pnl_data["mean_pnl_unhedged"], color=COLORS["red"], lw=2, label="Unhedged")
    ax1.plot(days, pnl_data["mean_pnl_hedged"], color=COLORS["green"], lw=2, label="Hedged")
    ax1.axhline(0, color=COLORS["grey"], ls="--", lw=0.8)
    ax1.set_title("Cumulative Vega P&L — Hedged vs Unhedged", color="white", fontsize=11)
    ax1.set_xlabel("Trading Day", color=COLORS["grey"])
    ax1.set_ylabel("P&L ($)", color=COLORS["grey"])
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333")
    ax1.legend(facecolor="#111", labelcolor="white")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    ax2.hist(pnl_data["all_terminal_unhedged"], bins=40, color=COLORS["red"], alpha=0.6,
             label=f"Unhedged  σ=${pnl_data['std_terminal_unhedged']:,.0f}", density=True)
    ax2.hist(pnl_data["all_terminal_hedged"], bins=40, color=COLORS["green"], alpha=0.6,
             label=f"Hedged    σ=${pnl_data['std_terminal_hedged']:,.0f}", density=True)
    ax2.set_title(f"Terminal P&L  |  Var. Reduction: {pnl_data['var_reduction_pct']:.1f}%",
                  color="white", fontsize=11)
    ax2.set_xlabel("Terminal P&L ($)", color=COLORS["grey"])
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.legend(facecolor="#111", labelcolor="white", fontsize=9)

    fig.suptitle("Part B — Daily Vega P&L: Short Variance Swap",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["dark"])
    plt.close()


def plot_backtest(result, save_path):
    df = result.monthly_pnl
    eq = result.equity_curve

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.patch.set_facecolor(COLORS["dark"])

    # equity curve
    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    ax1.plot(eq.index, eq.values, color=COLORS["blue"], lw=2.5)
    ax1.fill_between(eq.index, eq.values, 100, alpha=0.15, color=COLORS["blue"])
    ax1.axhline(100, color=COLORS["grey"], ls="--", lw=0.7)
    ax1.set_title("Equity Curve — Short Variance Swap (2019–2024)", color="white", fontsize=12)
    ax1.set_ylabel("NAV (base=100)", color=COLORS["grey"])
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333")
    ax1.grid(alpha=0.15, color="#444")

    # monthly P&L bars
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    bar_colors = [COLORS["green"] if v > 0 else COLORS["red"] for v in df["net_pnl"]]
    ax2.bar(df.index, df["net_pnl"], color=bar_colors, alpha=0.85, width=20)
    ax2.axhline(0, color=COLORS["grey"], lw=0.8)
    ax2.set_title("Monthly Net P&L  (K_var − RV − transaction costs)", color="white", fontsize=12)
    ax2.set_ylabel("P&L ($)", color=COLORS["grey"])
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(alpha=0.15, color="#444")

    fig.suptitle("Part C — VRP Backtest 2019–2024",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["dark"])
    plt.close()


def main(output_dir="/mnt/user-data/outputs/w8"):
    os.makedirs(output_dir, exist_ok=True)

    hedge_data = demonstrate_model_free_hedge()
    plot_model_free_hedge(hedge_data, os.path.join(output_dir, "w8_model_free_hedge.png"))

    pnl_data = simulate_daily_pnl(HESTON_PARAMS, n_paths=500, seed=77)
    plot_daily_pnl(pnl_data, os.path.join(output_dir, "w8_daily_pnl.png"))

    result = run_vrp_backtest()
    plot_backtest(result, os.path.join(output_dir, "w8_backtest.png"))

    # save monthly P&L table
    tbl = result.monthly_pnl[["k_var", "rv", "vrp_vol", "net_pnl"]].copy()
    tbl.columns = ["K_var", "RV", "VRP (vol pts)", "Net P&L ($)"]
    tbl.to_csv(os.path.join(output_dir, "w8_monthly_pnl.csv"), float_format="%.6f")

    print("\nPart A — model-free hedge:")
    print(f"  Flat IV:   {hedge_data['model_free_var_flat_surface']*10000:.2f} (x10^4)")
    print(f"  Skewed IV: {hedge_data['model_free_var_skewed_surface']*10000:.2f} (x10^4)")
    print(f"  Weights identical -> hedge is model-free")

    print("\nPart B — daily P&L:")
    print(f"  Notional:           ${pnl_data['notional']:,.0f}")
    print(f"  K_var:              {pnl_data['k_var']*10000:.1f} bps")
    print(f"  Unhedged sigma:     ${pnl_data['std_terminal_unhedged']:,.0f}")
    print(f"  Hedged sigma:       ${pnl_data['std_terminal_hedged']:,.0f}")
    print(f"  Variance reduction: {pnl_data['var_reduction_pct']:.1f}%")

    print("\nPart C — backtest 2019-2024:")
    r = result.monthly_pnl["net_pnl"] / VEGA_TARGET
    sharpe = r.mean() / r.std() * (MONTHS_PER_YEAR ** 0.5)
    cum = (1 + r).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd*100:.1f}%")
    print(f"  Win Rate:     {(r > 0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
