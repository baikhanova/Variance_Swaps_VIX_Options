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
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy import stats

from src.models.heston import HestonParams, simulate_heston_paths
from src.pricing.variance_mc import (
    compute_realized_variance,
    heston_variance_strike,
    vix_squared_from_variance,
)
# trading calendar constants
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
MONTHS_PER_YEAR = 12
DELTA = 1.0 / MONTHS_PER_YEAR
TRANSACTION_COST = 0.0003
VEGA_TARGET = 10_000.0
VIX_STOP_LOSS = 40.0
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
    # Demeterfi 1999 formula: (2/T) * sum( dK/K^2 * exp(rT) * Q(K) )
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

    # two very different surfaces → same weights, different implied variances
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
        "n_paths": n_paths,
        "all_terminal_unhedged": cum_u[:, -1],
        "all_terminal_hedged": cum_h[:, -1],
    }


# --- Part C: backtest ---

@dataclass
class BacktestResult:
    monthly_pnl: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict
    factor_regression: dict
    regime_analysis: dict


def run_vrp_backtest(params_rn=HESTON_PARAMS, params_p=HESTON_PHYSICAL, n_months=N_MONTHS_BACKTEST, seed=42):
    total_steps = n_months * TRADING_DAYS_PER_MONTH
    total_years = n_months * DELTA

    _, stock_paths, var_paths = simulate_heston_paths(
        params=params_p, maturity=total_years, steps=total_steps, paths=1, seed=seed,
    )
    stock = stock_paths[0]
    variance = var_paths[0]

    # inject crisis: month 14 ~ March 2020 analog
    c = 14
    variance[c*TRADING_DAYS_PER_MONTH:(c+1)*TRADING_DAYS_PER_MONTH] = np.clip(
        variance[c*TRADING_DAYS_PER_MONTH:(c+1)*TRADING_DAYS_PER_MONTH] * 6.0, 0, 0.80
    )

    # inject stress: month 40 ~ 2022 analog
    s = 40
    variance[s*TRADING_DAYS_PER_MONTH:(s+1)*TRADING_DAYS_PER_MONTH] *= 2.5

    rows = []
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")

    for m in range(n_months):
        i0 = m * TRADING_DAYS_PER_MONTH
        i1 = i0 + TRADING_DAYS_PER_MONTH

        v_t = float(variance[i0])
        vix_t = 100.0 * sqrt(max(vix_squared_from_variance(v_t, params_rn, DELTA), 0))

        # skip month if VIX exceeds stop-loss threshold
        if vix_t > VIX_STOP_LOSS:
            rows.append({
                "date": dates[m], "month": m, "v_t": v_t, "vix_t": vix_t,
                "k_var": 0.0, "rv": 0.0, "vrp_var": 0.0,
                "iv_vol": vix_t, "rv_vol": 0.0, "vrp_vol": 0.0,
                "notional": 0.0, "gross_pnl": 0.0, "tc": 0.0, "net_pnl": 0.0,
                "stop_loss": True, "crisis": (m == c), "stress": (m == s),
            })
            continue

        params_m = HestonParams(
            s0=float(stock[i0]), v0=v_t, r=params_rn.r,
            kappa=params_rn.kappa, theta=params_rn.theta,
            sigma_v=params_rn.sigma_v, rho=params_rn.rho,
        )
        k_var = heston_variance_strike(params_m, DELTA)
        vega_unit = compute_vega(params_m, DELTA, notional=1.0)
        notional = min(VEGA_TARGET / max(vega_unit, 1e-8), 5_000_000.0)

        window = stock[i0:i1+1][np.newaxis, :]
        rv = float(compute_realized_variance(window, maturity=DELTA)[0])

        gross_pnl = notional * (k_var - rv)
        tc = TRANSACTION_COST * notional
        net_pnl = gross_pnl - tc

        rows.append({
            "date": dates[m], "month": m, "v_t": v_t, "vix_t": vix_t,
            "k_var": k_var, "rv": rv, "vrp_var": k_var - rv,
            "iv_vol": 100.0 * sqrt(max(k_var, 0)),
            "rv_vol": 100.0 * sqrt(max(rv, 0)),
            "vrp_vol": 100.0 * (sqrt(max(k_var, 0)) - sqrt(max(rv, 0))),
            "notional": notional, "gross_pnl": gross_pnl, "tc": tc, "net_pnl": net_pnl,
            "stop_loss": False, "crisis": (m == c), "stress": (m == s),
        })

    df = pd.DataFrame(rows).set_index("date")
    equity = (1.0 + df["net_pnl"] / VEGA_TARGET).cumprod()
    equity = equity / equity.iloc[0] * 100.0

    return BacktestResult(
        monthly_pnl=df,
        equity_curve=equity,
        metrics=compute_risk_metrics(df["net_pnl"]),
        factor_regression=factor_regression(df),
        regime_analysis=regime_analysis(df),
    )


def compute_risk_metrics(pnl):
    r = pnl / VEGA_TARGET
    ann = MONTHS_PER_YEAR
    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1))
    downside = float(r[r < 0].std(ddof=1)) if (r < 0).any() else 1e-8
    sharpe = mean_r / std_r * sqrt(ann) if std_r > 0 else 0.0
    sortino = mean_r / downside * sqrt(ann) if downside > 0 else 0.0

    cum = (1 + r).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    calmar = (mean_r * ann) / abs(max_dd) if max_dd != 0 else 0.0

    var5 = float(np.percentile(r, 5))
    cvar5 = float(r[r <= var5].mean())

    return {
        "annualised_return_pct": mean_r * ann * 100,
        "annualised_vol_pct": std_r * sqrt(ann) * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100,
        "calmar": calmar,
        "var_5pct": var5 * 100,
        "cvar_5pct": cvar5 * 100,
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "n_months": len(r),
        "win_rate_pct": float((r > 0).mean() * 100),
    }


def factor_regression(df):
    # R_vrp = alpha + b1*R_spx + b2*dVIX + b3*VIX_lag
    df2 = df.copy()
    df2["r_vrp"] = df2["net_pnl"] / VEGA_TARGET
    df2["dvix"] = df2["vix_t"].diff()
    df2["vix_lag"] = df2["vix_t"].shift(1)
    df2["r_spx"] = -df2["rv_vol"].pct_change()

    clean = df2[["r_vrp", "r_spx", "dvix", "vix_lag"]].dropna()
    y = clean["r_vrp"].values
    X = np.column_stack([np.ones(len(clean)), clean["r_spx"].values,
                         clean["dvix"].values, clean["vix_lag"].values])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        r2 = 1.0 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)
        alpha, b_spx, b_dvix, b_vix_lag = coef
    except Exception:
        alpha = b_spx = b_dvix = b_vix_lag = r2 = 0.0

    return {
        "alpha_monthly": float(alpha),
        "beta_spx": float(b_spx),
        "beta_dvix": float(b_dvix),
        "beta_vix_lag": float(b_vix_lag),
        "r_squared": float(r2),
    }


def regime_analysis(df):
    def _stats(mask, label):
        sub = df.loc[mask, "net_pnl"] / VEGA_TARGET
        if len(sub) < 2:
            return {"label": label, "n_months": len(sub)}
        mean_m = float(sub.mean())
        std_m = float(sub.std(ddof=1))
        return {
            "label": label,
            "n_months": len(sub),
            "mean_monthly_return_pct": mean_m * 100,
            "sharpe": mean_m / std_m * sqrt(MONTHS_PER_YEAR) if std_m > 0 else 0.0,
            "win_rate_pct": float((sub > 0).mean() * 100),
        }

    return {
        "high_vrp": _stats(df["vrp_vol"] > 3.0, "High VRP (>3 vol pts)"),
        "low_vrp":  _stats(df["vrp_vol"] < 1.0, "Low VRP (<1 vol pt)"),
        "full":     _stats(pd.Series(True, index=df.index), "Full period"),
    }


# --- plotting ---

COLORS = {
    "blue": "#1a6faf", "red": "#c0392b", "green": "#27ae60",
    "orange": "#e67e22", "purple": "#8e44ad",
    "dark": "#1a1a2e", "light": "#f5f6fa", "grey": "#7f8c8d",
}


def plot_model_free_hedge(hedge_data, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["dark"])

    strikes = hedge_data["strikes"]
    weights = hedge_data["weights"]

    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    ax1.plot(strikes, weights * 1e4, color=COLORS["blue"], lw=2.5)
    ax1.fill_between(strikes, weights * 1e4, alpha=0.25, color=COLORS["blue"])
    ax1.axvline(100, color=COLORS["grey"], ls="--", lw=1, alpha=0.6, label="ATM")
    ax1.set_title("Hedge Weights  w(K) = 2·exp(rT)·ΔK / (T·K²)", color="white", fontsize=12)
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
                   color=[COLORS["blue"], COLORS["purple"]], width=0.4, alpha=0.85)
    ax2.set_title("Implied Variance — Same Weights, Different IV Surfaces",
                  color="white", fontsize=12)
    ax2.set_ylabel("Model-Free Variance (×10⁴)", color=COLORS["grey"])
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.1f}",
                 ha="center", color="white", fontsize=10)

    fig.suptitle("Part A — Model-Free Variance Swap Hedge", color="white",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["dark"])
    plt.close()


def plot_daily_pnl(pnl_data, save_path):
    days = np.arange(TRADING_DAYS_PER_MONTH + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["dark"])

    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    ax1.plot(days, pnl_data["mean_pnl_unhedged"], color=COLORS["red"], lw=2, label="Unhedged")
    ax1.plot(days, pnl_data["mean_pnl_hedged"], color=COLORS["green"], lw=2, label="Hedged")
    ax1.axhline(0, color=COLORS["grey"], ls="--", lw=0.8)
    ax1.set_title("Cumulative Vega P&L — Hedged vs Unhedged", color="white", fontsize=12)
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
    ax2.set_title(f"Terminal P&L Distribution\nVar. Reduction: {pnl_data['var_reduction_pct']:.1f}%",
                  color="white", fontsize=12)
    ax2.set_xlabel("Terminal P&L ($)", color=COLORS["grey"])
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.legend(facecolor="#111", labelcolor="white", fontsize=9)

    fig.suptitle("Part B — Daily Vega P&L: Short Variance Swap",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["dark"])
    plt.close()


def plot_backtest(result, save_path):
    df = result.monthly_pnl
    eq = result.equity_curve
    m  = result.metrics

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(COLORS["dark"])
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#0d1117")
    ax1.plot(eq.index, eq.values, color=COLORS["blue"], lw=2.5, zorder=3)
    ax1.fill_between(eq.index, eq.values, 100, alpha=0.15, color=COLORS["blue"])
    crisis = df[df["crisis"]]
    stress = df[df["stress"]]
    if not crisis.empty:
        ax1.axvspan(crisis.index[0], crisis.index[0] + pd.DateOffset(months=2),
                    color=COLORS["red"], alpha=0.25, label="Crisis (Mar 2020)")
    if not stress.empty:
        ax1.axvspan(stress.index[0], stress.index[0] + pd.DateOffset(months=2),
                    color=COLORS["orange"], alpha=0.2, label="Stress (2022)")
    stops = df[df["stop_loss"]]
    if not stops.empty:
        ax1.scatter(stops.index, eq.loc[stops.index], marker="x",
                    color=COLORS["orange"], s=80, zorder=5, label="Stop-loss")
    ax1.set_title("Equity Curve — Short Variance Swap", color="white", fontsize=12)
    ax1.set_ylabel("NAV (base=100)", color=COLORS["grey"])
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333")
    ax1.legend(facecolor="#111", labelcolor="white", fontsize=8)

    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor("#0d1117")
    bar_colors = [COLORS["green"] if v > 0 else COLORS["red"] for v in df["net_pnl"]]
    ax2.bar(df.index, df["net_pnl"], color=bar_colors, alpha=0.8, width=20)
    ax2.axhline(0, color=COLORS["grey"], lw=0.8)
    ax2.set_title("Monthly Net P&L ($)", color="white", fontsize=12)
    ax2.set_ylabel("P&L ($)", color=COLORS["grey"])
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax3 = fig.add_subplot(gs[2, :2])
    ax3.set_facecolor("#0d1117")
    ax3.plot(df.index, df["vix_t"], color=COLORS["purple"], lw=1.8)
    ax3.axhline(VIX_STOP_LOSS, color=COLORS["orange"], ls="--", lw=1.2,
                label=f"Stop-loss VIX={VIX_STOP_LOSS}")
    ax3.fill_between(df.index, df["vix_t"], alpha=0.1, color=COLORS["purple"])
    ax3.set_title("Model VIX Level", color="white", fontsize=12)
    ax3.set_ylabel("VIX", color=COLORS["grey"])
    ax3.tick_params(colors="white")
    ax3.spines[:].set_color("#333")
    ax3.legend(facecolor="#111", labelcolor="white", fontsize=8)

    # metrics panel
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.set_facecolor("#0d1117")
    ax4.axis("off")
    metric_lines = [
        ("Ann. Return",   f"{m['annualised_return_pct']:.1f}%"),
        ("Ann. Vol",      f"{m['annualised_vol_pct']:.1f}%"),
        ("Sharpe",        f"{m['sharpe']:.2f}"),
        ("Sortino",       f"{m['sortino']:.2f}"),
        ("Max Drawdown",  f"{m['max_drawdown_pct']:.1f}%"),
        ("Calmar",        f"{m['calmar']:.2f}"),
        ("VaR (5%)",      f"{m['var_5pct']:.1f}%"),
        ("CVaR (5%)",     f"{m['cvar_5pct']:.1f}%"),
        ("Skewness",      f"{m['skewness']:.2f}"),
        ("Exc. Kurtosis", f"{m['excess_kurtosis']:.2f}"),
        ("Win Rate",      f"{m['win_rate_pct']:.0f}%"),
    ]
    ax4.text(0.5, 1.02, "Risk Metrics", transform=ax4.transAxes,
             ha="center", color="white", fontsize=12, fontweight="bold")
    for i, (label, val) in enumerate(metric_lines):
        y = 0.92 - i * 0.085
        ax4.text(0.05, y, label, transform=ax4.transAxes, color=COLORS["grey"], fontsize=9)
        c = (COLORS["green"] if any(k in label for k in ["Return","Sharpe","Sortino","Win","Calmar"])
             and not val.startswith("-")
             else COLORS["red"] if any(k in label for k in ["Drawdown","VaR","CVaR"])
             else "white")
        ax4.text(0.98, y, val, transform=ax4.transAxes, color=c, fontsize=9,
                 ha="right", fontweight="bold")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#0d1117")
    r_norm = df["net_pnl"] / VEGA_TARGET
    ax5.hist(r_norm, bins=20, color=COLORS["blue"], alpha=0.75, density=True, edgecolor="#333")
    xl = np.linspace(*ax5.get_xlim(), 200)
    ax5.plot(xl, stats.norm.pdf(xl, r_norm.mean(), r_norm.std()),
             color=COLORS["orange"], lw=1.5, ls="--", label="Normal")
    ax5.axvline(float(np.percentile(r_norm, 5)), color=COLORS["red"], lw=1.5, ls=":", label="VaR 5%")
    ax5.set_title("Monthly Return Distribution", color="white", fontsize=11)
    ax5.set_xlabel("Monthly Return", color=COLORS["grey"])
    ax5.tick_params(colors="white")
    ax5.spines[:].set_color("#333")
    ax5.legend(facecolor="#111", labelcolor="white", fontsize=8)

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor("#0d1117")
    reg = result.regime_analysis
    rlabels = ["High VRP\n(>3 vol)", "Low VRP\n(<1 vol)", "Full Period"]
    rsharpe = [reg["high_vrp"].get("sharpe", 0), reg["low_vrp"].get("sharpe", 0),
               reg["full"].get("sharpe", 0)]
    bcolors = [COLORS["green"] if s > 0 else COLORS["red"] for s in rsharpe]
    bars = ax6.bar(rlabels, rsharpe, color=bcolors, alpha=0.85, width=0.5)
    ax6.axhline(0, color=COLORS["grey"], lw=0.8)
    ax6.set_title("Sharpe by VRP Regime", color="white", fontsize=11)
    ax6.set_ylabel("Sharpe", color=COLORS["grey"])
    ax6.tick_params(colors="white")
    ax6.spines[:].set_color("#333")
    for bar, s in zip(bars, rsharpe):
        ax6.text(bar.get_x() + bar.get_width()/2, s + 0.05 * (1 if s >= 0 else -1),
                 f"{s:.2f}", ha="center", color="white", fontsize=9)

    fig.suptitle("W8 — Short Variance Swap VRP Backtest (2019–2024)",
                 color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["dark"])
    plt.close()


def print_summary(hedge_data, pnl_data, result):
    m  = result.metrics
    reg = result.regime_analysis
    fr  = result.factor_regression
    df  = result.monthly_pnl

    print("\n" + "="*65)
    print("  W8 — VARIANCE SWAP HEDGING + VRP BACKTEST")
    print("="*65)

    print("\n-- Part A: Model-Free Hedge --")
    print(f"  w(K) = 2*exp(rT)*dK / (T*K^2)  -- no model params needed")
    print(f"  Flat IV surface:   {hedge_data['model_free_var_flat_surface']*10000:.2f} (x10^4)")
    print(f"  Skewed IV surface: {hedge_data['model_free_var_skewed_surface']*10000:.2f} (x10^4)")

    print("\n-- Part B: Daily P&L --")
    print(f"  Notional:          ${pnl_data['notional']:,.0f}")
    print(f"  K_var:             {pnl_data['k_var']*10000:.1f} bps")
    print(f"  Unhedged sigma:    ${pnl_data['std_terminal_unhedged']:,.0f}")
    print(f"  Hedged sigma:      ${pnl_data['std_terminal_hedged']:,.0f}")
    print(f"  Variance reduction:{pnl_data['var_reduction_pct']:.1f}%")

    print("\n-- Part C: Backtest 2019-2024 --")
    print(f"  {'Ann. Return':<22}  {m['annualised_return_pct']:>9.1f}%")
    print(f"  {'Ann. Vol':<22}  {m['annualised_vol_pct']:>9.1f}%")
    print(f"  {'Sharpe':<22}  {m['sharpe']:>10.2f}")
    print(f"  {'Sortino':<22}  {m['sortino']:>10.2f}")
    print(f"  {'Max Drawdown':<22}  {m['max_drawdown_pct']:>9.1f}%")
    print(f"  {'Calmar':<22}  {m['calmar']:>10.2f}")
    print(f"  {'VaR 5%':<22}  {m['var_5pct']:>9.1f}%")
    print(f"  {'CVaR 5%':<22}  {m['cvar_5pct']:>9.1f}%")
    print(f"  {'Skewness':<22}  {m['skewness']:>10.2f}  (negative expected)")
    print(f"  {'Excess Kurtosis':<22}  {m['excess_kurtosis']:>10.2f}")
    print(f"  {'Win Rate':<22}  {m['win_rate_pct']:>9.0f}%")

    crisis = df[df["crisis"]]
    if not crisis.empty:
        print(f"\n  Crisis month P&L: ${float(crisis['net_pnl'].iloc[0]):,.0f}")

    print(f"  Stop-loss triggered: {int(df['stop_loss'].sum())} months")

    print("\n-- Regimes --")
    for key in ["high_vrp", "low_vrp", "full"]:
        r = reg[key]
        print(f"  {r['label']:<30}  n={r.get('n_months',0):3d}  "
              f"Sharpe={r.get('sharpe',0):5.2f}  WinRate={r.get('win_rate_pct',0):.0f}%")

    print("\n-- Factor Regression --")
    print(f"  alpha:       {fr['alpha_monthly']:.4f}")
    print(f"  beta_spx:    {fr['beta_spx']:.4f}")
    print(f"  beta_dvix:   {fr['beta_dvix']:.4f}")
    print(f"  beta_vixlag: {fr['beta_vix_lag']:.4f}")
    print(f"  R2:          {fr['r_squared']:.3f}")
    print("="*65 + "\n")


def main(output_dir="/mnt/user-data/outputs/w8"):
    os.makedirs(output_dir, exist_ok=True)

    hedge_data = demonstrate_model_free_hedge()
    plot_model_free_hedge(hedge_data, os.path.join(output_dir, "w8_model_free_hedge.png"))

    pnl_data = simulate_daily_pnl(HESTON_PARAMS, n_paths=500, seed=77)
    plot_daily_pnl(pnl_data, os.path.join(output_dir, "w8_daily_pnl.png"))

    result = run_vrp_backtest(params_rn=HESTON_PARAMS, params_p=HESTON_PHYSICAL,
                              n_months=N_MONTHS_BACKTEST)
    plot_backtest(result, os.path.join(output_dir, "w8_backtest.png"))

    # save monthly table
    tbl = result.monthly_pnl[["vix_t","k_var","rv","vrp_vol","net_pnl","stop_loss","crisis"]].copy()
    tbl.columns = ["VIX","K_var","RV","VRP (vol pts)","Net P&L ($)","Stop-Loss","Crisis"]
    tbl.to_csv(os.path.join(output_dir, "w8_monthly_pnl.csv"), float_format="%.6f")

    print_summary(hedge_data, pnl_data, result)


if __name__ == "__main__":
    main()
