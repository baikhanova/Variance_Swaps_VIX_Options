from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.models.cir import (
    CIRParams,
    cir_truncation_interval,
    vix_level_from_variance,
)
from src.pricing.vix_cos import (
    COSSettings,
    cos_density_recovery,
    price_vix_call_cos,
    price_vix_futures_cos,
    price_vix_put_cos,
)

def convergence_over_n_terms(
    params: CIRParams,
    option_maturity: float,
    strike: float,
    r: float = 0.03,
    delta: float = 30 / 365,
    n_terms_grid: Sequence[int] | None = None,
    reference_n_terms: int = 2048,
) -> pd.DataFrame:
    """
    Study convergence of the COS VIX call price as N (number of terms) grows.

    A high-N reference price is computed first.  For each N in n_terms_grid
    the absolute error  |price(N) - reference|  and relative error are stored.

    Parameters
    ----------
    params          : CIR parameters.
    option_maturity : Option expiry in years.
    strike          : VIX call strike (index points, e.g. 20.0).
    r               : Risk-free rate.
    delta           : VIX horizon (30/365 by convention).
    n_terms_grid    : Sequence of N values to sweep.
    reference_n_terms: N used to build the reference price.

    Returns
    -------
    DataFrame with columns:
        n_terms | cos_price | abs_error | rel_error | log2_abs_error
    """
    if n_terms_grid is None:
        n_terms_grid = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    ref_settings = COSSettings(n_terms=reference_n_terms, truncation_std_width=10.0)
    ref_result = price_vix_call_cos(
        params=params,
        option_maturity=option_maturity,
        strike=strike,
        r=r,
        delta=delta,
        settings=ref_settings,
    )
    ref_price = ref_result["vix_call_cos_price"]

    rows = []
    for n in n_terms_grid:
        settings = COSSettings(n_terms=n, truncation_std_width=8.0)
        result = price_vix_call_cos(
            params=params,
            option_maturity=option_maturity,
            strike=strike,
            r=r,
            delta=delta,
            settings=settings,
        )
        price = result["vix_call_cos_price"]
        abs_err = abs(price - ref_price)
        rel_err = abs_err / ref_price if ref_price != 0 else np.nan
        rows.append(
            {
                "n_terms": n,
                "cos_price": price,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "log2_abs_error": np.log2(abs_err) if abs_err > 0 else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["reference_price"] = ref_price
    df.attrs["reference_n_terms"] = reference_n_terms
    return df


def convergence_over_truncation_width(
    params: CIRParams,
    option_maturity: float,
    strike: float,
    r: float = 0.03,
    delta: float = 30 / 365,
    std_widths: Sequence[float] | None = None,
    n_terms: int = 256,
    reference_std_width: float = 12.0,
) -> pd.DataFrame:
    """
    Study sensitivity of the COS price to the truncation interval [a, b].

    The interval is controlled by std_width: b = mean + std_width * std(v_T).
    A larger width reduces truncation error but may increase COS approximation
    error if the interval is much wider than the density support.

    Returns
    -------
    DataFrame with columns:
        std_width | lower | upper | cos_price | abs_error | rel_error
    """
    if std_widths is None:
        std_widths = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]

    # Reference
    ref_settings = COSSettings(n_terms=n_terms, truncation_std_width=reference_std_width)
    ref = price_vix_call_cos(
        params=params,
        option_maturity=option_maturity,
        strike=strike,
        r=r,
        delta=delta,
        settings=ref_settings,
    )
    ref_price = ref["vix_call_cos_price"]

    rows = []
    for w in std_widths:
        settings = COSSettings(n_terms=n_terms, truncation_std_width=w)
        lower, upper = cir_truncation_interval(params=params, maturity=option_maturity, std_width=w)
        result = price_vix_call_cos(
            params=params,
            option_maturity=option_maturity,
            strike=strike,
            r=r,
            delta=delta,
            settings=settings,
        )
        price = result["vix_call_cos_price"]
        abs_err = abs(price - ref_price)
        rel_err = abs_err / ref_price if ref_price != 0 else np.nan
        rows.append(
            {
                "std_width": w,
                "lower": lower,
                "upper": upper,
                "cos_price": price,
                "abs_error": abs_err,
                "rel_error": rel_err,
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["reference_price"] = ref_price
    return df

def cross_validate_with_mc(
    params: CIRParams,
    strikes: Sequence[float],
    option_maturity: float,
    mc_call_prices: Sequence[float],
    mc_put_prices: Sequence[float],
    mc_futures_price: float,
    r: float = 0.03,
    delta: float = 30 / 365,
    cos_settings: COSSettings | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """
    Cross-validate COS prices against W3 Monte Carlo results.

    Parameters
    ----------
    mc_call_prices : MC call price for each strike (same order as strikes).
    mc_put_prices  : MC put price for each strike.
    mc_futures_price: MC VIX futures price.

    Returns
    -------
    dict with keys:
        'futures'  — single-row DataFrame (COS vs MC futures price).
        'options'  — DataFrame by strike (call + put COS vs MC).
        'summary'  — dict with max absolute error and mean relative error.
    """
    if cos_settings is None:
        cos_settings = COSSettings(n_terms=256, truncation_std_width=8.0)

    # ── Futures ──────────────────────────────────────────────────────────────
    cos_fut = price_vix_futures_cos(
        params=params, maturity=option_maturity, r=r, delta=delta, settings=cos_settings
    )
    cos_futures_price = cos_fut["vix_futures_price"]
    fut_abs_err = abs(cos_futures_price - mc_futures_price)
    fut_rel_err = fut_abs_err / mc_futures_price if mc_futures_price != 0 else np.nan

    futures_df = pd.DataFrame(
        [
            {
                "cos_futures": cos_futures_price,
                "mc_futures": mc_futures_price,
                "abs_error": fut_abs_err,
                "rel_error_pct": 100 * fut_rel_err,
            }
        ]
    )

    rows = []
    for k, mc_c, mc_p in zip(strikes, mc_call_prices, mc_put_prices):
        cos_c = price_vix_call_cos(
            params=params,
            option_maturity=option_maturity,
            strike=float(k),
            r=r,
            delta=delta,
            settings=cos_settings,
        )["vix_call_cos_price"]
        cos_p = price_vix_put_cos(
            params=params,
            option_maturity=option_maturity,
            strike=float(k),
            r=r,
            delta=delta,
            settings=cos_settings,
        )["vix_put_cos_price"]

        rows.append(
            {
                "strike": k,
                "cos_call": cos_c,
                "mc_call": mc_c,
                "call_abs_err": abs(cos_c - mc_c),
                "call_rel_err_pct": 100 * abs(cos_c - mc_c) / mc_c if mc_c != 0 else np.nan,
                "cos_put": cos_p,
                "mc_put": mc_p,
                "put_abs_err": abs(cos_p - mc_p),
                "put_rel_err_pct": 100 * abs(cos_p - mc_p) / mc_p if mc_p != 0 else np.nan,
            }
        )

    options_df = pd.DataFrame(rows)

    summary = {
        "futures_abs_error": fut_abs_err,
        "futures_rel_error_pct": 100 * fut_rel_err,
        "max_call_abs_error": options_df["call_abs_err"].max(),
        "mean_call_rel_error_pct": options_df["call_rel_err_pct"].mean(),
        "max_put_abs_error": options_df["put_abs_err"].max(),
        "mean_put_rel_error_pct": options_df["put_rel_err_pct"].mean(),
    }

    return {"futures": futures_df, "options": options_df, "summary": summary}

def speed_benchmark_cos(
    params: CIRParams,
    option_maturity: float,
    strike: float,
    r: float = 0.03,
    delta: float = 30 / 365,
    cos_settings: COSSettings | None = None,
    n_repeats: int = 200,
) -> dict[str, float]:
    """
    Benchmark COS pricing speed (seconds per price).

    Run the COS pricer n_repeats times and report median and mean elapsed time.
    The first call is excluded to avoid Python import overhead.

    Returns
    -------
    dict:
        cos_median_s  — median time per call (seconds)
        cos_mean_s    — mean time per call (seconds)
        n_repeats     — number of repetitions used
    """
    if cos_settings is None:
        cos_settings = COSSettings(n_terms=256, truncation_std_width=8.0)

    # Warm-up
    price_vix_call_cos(
        params=params,
        option_maturity=option_maturity,
        strike=strike,
        r=r,
        delta=delta,
        settings=cos_settings,
    )

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        price_vix_call_cos(
            params=params,
            option_maturity=option_maturity,
            strike=strike,
            r=r,
            delta=delta,
            settings=cos_settings,
        )
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return {
        "cos_median_s": float(np.median(times_arr)),
        "cos_mean_s": float(np.mean(times_arr)),
        "cos_min_s": float(np.min(times_arr)),
        "n_repeats": n_repeats,
    }


def speed_benchmark_mc(
    mc_price_fn,
    n_repeats: int = 20,
) -> dict[str, float]:
    """
    Benchmark a Monte Carlo pricing function provided by W3.

    Parameters
    ----------
    mc_price_fn : callable with no arguments that returns a price.
                  Example: lambda: price_vix_call_mc(params, ...)
    n_repeats   : Number of MC runs to time (keep small — MC is slow).

    Returns
    -------
    dict:
        mc_median_s  — median time per call (seconds)
        mc_mean_s    — mean time per call (seconds)
        n_repeats    — number of repetitions
    """
    # Warm-up
    mc_price_fn()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        mc_price_fn()
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return {
        "mc_median_s": float(np.median(times_arr)),
        "mc_mean_s": float(np.mean(times_arr)),
        "mc_min_s": float(np.min(times_arr)),
        "n_repeats": n_repeats,
    }


def speedup_summary(cos_bench: dict, mc_bench: dict) -> dict[str, float]:
    """
    Compute speedup factor: MC time / COS time.

    Returns a dict with median_speedup and mean_speedup.
    """
    median_speedup = mc_bench["mc_median_s"] / cos_bench["cos_median_s"]
    mean_speedup = mc_bench["mc_mean_s"] / cos_bench["cos_mean_s"]
    return {
        "median_speedup_x": median_speedup,
        "mean_speedup_x": mean_speedup,
        "cos_median_ms": 1000 * cos_bench["cos_median_s"],
        "mc_median_ms": 1000 * mc_bench["mc_median_s"],
    }

def model_vix_term_structure(
    params: CIRParams,
    maturities: Sequence[float] | None = None,
    r: float = 0.03,
    delta: float = 30 / 365,
    cos_settings: COSSettings | None = None,
) -> pd.DataFrame:
    """
    Compute the model VIX futures curve F(0, T) for a range of maturities.

    Parameters
    ----------
    maturities : Option expiries in years (e.g. [1/12, 2/12, ..., 8/12]).

    Returns
    -------
    DataFrame with columns:
        maturity_years | maturity_months | model_futures_price
    """
    if maturities is None:
        # Standard VIX futures maturities: monthly out to 8 months
        maturities = [m / 12 for m in range(1, 9)]

    if cos_settings is None:
        cos_settings = COSSettings(n_terms=256, truncation_std_width=8.0)

    rows = []
    for T in maturities:
        result = price_vix_futures_cos(
            params=params,
            maturity=T,
            r=r,
            delta=delta,
            settings=cos_settings,
        )
        rows.append(
            {
                "maturity_years": T,
                "maturity_months": round(T * 12, 2),
                "model_futures_price": result["vix_futures_price"],
                "expected_vix": result["expected_vix"],
            }
        )

    return pd.DataFrame(rows)


def compare_term_structure(
    model_curve: pd.DataFrame,
    market_prices: Sequence[float],
    market_maturities: Sequence[float],
) -> pd.DataFrame:
    """
    Merge model and market VIX futures term structures for comparison.

    Parameters
    ----------
    model_curve       : Output of model_vix_term_structure().
    market_prices     : Market VIX futures settlement prices (VIX index points).
    market_maturities : Corresponding maturities in years.

    Returns
    -------
    DataFrame with columns:
        maturity_months | model_futures | market_futures | error | rel_error_pct
    """
    market_df = pd.DataFrame(
        {
            "maturity_years": list(market_maturities),
            "market_futures": list(market_prices),
        }
    )

    merged = pd.merge_asof(
        model_curve.sort_values("maturity_years"),
        market_df.sort_values("maturity_years"),
        on="maturity_years",
        tolerance=1 / 365,
        direction="nearest",
    )

    merged["error"] = merged["model_futures_price"] - merged["market_futures"]
    merged["rel_error_pct"] = 100 * merged["error"] / merged["market_futures"]

    return merged[[
        "maturity_months",
        "model_futures_price",
        "market_futures",
        "error",
        "rel_error_pct",
    ]]


def fetch_vix_futures_from_yahoo(
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download the most recent VIX futures closing prices from Yahoo Finance.

    VIX futures trade under tickers ^VIX1, ^VIX2, ... on Yahoo Finance.
    This is a best-effort download; if Yahoo data is unavailable, the function
    returns an empty DataFrame and the caller should supply market_prices manually.

    Returns
    -------
    DataFrame with columns: ticker | last_price | maturity_approx_months
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame(columns=["ticker", "last_price", "maturity_approx_months"])

    if tickers is None:
        # Yahoo Finance VIX front-month futures tickers
        tickers = ["^VIX1", "^VIX2", "^VIX3", "^VIX4", "^VIX5", "^VIX6"]

    rows = []
    for i, ticker in enumerate(tickers, start=1):
        try:
            data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if not data.empty:
                price = float(data["Close"].dropna().iloc[-1])
                rows.append(
                    {
                        "ticker": ticker,
                        "last_price": price,
                        "maturity_approx_months": i,
                        "maturity_years": i / 12,
                    }
                )
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ticker", "last_price", "maturity_approx_months", "maturity_years"]
    )

def density_recovery_check(
    params: CIRParams,
    maturity: float,
    n_terms: int = 256,
    std_width: float = 8.0,
    n_grid: int = 500,
) -> pd.DataFrame:
    """
    Verify that the COS-recovered density integrates to 1 and is non-negative.

    Also returns the VIX density (change of variables from variance density).

    Returns
    -------
    DataFrame with columns:
        variance | density_v | vix_level | density_vix_approx
    """
    lower, upper = cir_truncation_interval(
        params=params, maturity=maturity, std_width=std_width
    )
    v_grid = np.linspace(lower + 1e-8, upper, n_grid)

    density_v = cos_density_recovery(
        v_grid=v_grid,
        params=params,
        maturity=maturity,
        lower=lower,
        upper=upper,
        n_terms=n_terms,
    )

    delta = 30 / 365
    vix_levels = vix_level_from_variance(v_grid, params, delta)

    from src.models.cir import vix_affine_coefficients
    a_coef, b_coef = vix_affine_coefficients(params, delta)
    vix_sq = a_coef + b_coef * v_grid
    vix_sq = np.maximum(vix_sq, 1e-10)
    d_vix_d_v = 100.0 * b_coef / (2.0 * np.sqrt(vix_sq))
    density_vix = density_v / np.maximum(d_vix_d_v, 1e-12)

    integral = float(np.trapezoid(density_v, v_grid) if hasattr(np, "trapezoid") else np.trapz(density_v, v_grid))

    df = pd.DataFrame(
        {
            "variance": v_grid,
            "density_v": density_v,
            "vix_level": vix_levels,
            "density_vix_approx": density_vix,
        }
    )
    df.attrs["integral"] = integral
    df.attrs["lower"] = lower
    df.attrs["upper"] = upper
    return df
