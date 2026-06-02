"""
W7 (Vol Analyst) — implied-volatility tools.

Two surfaces are needed to compare smile shapes:

  * the VIX implied-volatility surface ("vol-of-vol" surface). VIX option
    prices come from the COS engine (``build_vix_option_surface``); each price
    is inverted into a Black-76 implied volatility using the VIX futures as the
    forward.
  * the SPX implied-volatility smile. SPX option prices come from a Heston
    Monte-Carlo simulation; each price is inverted into a Black-76 implied
    volatility using the SPX forward F = S0 * exp(r*T).

Both inversions share one Black-76 model so the two smiles are directly
comparable on the same axes.
"""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from src.models.heston import HestonParams, simulate_heston_paths
from src.pricing.vix_options_pricer import VIXOptionSurface


def black76_price(
    forward: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    is_call: bool,
) -> float:
    """Black-76 price of a European option written on a forward `forward`.

        C = exp(-r*T) * (F * N(d1) - K * N(d2))
        P = exp(-r*T) * (K * N(-d2) - F * N(-d1))
        d1 = (ln(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
    """
    if sigma <= 0 or maturity <= 0:
        # Degenerate (zero-vol) intrinsic value.
        intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
        return exp(-r * maturity) * intrinsic

    disc = exp(-r * maturity)
    vol_sqrt_t = sigma * sqrt(maturity)
    d1 = (log(forward / strike) + 0.5 * sigma**2 * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if is_call:
        return disc * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return disc * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def black76_implied_vol(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    r: float,
    is_call: bool,
    vol_bounds: tuple[float, float] = (1e-4, 10.0),
) -> float:
    """Invert a Black-76 price into an implied volatility via Brent's method.

    Returns ``np.nan`` when the price is below intrinsic / above the no-arbitrage
    bound, i.e. when no positive-vol root exists in ``vol_bounds``.
    """
    disc = exp(-r * maturity)
    intrinsic = disc * (
        max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    )
    upper_bound = disc * (forward if is_call else strike)

    if price <= intrinsic + 1e-12 or price >= upper_bound - 1e-12:
        return float("nan")

    def objective(sigma: float) -> float:
        return black76_price(forward, strike, maturity, sigma, r, is_call) - price

    lo, hi = vol_bounds
    if objective(lo) > 0 or objective(hi) < 0:
        return float("nan")

    return float(brentq(objective, lo, hi, xtol=1e-8, maxiter=200))


def vix_implied_vol_surface(
    surface: VIXOptionSurface,
    maturities: np.ndarray,
    strikes: np.ndarray,
    r: float = 0.03,
) -> np.ndarray:
    """Convert a VIX *price* surface into a Black-76 *implied-vol* surface.

    For each maturity the VIX futures price ``surface.forward_vix[i]`` is the
    forward. Out-of-the-money options are inverted (calls for K >= F, puts for
    K < F) because they carry all the time value and give the most stable root.

    Returns an array of shape ``(n_maturities, n_strikes)`` in vol-of-vol units
    (annualised volatility of the VIX, as a decimal).
    """
    maturities = np.asarray(maturities, dtype=float)
    strikes = np.asarray(strikes, dtype=float)

    iv = np.full((len(maturities), len(strikes)), np.nan)
    for i, T in enumerate(maturities):
        forward = float(surface.forward_vix[i])
        for j, K in enumerate(strikes):
            is_call = K >= forward
            price = surface.call_prices[i, j] if is_call else surface.put_prices[i, j]
            iv[i, j] = black76_implied_vol(
                price=price,
                forward=forward,
                strike=float(K),
                maturity=float(T),
                r=r,
                is_call=is_call,
            )
    return iv


def spx_implied_vol_smile(
    params: HestonParams,
    maturity: float,
    strikes: np.ndarray,
    paths: int = 200_000,
    steps_per_year: int = 252,
    seed: int | None = 7,
) -> dict[str, np.ndarray]:
    """SPX implied-vol smile under Heston, via Monte-Carlo pricing + inversion.

    The terminal SPX price is simulated once (common random numbers across
    strikes), out-of-the-money payoffs are discounted to prices, and each price
    is inverted into a Black-76 implied volatility against the SPX forward
    F = S0 * exp(r*T).

    Returns dict with keys ``strikes``, ``moneyness`` (K/F), ``implied_vol``,
    ``forward``.
    """
    strikes = np.asarray(strikes, dtype=float)
    steps = max(1, int(round(maturity * steps_per_year)))

    _, stock_paths, _ = simulate_heston_paths(
        params=params,
        maturity=maturity,
        steps=steps,
        paths=paths,
        seed=seed,
    )
    terminal = stock_paths[:, -1]

    forward = params.s0 * exp(params.r * maturity)
    disc = exp(-params.r * maturity)

    implied_vol = np.full(len(strikes), np.nan)
    for j, K in enumerate(strikes):
        is_call = K >= forward
        payoff = np.maximum(terminal - K, 0.0) if is_call else np.maximum(K - terminal, 0.0)
        price = disc * float(np.mean(payoff))
        implied_vol[j] = black76_implied_vol(
            price=price,
            forward=forward,
            strike=float(K),
            maturity=maturity,
            r=params.r,
            is_call=is_call,
        )

    return {
        "strikes": strikes,
        "moneyness": strikes / forward,
        "implied_vol": implied_vol,
        "forward": forward,
    }
