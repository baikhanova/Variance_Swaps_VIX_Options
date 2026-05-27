"""
Black-Scholes implied volatility inversion.

The market option chain provides mid-prices (and optionally bid/ask),
but the W6 calibration objective is expressed in implied volatility
space because IV is comparable across strikes and maturities.

This module provides Black-Scholes formulas and a Brent-based root
finder for the implied volatility consistent with an observed price.
"""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes formulas
# ─────────────────────────────────────────────────────────────────────────────


def bs_call_price(
    s0: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Black-Scholes price of a European call.

    The dividend yield q is included so that the formula also works for
    SPX index options when a continuous dividend assumption is used.
    """
    if maturity <= 0:
        return max(s0 * exp(-dividend_yield * maturity)
                   - strike * exp(-rate * maturity), 0.0)
    if sigma <= 0:
        return max(s0 * exp(-dividend_yield * maturity)
                   - strike * exp(-rate * maturity), 0.0)

    forward = s0 * exp((rate - dividend_yield) * maturity)
    discount = exp(-rate * maturity)

    sigma_root_t = sigma * sqrt(maturity)
    d1 = (log(forward / strike) + 0.5 * sigma_root_t**2) / sigma_root_t
    d2 = d1 - sigma_root_t

    return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def bs_put_price(
    s0: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Black-Scholes price of a European put via put-call parity.
    """
    call = bs_call_price(
        s0=s0,
        strike=strike,
        maturity=maturity,
        rate=rate,
        sigma=sigma,
        dividend_yield=dividend_yield,
    )

    discount_strike = strike * exp(-rate * maturity)
    discount_spot = s0 * exp(-dividend_yield * maturity)

    return call - discount_spot + discount_strike


def bs_vega(
    s0: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Black-Scholes vega — sensitivity of price to a one-unit change in sigma.

    The vega is the same for calls and puts. It is used in the calibration
    objective as a natural weighting for IV residuals.
    """
    if maturity <= 0 or sigma <= 0:
        return 0.0

    forward = s0 * exp((rate - dividend_yield) * maturity)
    sigma_root_t = sigma * sqrt(maturity)
    d1 = (log(forward / strike) + 0.5 * sigma_root_t**2) / sigma_root_t

    return s0 * exp(-dividend_yield * maturity) * norm.pdf(d1) * sqrt(maturity)


# ─────────────────────────────────────────────────────────────────────────────
# Implied volatility inversion
# ─────────────────────────────────────────────────────────────────────────────


def implied_volatility(
    price: float,
    s0: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    sigma_low: float = 1e-4,
    sigma_high: float = 5.0,
    tolerance: float = 1e-8,
) -> float:
    """
    Invert the Black-Scholes formula to recover implied volatility.

    Returns NaN if the observed price violates no-arbitrage bounds or if
    the search interval does not bracket a root. The Brent solver is used
    because the BS price is monotone in sigma.
    """
    if price <= 0 or maturity <= 0:
        return float("nan")

    discount_strike = strike * exp(-rate * maturity)
    discount_spot = s0 * exp(-dividend_yield * maturity)

    if option_type == "call":
        intrinsic = max(discount_spot - discount_strike, 0.0)
        upper_bound = discount_spot
    elif option_type == "put":
        intrinsic = max(discount_strike - discount_spot, 0.0)
        upper_bound = discount_strike
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    if price < intrinsic - 1e-10 or price > upper_bound + 1e-10:
        return float("nan")

    def objective(sigma: float) -> float:
        if option_type == "call":
            model = bs_call_price(s0, strike, maturity, rate, sigma, dividend_yield)
        else:
            model = bs_put_price(s0, strike, maturity, rate, sigma, dividend_yield)
        return model - price

    try:
        return float(brentq(objective, sigma_low, sigma_high, xtol=tolerance))
    except ValueError:
        return float("nan")


def implied_volatility_vector(
    prices: np.ndarray,
    s0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    rate: float,
    option_types: np.ndarray,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """
    Vectorised wrapper around the scalar inversion.

    Each market quote is inverted independently. The function loops in
    Python; this is acceptable because typical SPX option chains have
    only a few hundred observations per snapshot.
    """
    prices = np.asarray(prices, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    option_types = np.asarray(option_types)

    n = len(prices)
    if not (len(strikes) == len(maturities) == len(option_types) == n):
        raise ValueError("Input arrays must have matching length.")

    ivs = np.full(n, np.nan)
    for i in range(n):
        ivs[i] = implied_volatility(
            price=float(prices[i]),
            s0=s0,
            strike=float(strikes[i]),
            maturity=float(maturities[i]),
            rate=rate,
            option_type=str(option_types[i]),
            dividend_yield=dividend_yield,
        )

    return ivs
