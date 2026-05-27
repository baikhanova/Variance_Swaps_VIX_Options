"""
Heston European option pricing on SPX via the COS method.

The COS method of Fang and Oosterlee (2008) approximates the risk-neutral
expectation of a European payoff using a Fourier-cosine series expansion
of the density. For a payoff g(y) on y = log(S_T) the call/put price is:

    V(x0, T) = exp(-r T) * Re{ sum_{k=0}^{N-1}' phi_k * V_k }

where:
    phi_k = phi(k*pi / (b-a); T, x0) * exp(-i*k*pi*a / (b-a))
    V_k   = (2 / (b-a)) * integral_a^b g(y) cos(k*pi*(y-a)/(b-a)) dy
    prime = the k=0 term is multiplied by 1/2.

The payoff coefficients V_k are known in closed form for European calls
and puts (Fang and Oosterlee 2008, eq. 23 and 24). This module assembles
the full SPX implied-volatility surface used by the W6 calibration.
"""

from __future__ import annotations

from math import exp

import numpy as np
import pandas as pd

from src.models.heston_cf import (
    HestonCFParams,
    cos_truncation_interval,
    heston_log_stock_cf,
)
from src.pricing.iv_inversion import implied_volatility


# ─────────────────────────────────────────────────────────────────────────────
# Closed-form COS payoff coefficients
# ─────────────────────────────────────────────────────────────────────────────


def _chi_psi(
    k: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fang-Oosterlee chi_k and psi_k helpers on the integration sub-interval
    [c, d] sitting inside the truncation interval [a, b]:

        chi_k = integral_c^d exp(y) cos(k pi (y-a)/(b-a)) dy
        psi_k = integral_c^d           cos(k pi (y-a)/(b-a)) dy
    """
    width = b - a
    arg = k * np.pi / width

    cos_d = np.cos(arg * (d - a))
    cos_c = np.cos(arg * (c - a))
    sin_d = np.sin(arg * (d - a))
    sin_c = np.sin(arg * (c - a))

    denom = 1.0 + arg**2
    chi = (1.0 / denom) * (
        cos_d * exp(d) - cos_c * exp(c)
        + arg * sin_d * exp(d) - arg * sin_c * exp(c)
    )

    psi = np.where(
        k == 0,
        d - c,
        (sin_d - sin_c) / np.where(arg == 0, 1.0, arg),
    )

    return chi, psi


def _call_payoff_coefficients(
    n_terms: int,
    a: float,
    b: float,
    strike: float,
) -> np.ndarray:
    """
    Closed-form V_k for the call payoff (e^y - K)^+, integrated over
    the sub-interval [log(K), b].
    """
    k = np.arange(n_terms, dtype=float)
    c = np.log(strike)
    d = b

    chi, psi = _chi_psi(k=k, a=a, b=b, c=c, d=d)
    return (2.0 / (b - a)) * (chi - strike * psi)


def _put_payoff_coefficients(
    n_terms: int,
    a: float,
    b: float,
    strike: float,
) -> np.ndarray:
    """
    Closed-form V_k for the put payoff (K - e^y)^+, integrated over
    the sub-interval [a, log(K)].
    """
    k = np.arange(n_terms, dtype=float)
    c = a
    d = np.log(strike)

    chi, psi = _chi_psi(k=k, a=a, b=b, c=c, d=d)
    return (2.0 / (b - a)) * (strike * psi - chi)


# ─────────────────────────────────────────────────────────────────────────────
# Single-strike pricer
# ─────────────────────────────────────────────────────────────────────────────


def price_european_cos(
    spot: float,
    strike: float,
    maturity: float,
    params: HestonCFParams,
    option_type: str = "call",
    n_terms: int = 160,
    truncation_level: float = 10.0,
) -> float:
    """
    Price a single European call or put on SPX under Heston via COS.

    The truncation interval [a, b] is set from the Heston cumulants
    (Fang-Oosterlee rule). The default N = 160 is enough for IV
    accuracy of about 1e-4 across the typical SPX moneyness range.
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    params.validate()

    a, b = cos_truncation_interval(
        maturity=maturity,
        spot=spot,
        params=params,
        level=truncation_level,
    )

    k = np.arange(n_terms, dtype=float)
    u = k * np.pi / (b - a)

    cf_values = heston_log_stock_cf(
        u=u,
        maturity=maturity,
        spot=spot,
        params=params,
    )

    phase = np.exp(-1j * u * a)
    phi_k = np.real(cf_values * phase)

    if option_type == "call":
        v_k = _call_payoff_coefficients(n_terms=n_terms, a=a, b=b, strike=strike)
    else:
        v_k = _put_payoff_coefficients(n_terms=n_terms, a=a, b=b, strike=strike)

    terms = phi_k * v_k
    terms[0] *= 0.5

    discount = exp(-params.rate * maturity)
    price = discount * float(np.sum(terms))

    return max(price, 0.0)


def price_european_cos_batch(
    spot: float,
    strikes: np.ndarray,
    option_types: np.ndarray,
    maturity: float,
    params: HestonCFParams,
    n_terms: int = 160,
    truncation_level: float = 10.0,
) -> np.ndarray:
    """
    Vectorised COS pricer for many strikes at a single maturity.

    The Heston characteristic function depends on (u, T) but not on the
    strike, so within one maturity the expensive CF evaluation can be
    reused across the whole strike slice. This is the inner loop of the
    SPX calibration objective.
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive.")

    params.validate()
    strikes = np.asarray(strikes, dtype=float)
    option_types = np.asarray(option_types)

    a, b = cos_truncation_interval(
        maturity=maturity,
        spot=spot,
        params=params,
        level=truncation_level,
    )

    k = np.arange(n_terms, dtype=float)
    u = k * np.pi / (b - a)

    cf_values = heston_log_stock_cf(
        u=u,
        maturity=maturity,
        spot=spot,
        params=params,
    )
    phase = np.exp(-1j * u * a)
    phi_k = np.real(cf_values * phase)
    phi_k_weighted = phi_k.copy()
    phi_k_weighted[0] *= 0.5

    discount = exp(-params.rate * maturity)
    prices = np.empty(len(strikes))

    width = b - a
    arg = k * np.pi / width
    denom = 1.0 + arg**2

    for i, (strike, otype) in enumerate(zip(strikes, option_types)):
        if otype == "call":
            c, d = np.log(strike), b
        else:
            c, d = a, np.log(strike)

        cos_d = np.cos(arg * (d - a))
        cos_c = np.cos(arg * (c - a))
        sin_d = np.sin(arg * (d - a))
        sin_c = np.sin(arg * (c - a))

        chi = (1.0 / denom) * (
            cos_d * exp(d) - cos_c * exp(c)
            + arg * sin_d * exp(d) - arg * sin_c * exp(c)
        )
        psi = np.where(
            k == 0,
            d - c,
            (sin_d - sin_c) / np.where(arg == 0, 1.0, arg),
        )

        if otype == "call":
            v_k = (2.0 / width) * (chi - strike * psi)
        else:
            v_k = (2.0 / width) * (strike * psi - chi)

        prices[i] = max(discount * float(np.sum(phi_k_weighted * v_k)), 0.0)

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Surface helpers
# ─────────────────────────────────────────────────────────────────────────────


def model_iv_surface(
    spot: float,
    strikes_by_maturity: dict[float, np.ndarray],
    params: HestonCFParams,
    option_type_by_maturity: dict[float, np.ndarray] | None = None,
    n_terms: int = 160,
    truncation_level: float = 10.0,
) -> pd.DataFrame:
    """
    Compute Heston model implied volatilities for a (maturity, strike) grid.

    Parameters
    ----------
    spot : float
        Current SPX level.
    strikes_by_maturity : dict
        Mapping of maturity (in years) to an array of strikes.
    params : HestonCFParams
        Heston parameters under the risk-neutral measure.
    option_type_by_maturity : dict or None
        Optional mapping of maturity to an array of "call"/"put" strings.
        If None, OTM convention is used (calls for K >= forward, puts for
        K < forward) to keep IV inversion away from deep ITM noise.

    Returns
    -------
    pd.DataFrame
        Columns ['maturity', 'strike', 'option_type', 'price', 'iv'].
    """
    params.validate()
    rows = []

    for maturity, strikes in strikes_by_maturity.items():
        forward = spot * exp((params.rate - params.dividend_yield) * maturity)

        if option_type_by_maturity is not None:
            types = option_type_by_maturity[maturity]
        else:
            types = np.where(strikes >= forward, "call", "put")

        for strike, otype in zip(strikes, types):
            price = price_european_cos(
                spot=spot,
                strike=float(strike),
                maturity=float(maturity),
                params=params,
                option_type=str(otype),
                n_terms=n_terms,
                truncation_level=truncation_level,
            )

            iv = implied_volatility(
                price=price,
                s0=spot,
                strike=float(strike),
                maturity=float(maturity),
                rate=params.rate,
                option_type=str(otype),
                dividend_yield=params.dividend_yield,
            )

            rows.append(
                {
                    "maturity": float(maturity),
                    "strike": float(strike),
                    "option_type": str(otype),
                    "price": price,
                    "iv": iv,
                }
            )

    return pd.DataFrame(rows)


def check_put_call_parity(
    spot: float,
    strike: float,
    maturity: float,
    params: HestonCFParams,
    n_terms: int = 160,
) -> dict[str, float]:
    """
    Compute call - put under Heston COS and compare with the parity value
    S0 * exp(-q T) - K * exp(-r T).
    """
    call = price_european_cos(
        spot=spot,
        strike=strike,
        maturity=maturity,
        params=params,
        option_type="call",
        n_terms=n_terms,
    )
    put = price_european_cos(
        spot=spot,
        strike=strike,
        maturity=maturity,
        params=params,
        option_type="put",
        n_terms=n_terms,
    )

    parity_lhs = call - put
    parity_rhs = spot * exp(-params.dividend_yield * maturity) - strike * exp(
        -params.rate * maturity
    )

    return {
        "call": call,
        "put": put,
        "parity_lhs": parity_lhs,
        "parity_rhs": parity_rhs,
        "absolute_error": abs(parity_lhs - parity_rhs),
    }
