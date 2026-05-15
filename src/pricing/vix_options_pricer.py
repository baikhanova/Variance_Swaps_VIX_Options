"""
VIX Options Pricer — W4 Person 2

Builds on Person 1's COS density engine (vix_cos.py / cir.py) to add:
- vectorised call/put pricing over a strike grid in a single COS pass
- GL-quadrature payoff coefficients (better accuracy than trapz at the same N)
- put-call parity check with configurable tolerance
- surface builder over a full (maturity, strike) grid

Public API:
    price_vix_call_vectorised
    price_vix_put_vectorised
    check_put_call_parity
    build_vix_option_surface
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import exp

import numpy as np

from src.models.cir import (
    CIRParams,
    cir_terminal_variance_cf,
    cir_truncation_interval,
    vix_affine_coefficients,
)
from src.pricing.vix_cos import COSSettings, price_vix_futures_cos

# COS payoff coefficients via Gauss-Legendre quadrature


def _cos_payoff_coefficients_gl(
    strikes: np.ndarray,
    lower: float,
    upper: float,
    n_terms: int,
    cir_a: float,
    cir_b: float,
    option_type: str,  # "call" or "put"
) -> np.ndarray:
    """
    COS payoff coefficients for VIX calls or puts, computed via
    Gauss-Legendre quadrature on the active payoff region.

    The key idea: the payoff g(v) = max(VIX(v) - K, 0) is zero below
    v*(K) for calls, and zero above v*(K) for puts, where:

        v*(K) = ((K/100)^2 - A) / B

    So we only integrate over the part of [lower, upper] where the
    payoff is nonzero. GL quadrature on that sub-interval converges
    much faster than trapz on a uniform grid of the full domain.

    Returns
    -------
    coefficients : ndarray, shape (n_terms, n_strikes)
        V_k for k = 0, ..., n_terms-1 and each strike.
    """
    n_strikes = len(strikes)
    width = upper - lower

    # variance level at which payoff switches sign for each strike
    v_star = ((strikes / 100.0) ** 2 - cir_a) / cir_b

    n_gl = max(64, n_terms)
    xi, wi = np.polynomial.legendre.leggauss(n_gl)
    k_idx = np.arange(n_terms, dtype=float)

    coefficients = np.zeros((n_terms, n_strikes))

    for s_idx in range(n_strikes):
        K = strikes[s_idx]

        if option_type == "call":
            v_lo = max(v_star[s_idx], lower)
            v_hi = upper
            if v_lo >= upper:
                continue  # call worthless across whole domain
        else:
            v_lo = lower
            v_hi = min(v_star[s_idx], upper)
            if v_hi <= lower:
                continue  # put worthless across whole domain

        # affine map from GL reference interval [-1,1] -> [v_lo, v_hi]
        half = 0.5 * (v_hi - v_lo)
        mid = 0.5 * (v_hi + v_lo)
        v_nodes = half * xi + mid

        vix_nodes = 100.0 * np.sqrt(np.maximum(cir_a + cir_b * v_nodes, 0.0))

        if option_type == "call":
            payoff_nodes = np.maximum(vix_nodes - K, 0.0)
        else:
            payoff_nodes = np.maximum(K - vix_nodes, 0.0)

        angles = k_idx[:, np.newaxis] * np.pi * (v_nodes[np.newaxis, :] - lower) / width
        cos_basis = np.cos(angles)

        integrals = half * (cos_basis * (wi * payoff_nodes)[np.newaxis, :]).sum(axis=1)
        coefficients[:, s_idx] = (2.0 / width) * integrals

    return coefficients



# Core COS summation (shared between calls and puts)


def _cos_sum_vectorised(
    payoff_coefficients: np.ndarray,
    params: CIRParams,
    maturity: float,
    lower: float,
    upper: float,
    n_terms: int,
) -> np.ndarray:
    """
    COS summation E[g(v_T)] for multiple payoffs at once.

    Takes precomputed payoff coefficients (n_terms x n_strikes) and
    the CIR characteristic function, returns the expected payoff for
    each strike in one matrix multiply.

    Returns
    -------
    expected_payoffs : (n_strikes,)
    """
    width = upper - lower
    k = np.arange(n_terms, dtype=float)
    u = k * np.pi / width

    cf_values = cir_terminal_variance_cf(u=u, params=params, maturity=maturity)
    phase = np.exp(-1j * u * lower)

    # shape: (n_terms,)
    re_cf_phase = np.real(cf_values * phase)

    # shape: (n_terms, n_strikes)
    terms = re_cf_phase[:, np.newaxis] * payoff_coefficients

    # k=0 term weighted by 0.5
    terms[0, :] *= 0.5

    expected_payoffs = terms.sum(axis=0)  # (n_strikes,)
    return np.maximum(expected_payoffs, 0.0)



# Public vectorised pricers


def _price_vix_option_vectorised(
    option_type: str,
    params: CIRParams,
    maturity: float,
    strikes: np.ndarray | list[float],
    r: float,
    delta: float,
    settings: COSSettings | None,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Shared pricing logic for calls and puts.

    Returns (option_prices, expected_payoffs, discount_factor, lower, upper).
    """
    params.validate()

    if maturity <= 0:
        raise ValueError("maturity must be positive.")

    strikes = np.asarray(strikes, dtype=float)
    if strikes.ndim == 0:
        strikes = strikes.reshape(1)
    if np.any(strikes < 0):
        raise ValueError("all strikes must be non-negative.")

    if settings is None:
        settings = COSSettings()
    settings.validate()

    cir_a, cir_b = vix_affine_coefficients(params=params, delta=delta)
    lower, upper = cir_truncation_interval(
        params=params,
        maturity=maturity,
        std_width=settings.truncation_std_width,
    )

    coefficients = _cos_payoff_coefficients_gl(
        strikes=strikes,
        lower=lower,
        upper=upper,
        n_terms=settings.n_terms,
        cir_a=cir_a,
        cir_b=cir_b,
        option_type=option_type,
    )

    expected_payoffs = _cos_sum_vectorised(
        payoff_coefficients=coefficients,
        params=params,
        maturity=maturity,
        lower=lower,
        upper=upper,
        n_terms=settings.n_terms,
    )

    discount_factor = exp(-r * maturity)
    option_prices = discount_factor * expected_payoffs

    return strikes, option_prices, expected_payoffs, discount_factor, lower, upper, settings.n_terms


def price_vix_call_vectorised(
    params: CIRParams,
    maturity: float,
    strikes: np.ndarray | list[float],
    r: float = 0.03,
    delta: float = 30 / 365,
    settings: COSSettings | None = None,
) -> dict[str, np.ndarray]:
    """
    Price VIX call options for a vector of strikes in one COS pass.

    Uses GL-quadrature payoff coefficients, which gives better accuracy
    than Person 1's trapz approach at the same N.

    Parameters
    ----------
    params : CIRParams
    maturity : float
        Option expiry in years.
    strikes : array-like
        VIX strike levels in index points (e.g. 20.0 = VIX at 20).
    r : float
        Continuously compounded risk-free rate.
    delta : float
        VIX averaging window in years (default = 30 calendar days).
    settings : COSSettings or None

    Returns
    -------
    dict with keys:
        call_prices, expected_payoffs, strikes, maturity,
        discount_factor, lower_truncation, upper_truncation, n_terms
    """
    strikes, call_prices, expected_payoffs, disc, lower, upper, n_terms = (
        _price_vix_option_vectorised("call", params, maturity, strikes, r, delta, settings)
    )
    return {
        "call_prices": call_prices,
        "expected_payoffs": expected_payoffs,
        "strikes": strikes,
        "maturity": maturity,
        "discount_factor": disc,
        "lower_truncation": lower,
        "upper_truncation": upper,
        "n_terms": n_terms,
    }


def price_vix_put_vectorised(
    params: CIRParams,
    maturity: float,
    strikes: np.ndarray | list[float],
    r: float = 0.03,
    delta: float = 30 / 365,
    settings: COSSettings | None = None,
) -> dict[str, np.ndarray]:
    """
    Price VIX put options for a vector of strikes in one COS pass.

    Parameters
    ----------
    Same as price_vix_call_vectorised.

    Returns
    -------
    dict with keys:
        put_prices, expected_payoffs, strikes, maturity,
        discount_factor, lower_truncation, upper_truncation, n_terms
    """
    strikes, put_prices, expected_payoffs, disc, lower, upper, n_terms = (
        _price_vix_option_vectorised("put", params, maturity, strikes, r, delta, settings)
    )
    return {
        "put_prices": put_prices,
        "expected_payoffs": expected_payoffs,
        "strikes": strikes,
        "maturity": maturity,
        "discount_factor": disc,
        "lower_truncation": lower,
        "upper_truncation": upper,
        "n_terms": n_terms,
    }



# Put-call parity check


def check_put_call_parity(
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    forward_vix: float,
    maturity: float,
    strikes: np.ndarray,
    r: float = 0.03,
    tol: float = 0.05,
) -> dict[str, np.ndarray | bool]:
    """
    Verify put-call parity for VIX European options:

        C(K) - P(K) = exp(-r*T) * (F_VIX - K)

    Emits a UserWarning if any strike violates parity beyond tol.

    Parameters
    ----------
    call_prices, put_prices : (n_strikes,)
    forward_vix : float
        VIX futures price at maturity T (from price_vix_futures_cos).
    maturity : float
    strikes : (n_strikes,)
    r : float
    tol : float
        Max allowed absolute residual in VIX index points.

    Returns
    -------
    dict with keys:
        lhs, rhs, residuals, max_violation, parity_passed, strikes
    """
    disc = exp(-r * maturity)
    lhs = call_prices - put_prices
    rhs = disc * (forward_vix - strikes)
    residuals = np.abs(lhs - rhs)
    max_violation = float(residuals.max())

    parity_passed = max_violation < tol

    if not parity_passed:
        warnings.warn(
            f"Put-call parity violation: max residual = {max_violation:.6f} "
            f"exceeds tolerance {tol:.6f}. "
            "Consider increasing n_terms or truncation_std_width.",
            stacklevel=2,
        )

    return {
        "lhs": lhs,
        "rhs": rhs,
        "residuals": residuals,
        "max_violation": max_violation,
        "parity_passed": parity_passed,
        "strikes": strikes,
    }



# Full surface builder


@dataclass
class VIXOptionSurface:
    """
    Container for a VIX option price surface.

    Shapes:
        maturities       : (n_maturities,)
        strikes          : (n_strikes,)
        call_prices      : (n_maturities, n_strikes)
        put_prices       : (n_maturities, n_strikes)
        forward_vix      : (n_maturities,)
        parity_residuals : (n_maturities, n_strikes)
        parity_passed    : (n_maturities,) bool
    """

    maturities: np.ndarray
    strikes: np.ndarray
    call_prices: np.ndarray
    put_prices: np.ndarray
    forward_vix: np.ndarray
    parity_residuals: np.ndarray
    parity_passed: np.ndarray

    def summary_table(self) -> str:
        """Print call prices as a maturity × strike grid."""
        header = "Call prices (rows = maturities, cols = strikes)\n"
        header += "T\\K   " + "  ".join(f"{k:>8.1f}" for k in self.strikes) + "\n"
        rows = []
        for i, t in enumerate(self.maturities):
            row = f"{t:.4f} " + "  ".join(
                f"{self.call_prices[i, j]:>8.4f}" for j in range(len(self.strikes))
            )
            rows.append(row)
        return header + "\n".join(rows)


def build_vix_option_surface(
    params: CIRParams,
    maturities: np.ndarray | list[float],
    strikes: np.ndarray | list[float],
    r: float = 0.03,
    delta: float = 30 / 365,
    settings: COSSettings | None = None,
    parity_tol: float = 0.05,
) -> VIXOptionSurface:
    """
    Build a full VIX option surface over a (maturities, strikes) grid.

    For each maturity:
      1. VIX futures price via COS (used as the forward in parity check)
      2. Vectorised call and put pricing (one COS pass each)
      3. Put-call parity check

    Parameters
    ----------
    params : CIRParams
    maturities : array-like of floats (years)
    strikes : array-like of floats (VIX index points)
    r, delta : float
    settings : COSSettings or None
    parity_tol : float

    Returns
    -------
    VIXOptionSurface
    """
    params.validate()

    maturities = np.asarray(maturities, dtype=float)
    strikes = np.asarray(strikes, dtype=float)

    if settings is None:
        settings = COSSettings()
    settings.validate()

    n_t = len(maturities)
    n_k = len(strikes)

    call_prices = np.zeros((n_t, n_k))
    put_prices = np.zeros((n_t, n_k))
    forward_vix = np.zeros(n_t)
    parity_residuals = np.zeros((n_t, n_k))
    parity_passed = np.ones(n_t, dtype=bool)

    for i, T in enumerate(maturities):
        # VIX futures = forward
        fwd_result = price_vix_futures_cos(
            params=params,
            maturity=T,
            r=r,
            delta=delta,
            settings=settings,
        )
        forward_vix[i] = fwd_result["vix_futures_price"]

        # Vectorised call and put pricing
        call_result = price_vix_call_vectorised(
            params=params,
            maturity=T,
            strikes=strikes,
            r=r,
            delta=delta,
            settings=settings,
        )
        put_result = price_vix_put_vectorised(
            params=params,
            maturity=T,
            strikes=strikes,
            r=r,
            delta=delta,
            settings=settings,
        )

        call_prices[i] = call_result["call_prices"]
        put_prices[i] = put_result["put_prices"]

        # Put-call parity
        parity = check_put_call_parity(
            call_prices=call_prices[i],
            put_prices=put_prices[i],
            forward_vix=forward_vix[i],
            maturity=T,
            strikes=strikes,
            r=r,
            tol=parity_tol,
        )
        parity_residuals[i] = parity["residuals"]
        parity_passed[i] = parity["parity_passed"]

    return VIXOptionSurface(
        maturities=maturities,
        strikes=strikes,
        call_prices=call_prices,
        put_prices=put_prices,
        forward_vix=forward_vix,
        parity_residuals=parity_residuals,
        parity_passed=parity_passed,
    )
