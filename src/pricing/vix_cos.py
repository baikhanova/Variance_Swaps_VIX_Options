from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from src.models.cir import (
    CIRParams,
    cir_terminal_variance_cf,
    cir_truncation_interval,
    vix_level_from_variance,
)


@dataclass(frozen=True)
class COSSettings:
    """
    Numerical settings for the COS pricing prototype.

    The payoff coefficients are computed on a grid. This keeps the first
    implementation readable and easier to check before moving to closed-form
    payoff coefficients.
    """

    n_terms: int = 128
    truncation_std_width: float = 8.0
    coefficient_grid_size: int = 4096

    def validate(self) -> None:
        if self.n_terms <= 0:
            raise ValueError("n_terms must be positive.")
        if self.truncation_std_width <= 0:
            raise ValueError("truncation_std_width must be positive.")
        if self.coefficient_grid_size < 100:
            raise ValueError("coefficient_grid_size is too small.")


def _trapezoid_integral(values: np.ndarray, grid: np.ndarray) -> float:
    """Use trapezoidal integration with NumPy version compatibility."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, grid))

    return float(np.trapz(values, grid))


def _cos_payoff_coefficients(
    variance_grid: np.ndarray,
    payoff_values: np.ndarray,
    lower: float,
    upper: float,
    n_terms: int,
) -> np.ndarray:
    """
    Compute COS payoff coefficients by numerical integration.

    Coefficient k is:

        2 / (b - a) * integral_a^b payoff(x) cos(k*pi*(x-a)/(b-a)) dx

    The first term is later multiplied by 0.5 in the COS summation.
    """
    width = upper - lower
    coefficients = np.empty(n_terms)

    for k in range(n_terms):
        angle = k * np.pi * (variance_grid - lower) / width
        integrand = payoff_values * np.cos(angle)

        coefficients[k] = 2.0 / width * _trapezoid_integral(
            values=integrand,
            grid=variance_grid,
        )

    return coefficients


def _cos_expected_payoff(
    params: CIRParams,
    maturity: float,
    payoff_values: np.ndarray,
    variance_grid: np.ndarray,
    lower: float,
    upper: float,
    n_terms: int,
) -> float:
    """
    Approximate expected payoff using the COS expansion.

    The distribution input comes from the characteristic function of terminal
    CIR variance.
    """
    width = upper - lower

    coefficients = _cos_payoff_coefficients(
        variance_grid=variance_grid,
        payoff_values=payoff_values,
        lower=lower,
        upper=upper,
        n_terms=n_terms,
    )

    k = np.arange(n_terms)
    u = k * np.pi / width

    cf_values = cir_terminal_variance_cf(
        u=u,
        params=params,
        maturity=maturity,
    )

    phase = np.exp(-1j * u * lower)
    terms = np.real(cf_values * phase) * coefficients

    terms[0] *= 0.5

    expected_payoff = float(np.sum(terms))

    return max(expected_payoff, 0.0)


def price_vix_call_cos(
    params: CIRParams,
    option_maturity: float,
    strike: float,
    r: float = 0.03,
    delta: float = 30 / 365,
    settings: COSSettings | None = None,
) -> dict[str, float]:
    """
    Price a VIX call option using a first COS prototype.

    The terminal variance v_T is mapped into VIX_T using:

        VIX_T = 100 * sqrt(A + B * v_T)

    The payoff is:

        max(VIX_T - K, 0)
    """
    params.validate()

    if option_maturity <= 0:
        raise ValueError("option_maturity must be positive.")
    if strike < 0:
        raise ValueError("strike must be non-negative.")

    if settings is None:
        settings = COSSettings()

    settings.validate()

    lower, upper = cir_truncation_interval(
        params=params,
        maturity=option_maturity,
        std_width=settings.truncation_std_width,
    )

    variance_grid = np.linspace(
        lower,
        upper,
        settings.coefficient_grid_size,
    )

    terminal_vix = vix_level_from_variance(
        variance=variance_grid,
        params=params,
        delta=delta,
    )

    payoff_values = np.maximum(terminal_vix - strike, 0.0)

    expected_payoff = _cos_expected_payoff(
        params=params,
        maturity=option_maturity,
        payoff_values=payoff_values,
        variance_grid=variance_grid,
        lower=lower,
        upper=upper,
        n_terms=settings.n_terms,
    )

    discount_factor = exp(-r * option_maturity)
    price = discount_factor * expected_payoff

    return {
        "vix_call_cos_price": price,
        "expected_payoff": expected_payoff,
        "strike": strike,
        "lower_truncation": lower,
        "upper_truncation": upper,
        "n_terms": settings.n_terms,
    }


def price_vix_put_cos(
    params: CIRParams,
    option_maturity: float,
    strike: float,
    r: float = 0.03,
    delta: float = 30 / 365,
    settings: COSSettings | None = None,
) -> dict[str, float]:
    """
    Price a VIX put option using the same COS prototype.
    """
    params.validate()

    if option_maturity <= 0:
        raise ValueError("option_maturity must be positive.")
    if strike < 0:
        raise ValueError("strike must be non-negative.")

    if settings is None:
        settings = COSSettings()

    settings.validate()

    lower, upper = cir_truncation_interval(
        params=params,
        maturity=option_maturity,
        std_width=settings.truncation_std_width,
    )

    variance_grid = np.linspace(
        lower,
        upper,
        settings.coefficient_grid_size,
    )

    terminal_vix = vix_level_from_variance(
        variance=variance_grid,
        params=params,
        delta=delta,
    )

    payoff_values = np.maximum(strike - terminal_vix, 0.0)

    expected_payoff = _cos_expected_payoff(
        params=params,
        maturity=option_maturity,
        payoff_values=payoff_values,
        variance_grid=variance_grid,
        lower=lower,
        upper=upper,
        n_terms=settings.n_terms,
    )

    discount_factor = exp(-r * option_maturity)
    price = discount_factor * expected_payoff

    return {
        "vix_put_cos_price": price,
        "expected_payoff": expected_payoff,
        "strike": strike,
        "lower_truncation": lower,
        "upper_truncation": upper,
        "n_terms": settings.n_terms,
    }


def compare_vix_call_cos_with_mc(
    cos_price: float,
    mc_price: float,
) -> dict[str, float]:
    """
    Small helper for comparing the W4 COS price with the W3 Monte Carlo price.
    """
    absolute_difference = abs(cos_price - mc_price)

    relative_difference = (
        absolute_difference / mc_price
        if mc_price != 0
        else np.nan
    )

    return {
        "cos_price": cos_price,
        "mc_price": mc_price,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
    }