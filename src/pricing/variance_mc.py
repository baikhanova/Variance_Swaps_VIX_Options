from __future__ import annotations

from math import exp, sqrt

import numpy as np
import pandas as pd

from src.models.heston import HestonParams, simulate_heston_paths


def compute_realized_variance(
    stock_paths: np.ndarray,
    maturity: float,
) -> np.ndarray:
    """
    Compute annualised realised variance from simulated stock paths.

    Since maturity is measured in years, the realised variance is computed as:

        RV = sum(log_returns^2) / T

    This is the discrete version of the average integrated variance.
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive.")

    if stock_paths.ndim != 2:
        raise ValueError("stock_paths must be a 2D array.")

    if np.any(stock_paths <= 0):
        raise ValueError("stock paths must be positive to compute log returns.")

    log_returns = np.diff(np.log(stock_paths), axis=1)
    realized_variance = np.sum(log_returns**2, axis=1) / maturity

    return realized_variance


def heston_variance_strike(
    params: HestonParams,
    maturity: float,
) -> float:
    """
    Analytical fair variance strike under the Heston variance process.

    K_var = theta + (v0 - theta) * (1 - exp(-kappa*T)) / (kappa*T)
    """
    params.validate()

    if maturity <= 0:
        raise ValueError("maturity must be positive.")

    return params.theta + (params.v0 - params.theta) * (
        1.0 - exp(-params.kappa * maturity)
    ) / (params.kappa * maturity)


def vix_squared_from_variance(
    variance: np.ndarray | float,
    params: HestonParams,
    delta: float = 30 / 365,
) -> np.ndarray | float:
    """
    Convert instantaneous variance into the Heston approximation of VIX squared.

    The formula used is:

        VIX_t^2 = theta + (v_t - theta) * (1 - exp(-kappa*Delta)) / (kappa*Delta)

    The output is still in variance units, not VIX index points.
    """
    params.validate()

    if delta <= 0:
        raise ValueError("delta must be positive.")

    factor = (1.0 - exp(-params.kappa * delta)) / (params.kappa * delta)

    vix_squared = params.theta + (variance - params.theta) * factor

    return np.maximum(vix_squared, 0.0)


def monte_carlo_confidence_interval(
    samples: np.ndarray,
    confidence_z: float = 1.96,
) -> tuple[float, float, float]:
    """
    Return Monte Carlo mean and normal-approximation confidence interval.
    """
    if samples.ndim != 1:
        raise ValueError("samples must be a 1D array.")

    if len(samples) < 2:
        raise ValueError("at least two samples are required.")

    mean = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / sqrt(len(samples)))

    lower = mean - confidence_z * standard_error
    upper = mean + confidence_z * standard_error

    return mean, lower, upper


def price_variance_swap_mc(
    params: HestonParams,
    maturity: float,
    steps: int,
    paths: int,
    strike: float | None = None,
    seed: int | None = 42,
) -> dict[str, float]:
    """
    Price the floating variance leg by Monte Carlo and compare it with K_var.

    If strike is not given, the analytical Heston variance strike is used.
    The swap value is reported for the payoff:

        realised variance - strike

    A short variance swap position would use the opposite sign.
    """
    _, stock_paths, _ = simulate_heston_paths(
        params=params,
        maturity=maturity,
        steps=steps,
        paths=paths,
        seed=seed,
    )

    realized_variance = compute_realized_variance(stock_paths, maturity)
    mc_mean, ci_lower, ci_upper = monte_carlo_confidence_interval(realized_variance)

    analytical_strike = heston_variance_strike(params, maturity)
    used_strike = analytical_strike if strike is None else strike

    discount_factor = exp(-params.r * maturity)
    swap_value = discount_factor * (mc_mean - used_strike)

    return {
        "mc_realized_variance": mc_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "analytical_k_var": analytical_strike,
        "used_strike": used_strike,
        "absolute_error": abs(mc_mean - analytical_strike),
        "relative_error": abs(mc_mean - analytical_strike) / analytical_strike,
        "swap_value": swap_value,
    }


def price_vix_call_mc(
    params: HestonParams,
    option_maturity: float,
    strike: float,
    steps: int,
    paths: int,
    delta: float = 30 / 365,
    seed: int | None = 42,
) -> dict[str, float]:
    """
    Price a VIX call option by Monte Carlo.

    The simulated terminal variance is converted into a VIX level using
    the affine Heston relation. The VIX level is expressed in index points:

        VIX = 100 * sqrt(VIX^2)

    Therefore, a strike such as 20 means 20 VIX points.
    """
    if strike < 0:
        raise ValueError("strike must be non-negative.")

    _, _, variance_paths = simulate_heston_paths(
        params=params,
        maturity=option_maturity,
        steps=steps,
        paths=paths,
        seed=seed,
    )

    terminal_variance = variance_paths[:, -1]
    terminal_vix_squared = vix_squared_from_variance(
        variance=terminal_variance,
        params=params,
        delta=delta,
    )

    terminal_vix = 100.0 * np.sqrt(terminal_vix_squared)
    payoff = np.maximum(terminal_vix - strike, 0.0)

    payoff_mean, ci_lower, ci_upper = monte_carlo_confidence_interval(payoff)

    discount_factor = exp(-params.r * option_maturity)
    price = discount_factor * payoff_mean

    return {
        "vix_call_price": price,
        "undiscounted_payoff_mean": payoff_mean,
        "ci_lower_discounted": discount_factor * ci_lower,
        "ci_upper_discounted": discount_factor * ci_upper,
        "mean_terminal_vix": float(np.mean(terminal_vix)),
        "strike": strike,
    }


def run_convergence_study(
    params: HestonParams,
    maturity: float,
    steps: int,
    path_counts: list[int],
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Check how the Monte Carlo estimate changes as the number of paths increases.
    """
    analytical_strike = heston_variance_strike(params, maturity)

    rows = []

    for paths in path_counts:
        result = price_variance_swap_mc(
            params=params,
            maturity=maturity,
            steps=steps,
            paths=paths,
            seed=seed,
        )

        rows.append(
            {
                "paths": paths,
                "mc_realized_variance": result["mc_realized_variance"],
                "analytical_k_var": analytical_strike,
                "absolute_error": result["absolute_error"],
                "relative_error": result["relative_error"],
                "ci_width": result["ci_upper"] - result["ci_lower"],
            }
        )

    return pd.DataFrame(rows)