import numpy as np

from src.models.heston import HestonParams, simulate_heston_paths
from src.pricing.variance_mc import (
    compute_realized_variance,
    heston_variance_strike,
    price_variance_swap_mc,
    price_vix_call_mc,
    vix_squared_from_variance,
)


def test_heston_variance_strike_equals_theta_when_v0_equals_theta():
    params = HestonParams(v0=0.04, theta=0.04)

    k_var = heston_variance_strike(params=params, maturity=1.0)

    assert np.isclose(k_var, params.theta)


def test_vix_squared_equals_theta_when_variance_equals_theta():
    params = HestonParams(theta=0.04)

    vix_squared = vix_squared_from_variance(
        variance=params.theta,
        params=params,
        delta=30 / 365,
    )

    assert np.isclose(vix_squared, params.theta)


def test_realized_variance_output_shape():
    params = HestonParams()

    _, stock_paths, _ = simulate_heston_paths(
        params=params,
        maturity=1.0,
        steps=252,
        paths=300,
        seed=42,
    )

    realized_variance = compute_realized_variance(
        stock_paths=stock_paths,
        maturity=1.0,
    )

    assert realized_variance.shape == (300,)
    assert np.all(realized_variance >= 0)


def test_variance_swap_mc_returns_expected_keys():
    params = HestonParams()

    result = price_variance_swap_mc(
        params=params,
        maturity=1.0,
        steps=252,
        paths=1000,
        seed=42,
    )

    expected_keys = {
        "mc_realized_variance",
        "ci_lower",
        "ci_upper",
        "analytical_k_var",
        "used_strike",
        "absolute_error",
        "relative_error",
        "swap_value",
    }

    assert expected_keys.issubset(result.keys())


def test_variance_swap_mc_is_reasonably_close_to_analytical_strike():
    params = HestonParams(v0=0.04, theta=0.04)

    result = price_variance_swap_mc(
        params=params,
        maturity=1.0,
        steps=252,
        paths=5000,
        seed=42,
    )

    assert result["absolute_error"] < 0.02


def test_vix_call_price_is_non_negative():
    params = HestonParams()

    result = price_vix_call_mc(
        params=params,
        option_maturity=30 / 365,
        strike=20.0,
        steps=30,
        paths=1000,
        seed=42,
    )

    assert result["vix_call_price"] >= 0
    assert result["mean_terminal_vix"] >= 0