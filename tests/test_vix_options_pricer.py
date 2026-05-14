"""
Tests for vix_options_pricer.py  (W4 Person 2)
"""

import warnings

import numpy as np
import pytest

from src.models.cir import CIRParams
from src.pricing.vix_cos import COSSettings, price_vix_futures_cos
from src.pricing.vix_options_pricer import (
    VIXOptionSurface,
    build_vix_option_surface,
    check_put_call_parity,
    price_vix_call_vectorised,
    price_vix_put_vectorised,
)

# ── shared fixtures ──────────────────────────────────────────────────────────

BASE_PARAMS = CIRParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5)
FAST_SETTINGS = COSSettings(n_terms=64, coefficient_grid_size=512)
GOOD_SETTINGS = COSSettings(n_terms=128, coefficient_grid_size=2048)

STRIKES_ATM = np.array([20.0])
STRIKES_RANGE = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
MATURITY_1M = 30 / 365
MATURITY_3M = 90 / 365


# ── call pricer: basic sanity ────────────────────────────────────────────────


def test_call_price_non_negative_scalar():
    result = price_vix_call_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_ATM,
        r=0.03,
        settings=FAST_SETTINGS,
    )
    assert np.all(result["call_prices"] >= 0)


def test_call_price_non_negative_vector():
    result = price_vix_call_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_RANGE,
        r=0.03,
        settings=FAST_SETTINGS,
    )
    assert result["call_prices"].shape == (len(STRIKES_RANGE),)
    assert np.all(result["call_prices"] >= 0)


def test_call_price_decreasing_in_strike():
    """Call prices must be monotonically non-increasing in strike."""
    result = price_vix_call_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_RANGE,
        r=0.03,
        settings=FAST_SETTINGS,
    )
    prices = result["call_prices"]
    assert np.all(np.diff(prices) <= 1e-6), (
        f"Call prices not monotone: {prices}"
    )


def test_call_price_itm_greater_than_otm():
    """Deep ITM call (K=10) > ATM call (K=20) > deep OTM call (K=35)."""
    result = price_vix_call_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=np.array([10.0, 20.0, 35.0]),
        r=0.03,
        settings=FAST_SETTINGS,
    )
    p = result["call_prices"]
    assert p[0] > p[1] > p[2]


def test_call_returns_expected_keys():
    result = price_vix_call_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_ATM,
        settings=FAST_SETTINGS,
    )
    for key in (
        "call_prices",
        "expected_payoffs",
        "strikes",
        "maturity",
        "discount_factor",
        "lower_truncation",
        "upper_truncation",
        "n_terms",
    ):
        assert key in result, f"Missing key: {key}"


def test_call_raises_on_negative_maturity():
    with pytest.raises(ValueError):
        price_vix_call_vectorised(
            params=BASE_PARAMS, maturity=-0.1, strikes=STRIKES_ATM
        )


def test_call_raises_on_negative_strike():
    with pytest.raises(ValueError):
        price_vix_call_vectorised(
            params=BASE_PARAMS,
            maturity=MATURITY_1M,
            strikes=np.array([-5.0, 20.0]),
        )


# ── put pricer: basic sanity ─────────────────────────────────────────────────


def test_put_price_non_negative_vector():
    result = price_vix_put_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_RANGE,
        r=0.03,
        settings=FAST_SETTINGS,
    )
    assert result["put_prices"].shape == (len(STRIKES_RANGE),)
    assert np.all(result["put_prices"] >= 0)


def test_put_price_increasing_in_strike():
    """Put prices must be monotonically non-decreasing in strike."""
    result = price_vix_put_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_RANGE,
        r=0.03,
        settings=FAST_SETTINGS,
    )
    prices = result["put_prices"]
    assert np.all(np.diff(prices) >= -1e-6), (
        f"Put prices not monotone: {prices}"
    )


def test_put_returns_expected_keys():
    result = price_vix_put_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_ATM,
        settings=FAST_SETTINGS,
    )
    for key in (
        "put_prices",
        "expected_payoffs",
        "strikes",
        "maturity",
        "discount_factor",
        "lower_truncation",
        "upper_truncation",
        "n_terms",
    ):
        assert key in result, f"Missing key: {key}"


# ── put-call parity ───────────────────────────────────────────────────────────


def test_put_call_parity_atm(recwarn):
    """At ATM, parity should hold to within 0.1 VIX points for N=128."""
    fwd = price_vix_futures_cos(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        r=0.03,
        settings=GOOD_SETTINGS,
    )["vix_futures_price"]

    call_res = price_vix_call_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_ATM,
        r=0.03,
        settings=GOOD_SETTINGS,
    )
    put_res = price_vix_put_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_ATM,
        r=0.03,
        settings=GOOD_SETTINGS,
    )

    parity = check_put_call_parity(
        call_prices=call_res["call_prices"],
        put_prices=put_res["put_prices"],
        forward_vix=fwd,
        maturity=MATURITY_1M,
        strikes=STRIKES_ATM,
        r=0.03,
        tol=0.10,
    )

    assert parity["parity_passed"], (
        f"Parity violated at K=20: residual = {parity['max_violation']:.6f}"
    )


def test_put_call_parity_full_strike_range():
    """Parity across full strike range with N=128."""
    fwd = price_vix_futures_cos(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        r=0.03,
        settings=GOOD_SETTINGS,
    )["vix_futures_price"]

    call_res = price_vix_call_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_RANGE,
        r=0.03,
        settings=GOOD_SETTINGS,
    )
    put_res = price_vix_put_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=STRIKES_RANGE,
        r=0.03,
        settings=GOOD_SETTINGS,
    )

    parity = check_put_call_parity(
        call_prices=call_res["call_prices"],
        put_prices=put_res["put_prices"],
        forward_vix=fwd,
        maturity=MATURITY_1M,
        strikes=STRIKES_RANGE,
        r=0.03,
        tol=0.15,
    )

    assert parity["parity_passed"], (
        f"Parity violated: max residual = {parity['max_violation']:.6f}"
    )


def test_parity_check_issues_warning_on_large_violation():
    """When residual > tol a UserWarning must be emitted."""
    bad_calls = np.array([100.0, 100.0])
    bad_puts = np.array([0.0, 0.0])
    strikes = np.array([20.0, 25.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_put_call_parity(
            call_prices=bad_calls,
            put_prices=bad_puts,
            forward_vix=20.0,
            maturity=MATURITY_1M,
            strikes=strikes,
            r=0.03,
            tol=0.05,
        )
    assert not result["parity_passed"]
    assert any(issubclass(warning.category, UserWarning) for warning in w)


def test_parity_residuals_shape():
    fwd = price_vix_futures_cos(
        params=BASE_PARAMS, maturity=MATURITY_1M, r=0.03, settings=FAST_SETTINGS
    )["vix_futures_price"]

    call_res = price_vix_call_vectorised(
        params=BASE_PARAMS, maturity=MATURITY_1M, strikes=STRIKES_RANGE,
        r=0.03, settings=FAST_SETTINGS,
    )
    put_res = price_vix_put_vectorised(
        params=BASE_PARAMS, maturity=MATURITY_1M, strikes=STRIKES_RANGE,
        r=0.03, settings=FAST_SETTINGS,
    )

    parity = check_put_call_parity(
        call_prices=call_res["call_prices"],
        put_prices=put_res["put_prices"],
        forward_vix=fwd,
        maturity=MATURITY_1M,
        strikes=STRIKES_RANGE,
        r=0.03,
    )

    assert parity["residuals"].shape == (len(STRIKES_RANGE),)
    assert parity["lhs"].shape == (len(STRIKES_RANGE),)
    assert parity["rhs"].shape == (len(STRIKES_RANGE),)


# ── surface builder ───────────────────────────────────────────────────────────


def test_surface_builder_output_shapes():
    maturities = np.array([MATURITY_1M, MATURITY_3M])
    strikes = np.array([15.0, 20.0, 25.0])

    surface = build_vix_option_surface(
        params=BASE_PARAMS,
        maturities=maturities,
        strikes=strikes,
        r=0.03,
        settings=FAST_SETTINGS,
    )

    assert isinstance(surface, VIXOptionSurface)
    assert surface.call_prices.shape == (2, 3)
    assert surface.put_prices.shape == (2, 3)
    assert surface.forward_vix.shape == (2,)
    assert surface.parity_residuals.shape == (2, 3)
    assert surface.parity_passed.shape == (2,)


def test_surface_call_prices_non_negative():
    maturities = np.array([MATURITY_1M, MATURITY_3M])
    strikes = STRIKES_RANGE

    surface = build_vix_option_surface(
        params=BASE_PARAMS,
        maturities=maturities,
        strikes=strikes,
        settings=FAST_SETTINGS,
    )

    assert np.all(surface.call_prices >= 0)
    assert np.all(surface.put_prices >= 0)


def test_surface_forward_vix_positive():
    maturities = np.array([MATURITY_1M, MATURITY_3M])

    surface = build_vix_option_surface(
        params=BASE_PARAMS,
        maturities=maturities,
        strikes=np.array([20.0]),
        settings=FAST_SETTINGS,
    )

    assert np.all(surface.forward_vix > 0)


def test_surface_call_decreasing_in_strike():
    maturities = np.array([MATURITY_1M])
    strikes = np.array([10.0, 15.0, 20.0, 25.0, 30.0])

    surface = build_vix_option_surface(
        params=BASE_PARAMS,
        maturities=maturities,
        strikes=strikes,
        settings=FAST_SETTINGS,
    )

    row = surface.call_prices[0]
    assert np.all(np.diff(row) <= 1e-6), f"Calls not monotone: {row}"


def test_surface_summary_table_is_string():
    surface = build_vix_option_surface(
        params=BASE_PARAMS,
        maturities=np.array([MATURITY_1M]),
        strikes=np.array([20.0, 25.0]),
        settings=FAST_SETTINGS,
    )
    table = surface.summary_table()
    assert isinstance(table, str)
    assert "Call prices" in table


# ── consistency with Person 1 scalar pricer ───────────────────────────────────


def test_vectorised_call_matches_person1_scalar_pricer():
    """
    The vectorised pricer (closed-form GL coefficients) should agree with
    Person 1's numerical-grid pricer to within 0.5 VIX points for N=128.
    The small remaining gap is the improvement from closed-form coefficients
    vs. trapezoidal integration on a 4096-point grid.
    """
    from src.pricing.vix_cos import price_vix_call_cos

    K = 20.0
    settings = GOOD_SETTINGS

    scalar_result = price_vix_call_cos(
        params=BASE_PARAMS,
        option_maturity=MATURITY_1M,
        strike=K,
        r=0.03,
        settings=settings,
    )
    vec_result = price_vix_call_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=np.array([K]),
        r=0.03,
        settings=settings,
    )

    p1 = scalar_result["vix_call_cos_price"]
    p2 = float(vec_result["call_prices"][0])

    assert abs(p1 - p2) < 0.5, (
        f"Person 1 call = {p1:.6f}, Person 2 call = {p2:.6f}, "
        f"diff = {abs(p1 - p2):.6f}"
    )


def test_vectorised_put_matches_person1_scalar_pricer():
    from src.pricing.vix_cos import price_vix_put_cos

    K = 20.0
    settings = GOOD_SETTINGS

    scalar_result = price_vix_put_cos(
        params=BASE_PARAMS,
        option_maturity=MATURITY_1M,
        strike=K,
        r=0.03,
        settings=settings,
    )
    vec_result = price_vix_put_vectorised(
        params=BASE_PARAMS,
        maturity=MATURITY_1M,
        strikes=np.array([K]),
        r=0.03,
        settings=settings,
    )

    p1 = scalar_result["vix_put_cos_price"]
    p2 = float(vec_result["put_prices"][0])

    assert abs(p1 - p2) < 0.5, (
        f"Person 1 put = {p1:.6f}, Person 2 put = {p2:.6f}, "
        f"diff = {abs(p1 - p2):.6f}"
    )
