import numpy as np
import pytest

from src.pricing.iv_inversion import (
    bs_call_price,
    bs_put_price,
    bs_vega,
    implied_volatility,
    implied_volatility_vector,
)


def test_bs_call_at_zero_volatility_is_intrinsic_forward():
    s0 = 100.0
    strike = 90.0
    maturity = 0.5
    rate = 0.02

    price = bs_call_price(s0=s0, strike=strike, maturity=maturity, rate=rate, sigma=0.0)
    forward = s0 - strike * np.exp(-rate * maturity)

    assert np.isclose(price, max(forward, 0.0))


def test_bs_put_call_parity_holds():
    s0, strike, maturity, rate, sigma = 100.0, 105.0, 0.75, 0.03, 0.25

    call = bs_call_price(s0, strike, maturity, rate, sigma)
    put = bs_put_price(s0, strike, maturity, rate, sigma)

    parity_gap = call - put - (s0 - strike * np.exp(-rate * maturity))
    assert abs(parity_gap) < 1e-10


def test_implied_volatility_round_trip_for_call():
    s0, strike, maturity, rate, sigma = 100.0, 100.0, 1.0, 0.025, 0.22

    price = bs_call_price(s0, strike, maturity, rate, sigma)
    recovered = implied_volatility(
        price=price,
        s0=s0,
        strike=strike,
        maturity=maturity,
        rate=rate,
        option_type="call",
    )

    assert abs(recovered - sigma) < 1e-7


def test_implied_volatility_round_trip_for_put():
    s0, strike, maturity, rate, sigma = 100.0, 110.0, 0.5, 0.02, 0.35

    price = bs_put_price(s0, strike, maturity, rate, sigma)
    recovered = implied_volatility(
        price=price,
        s0=s0,
        strike=strike,
        maturity=maturity,
        rate=rate,
        option_type="put",
    )

    assert abs(recovered - sigma) < 1e-7


def test_implied_volatility_returns_nan_for_arbitrage_violation():
    iv = implied_volatility(
        price=1e6,
        s0=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.02,
        option_type="call",
    )

    assert np.isnan(iv)


def test_vega_is_positive_at_the_money():
    vega = bs_vega(s0=100.0, strike=100.0, maturity=1.0, rate=0.02, sigma=0.2)
    assert vega > 0


def test_implied_volatility_vector_handles_mixed_types():
    s0, rate = 100.0, 0.02
    strikes = np.array([95.0, 100.0, 105.0])
    maturities = np.array([0.5, 0.5, 0.5])
    sigmas = np.array([0.30, 0.25, 0.20])
    types = np.array(["call", "call", "put"])

    prices = np.array([
        bs_call_price(s0, 95.0, 0.5, rate, 0.30),
        bs_call_price(s0, 100.0, 0.5, rate, 0.25),
        bs_put_price(s0, 105.0, 0.5, rate, 0.20),
    ])

    recovered = implied_volatility_vector(
        prices=prices,
        s0=s0,
        strikes=strikes,
        maturities=maturities,
        rate=rate,
        option_types=types,
    )

    assert np.allclose(recovered, sigmas, atol=1e-6)
