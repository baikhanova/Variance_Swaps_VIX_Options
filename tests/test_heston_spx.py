import numpy as np
import pytest

from src.models.heston import HestonParams, simulate_heston_paths
from src.models.heston_cf import HestonCFParams
from src.pricing.heston_spx import (
    check_put_call_parity,
    model_iv_surface,
    price_european_cos,
)
from src.pricing.iv_inversion import bs_call_price, implied_volatility


def _default_params() -> HestonCFParams:
    return HestonCFParams(
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.5,
        rho=-0.7,
        rate=0.03,
        dividend_yield=0.0,
    )


def test_call_price_is_positive_and_bounded_by_spot():
    params = _default_params()
    price = price_european_cos(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        params=params,
        option_type="call",
    )
    assert 0.0 < price < 100.0


def test_put_price_is_positive_and_bounded_by_strike():
    params = _default_params()
    price = price_european_cos(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        params=params,
        option_type="put",
    )
    assert 0.0 < price < 100.0


def test_put_call_parity_holds_under_cos():
    params = _default_params()
    result = check_put_call_parity(
        spot=100.0,
        strike=95.0,
        maturity=0.75,
        params=params,
    )
    assert result["absolute_error"] < 5e-3


def test_cos_matches_black_scholes_in_low_vol_of_vol_limit():
    v0 = 0.04
    params = HestonCFParams(
        v0=v0,
        kappa=2.0,
        theta=v0,
        sigma_v=0.001,
        rho=0.0,
        rate=0.02,
        dividend_yield=0.0,
    )
    spot, strike, maturity = 100.0, 100.0, 0.5

    cos_price = price_european_cos(
        spot=spot,
        strike=strike,
        maturity=maturity,
        params=params,
    )
    bs_price = bs_call_price(
        s0=spot,
        strike=strike,
        maturity=maturity,
        rate=params.rate,
        sigma=np.sqrt(v0),
    )

    assert abs(cos_price - bs_price) < 0.05


def test_cos_price_matches_monte_carlo_within_two_percent():
    cf_params = _default_params()
    mc_params = HestonParams(
        s0=100.0,
        v0=cf_params.v0,
        r=cf_params.rate,
        kappa=cf_params.kappa,
        theta=cf_params.theta,
        sigma_v=cf_params.sigma_v,
        rho=cf_params.rho,
    )

    spot, strike, maturity = 100.0, 100.0, 1.0

    cos_price = price_european_cos(
        spot=spot,
        strike=strike,
        maturity=maturity,
        params=cf_params,
    )

    _, stock_paths, _ = simulate_heston_paths(
        params=mc_params,
        maturity=maturity,
        steps=252,
        paths=50_000,
        seed=42,
    )
    terminal = stock_paths[:, -1]
    payoff = np.maximum(terminal - strike, 0.0)
    mc_price = np.exp(-cf_params.rate * maturity) * float(np.mean(payoff))

    rel = abs(cos_price - mc_price) / mc_price
    assert rel < 0.02


def test_model_iv_surface_returns_sensible_smile():
    params = _default_params()
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    surface = model_iv_surface(
        spot=100.0,
        strikes_by_maturity={0.25: strikes, 1.0: strikes},
        params=params,
    )
    assert set(surface.columns) == {"maturity", "strike", "option_type", "price", "iv"}
    assert surface["iv"].notna().all()
    # The negative-rho Heston model produces a downward-sloping skew in K
    # at every maturity, so OTM puts have IV >= ATM IV.
    atm = surface[(surface["strike"] == 100.0) & (surface["maturity"] == 1.0)]["iv"].iloc[0]
    otm_put = surface[(surface["strike"] == 80.0) & (surface["maturity"] == 1.0)]["iv"].iloc[0]
    assert otm_put > atm
