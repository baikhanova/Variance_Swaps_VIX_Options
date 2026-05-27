import numpy as np
import pytest

from src.models.heston_cf import (
    HestonCFParams,
    cos_truncation_interval,
    heston_cumulants,
    heston_log_stock_cf,
)


def test_params_validation_rejects_negative_v0():
    params = HestonCFParams(v0=-0.01)
    with pytest.raises(ValueError):
        params.validate()


def test_params_validation_rejects_bad_rho():
    params = HestonCFParams(rho=-1.5)
    with pytest.raises(ValueError):
        params.validate()


def test_cf_at_u_zero_returns_one_in_modulus():
    params = HestonCFParams()
    value = heston_log_stock_cf(u=0.0, maturity=0.5, spot=100.0, params=params)
    # phi(0) = exp(0) = 1
    assert np.isclose(value, 1.0 + 0.0j, atol=1e-10)


def test_cf_at_zero_maturity_is_exp_iu_log_spot():
    params = HestonCFParams()
    u = 0.7
    spot = 100.0
    value = heston_log_stock_cf(u=u, maturity=0.0, spot=spot, params=params)
    expected = np.exp(1j * u * np.log(spot))
    assert np.isclose(value, expected, atol=1e-12)


def test_cf_vector_input_returns_vector():
    params = HestonCFParams()
    u = np.linspace(-5, 5, 11)
    values = heston_log_stock_cf(u=u, maturity=1.0, spot=100.0, params=params)
    assert values.shape == (11,)
    # phi(-u) = conjugate(phi(u)) for a real random variable
    assert np.allclose(values, np.conjugate(values[::-1]), atol=1e-10)


def test_cf_matches_black_scholes_when_vol_of_vol_small():
    # In the limit sigma_v -> 0 and v_t == constant = v0 = theta,
    # the Heston CF should approach a Black-Scholes CF with sigma = sqrt(v0).
    v0 = 0.04
    params = HestonCFParams(
        v0=v0,
        kappa=2.0,
        theta=v0,
        sigma_v=0.001,
        rho=0.0,
        rate=0.0,
        dividend_yield=0.0,
    )
    maturity = 0.5
    spot = 100.0
    u = 1.3

    bs_cf = np.exp(
        1j * u * np.log(spot)
        - 0.5 * v0 * (u**2 + 1j * u) * maturity
    )

    heston = heston_log_stock_cf(u=u, maturity=maturity, spot=spot, params=params)
    assert abs(heston - bs_cf) < 1e-4


def test_heston_cumulants_have_expected_signs():
    params = HestonCFParams(rate=0.03, dividend_yield=0.0)
    c1, c2, _ = heston_cumulants(maturity=1.0, params=params)

    # c2 must be strictly positive (it is a variance).
    assert c2 > 0
    # c1 has no fixed sign in general, but is finite.
    assert np.isfinite(c1)


def test_cos_truncation_interval_contains_log_spot():
    params = HestonCFParams()
    a, b = cos_truncation_interval(maturity=0.5, spot=100.0, params=params)
    assert a < np.log(100.0) < b
