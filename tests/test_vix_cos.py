import numpy as np
import pytest

from src.models.cir import (
    CIRParams,
    cir_char_fn,
    cir_terminal_moments,
    cir_terminal_variance_cf,
    cir_truncation_interval,
    vix_affine_coefficients,
    vix_level_from_variance,
)

from src.pricing.vix_cos import (
    COSSettings,
    compare_vix_call_cos_with_mc,
    cos_density_recovery,
    price_vix_call_cos,
    price_vix_futures_cos,
    price_vix_put_cos,
)


def test_cir_params_validation_accepts_base_case():
    params = CIRParams()

    params.validate()


def test_cir_params_validation_rejects_negative_v0():
    params = CIRParams(v0=-0.01)

    with pytest.raises(ValueError):
        params.validate()


def test_cir_terminal_variance_cf_at_zero_is_one():
    params = CIRParams()

    value = cir_terminal_variance_cf(
        u=0.0,
        params=params,
        maturity=1.0,
    )

    assert np.isclose(value, 1.0 + 0.0j)


def test_cir_terminal_variance_cf_at_zero_for_vector_input():
    params = CIRParams()

    u_values = np.array([0.0, 0.0, 0.0])

    values = cir_terminal_variance_cf(
        u=u_values,
        params=params,
        maturity=1.0,
    )

    assert values.shape == (3,)
    assert np.allclose(values, np.ones(3, dtype=complex))


def test_vix_affine_coefficients_are_reasonable():
    params = CIRParams(theta=0.04)

    a, b = vix_affine_coefficients(
        params=params,
        delta=30 / 365,
    )

    assert a >= 0
    assert 0 < b <= 1


def test_vix_level_equals_20_when_variance_equals_theta():
    params = CIRParams(theta=0.04)

    vix_level = vix_level_from_variance(
        variance=params.theta,
        params=params,
        delta=30 / 365,
    )

    assert np.isclose(vix_level, 20.0)


def test_vix_level_vector_input():
    params = CIRParams(theta=0.04)

    variances = np.array([0.01, 0.04, 0.09])

    vix_levels = vix_level_from_variance(
        variance=variances,
        params=params,
        delta=30 / 365,
    )

    assert vix_levels.shape == variances.shape
    assert np.all(vix_levels >= 0)


def test_cir_terminal_moments_at_zero_maturity():
    params = CIRParams(v0=0.04)

    mean, variance = cir_terminal_moments(
        params=params,
        maturity=0.0,
    )

    assert np.isclose(mean, params.v0)
    assert np.isclose(variance, 0.0)


def test_cir_terminal_moments_are_non_negative():
    params = CIRParams()

    mean, variance = cir_terminal_moments(
        params=params,
        maturity=1.0,
    )

    assert mean >= 0
    assert variance >= 0


def test_cir_truncation_interval_is_valid():
    params = CIRParams()

    lower, upper = cir_truncation_interval(
        params=params,
        maturity=1.0,
        std_width=8.0,
    )

    assert lower == 0.0
    assert upper > lower


def test_cos_settings_validation_accepts_base_case():
    settings = COSSettings(
        n_terms=64,
        truncation_std_width=8.0,
        coefficient_grid_size=1024,
    )

    settings.validate()


def test_cos_settings_validation_rejects_invalid_terms():
    settings = COSSettings(n_terms=0)

    with pytest.raises(ValueError):
        settings.validate()


def test_cos_settings_validation_rejects_small_grid():
    settings = COSSettings(coefficient_grid_size=50)

    with pytest.raises(ValueError):
        settings.validate()


def test_vix_call_cos_price_is_non_negative():
    params = CIRParams()

    result = price_vix_call_cos(
        params=params,
        option_maturity=30 / 365,
        strike=20.0,
        r=0.03,
        settings=COSSettings(
            n_terms=64,
            coefficient_grid_size=1024,
        ),
    )

    assert result["vix_call_cos_price"] >= 0
    assert result["expected_payoff"] >= 0
    assert result["upper_truncation"] > result["lower_truncation"]


def test_vix_put_cos_price_is_non_negative():
    params = CIRParams()

    result = price_vix_put_cos(
        params=params,
        option_maturity=30 / 365,
        strike=20.0,
        r=0.03,
        settings=COSSettings(
            n_terms=64,
            coefficient_grid_size=1024,
        ),
    )

    assert result["vix_put_cos_price"] >= 0
    assert result["expected_payoff"] >= 0
    assert result["upper_truncation"] > result["lower_truncation"]


def test_vix_call_cos_returns_expected_keys():
    params = CIRParams()

    result = price_vix_call_cos(
        params=params,
        option_maturity=30 / 365,
        strike=20.0,
        settings=COSSettings(
            n_terms=64,
            coefficient_grid_size=1024,
        ),
    )

    expected_keys = {
        "vix_call_cos_price",
        "expected_payoff",
        "strike",
        "lower_truncation",
        "upper_truncation",
        "n_terms",
    }

    assert expected_keys.issubset(result.keys())


def test_vix_put_cos_returns_expected_keys():
    params = CIRParams()

    result = price_vix_put_cos(
        params=params,
        option_maturity=30 / 365,
        strike=20.0,
        settings=COSSettings(
            n_terms=64,
            coefficient_grid_size=1024,
        ),
    )

    expected_keys = {
        "vix_put_cos_price",
        "expected_payoff",
        "strike",
        "lower_truncation",
        "upper_truncation",
        "n_terms",
    }

    assert expected_keys.issubset(result.keys())


def test_cir_char_fn_at_zero_is_one():
    params = CIRParams()

    value = cir_char_fn(u=0.0, tau=1.0, params=params)

    assert np.isclose(value, 1.0 + 0.0j)


def test_cir_char_fn_at_zero_tau_is_one():
    params = CIRParams()

    u_values = np.array([0.5, 1.0, 2.0])
    values = cir_char_fn(u=u_values, tau=0.0, params=params)

    assert values.shape == (3,)
    assert np.allclose(values, np.ones(3, dtype=complex))


def test_cir_char_fn_modulus_less_than_one_for_real_u():
    params = CIRParams()

    u_values = np.linspace(0.1, 5.0, 10)
    values = cir_char_fn(u=u_values, tau=1.0, params=params)

    assert np.all(np.abs(values) <= 1.0 + 1e-10)


def test_cir_char_fn_vector_output_shape():
    params = CIRParams()

    u_values = np.array([0.0, 1.0, 2.0, 3.0])
    values = cir_char_fn(u=u_values, tau=0.5, params=params)

    assert values.shape == (4,)


def test_cos_density_recovery_non_negative():
    params = CIRParams()

    lower, upper = cir_truncation_interval(params=params, maturity=1.0)
    v_grid = np.linspace(lower + 1e-6, upper, 200)

    density = cos_density_recovery(
        v_grid=v_grid,
        params=params,
        maturity=1.0,
        lower=lower,
        upper=upper,
        n_terms=64,
    )

    assert density.shape == v_grid.shape
    assert np.all(density >= 0.0)


def test_cos_density_recovery_integrates_to_approximately_one():
    params = CIRParams()

    lower, upper = cir_truncation_interval(params=params, maturity=1.0)
    v_grid = np.linspace(lower, upper, 2000)

    density = cos_density_recovery(
        v_grid=v_grid,
        params=params,
        maturity=1.0,
        lower=lower,
        upper=upper,
        n_terms=128,
    )

    if hasattr(np, "trapezoid"):
        integral = float(np.trapezoid(density, v_grid))
    else:
        integral = float(np.trapz(density, v_grid))

    assert np.isclose(integral, 1.0, atol=0.05)


def test_vix_futures_price_is_positive():
    params = CIRParams()

    result = price_vix_futures_cos(
        params=params,
        maturity=30 / 365,
        r=0.03,
        settings=COSSettings(n_terms=64, coefficient_grid_size=1024),
    )

    assert result["vix_futures_price"] > 0
    assert result["expected_vix"] > 0
    assert result["upper_truncation"] > result["lower_truncation"]


def test_vix_futures_price_returns_expected_keys():
    params = CIRParams()

    result = price_vix_futures_cos(
        params=params,
        maturity=30 / 365,
        settings=COSSettings(n_terms=64, coefficient_grid_size=1024),
    )

    expected_keys = {
        "vix_futures_price",
        "expected_vix",
        "lower_truncation",
        "upper_truncation",
        "n_terms",
    }

    assert expected_keys.issubset(result.keys())


def test_vix_futures_price_near_atm_vix():
    params = CIRParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5)

    result = price_vix_futures_cos(
        params=params,
        maturity=30 / 365,
        r=0.0,
        settings=COSSettings(n_terms=128, coefficient_grid_size=4096),
    )

    assert np.isclose(result["vix_futures_price"], 20.0, atol=1.5)


def test_compare_vix_call_cos_with_mc():
    result = compare_vix_call_cos_with_mc(
        cos_price=2.0,
        mc_price=1.8,
    )

    assert result["cos_price"] == 2.0
    assert result["mc_price"] == 1.8
    assert result["absolute_difference"] == pytest.approx(0.2)
    assert result["relative_difference"] == pytest.approx(0.2 / 1.8)