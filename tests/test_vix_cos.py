import numpy as np
import pytest

from src.models.cir import (
    CIRParams,
    cir_terminal_moments,
    cir_terminal_variance_cf,
    cir_truncation_interval,
    vix_affine_coefficients,
    vix_level_from_variance,
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