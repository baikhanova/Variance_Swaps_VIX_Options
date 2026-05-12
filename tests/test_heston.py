import numpy as np
import pytest

from src.models.heston import HestonParams, simulate_heston_paths


def test_heston_simulation_shapes():
    params = HestonParams()

    times, stock_paths, variance_paths = simulate_heston_paths(
        params=params,
        maturity=1.0,
        steps=252,
        paths=500,
        seed=42,
    )

    assert times.shape == (253,)
    assert stock_paths.shape == (500, 253)
    assert variance_paths.shape == (500, 253)


def test_heston_paths_start_from_initial_values():
    params = HestonParams(s0=100.0, v0=0.04)

    _, stock_paths, variance_paths = simulate_heston_paths(
        params=params,
        maturity=1.0,
        steps=252,
        paths=500,
        seed=42,
    )

    assert np.allclose(stock_paths[:, 0], params.s0)
    assert np.allclose(variance_paths[:, 0], params.v0)


def test_heston_stock_paths_stay_positive():
    params = HestonParams()

    _, stock_paths, _ = simulate_heston_paths(
        params=params,
        maturity=1.0,
        steps=252,
        paths=500,
        seed=42,
    )

    assert np.all(stock_paths > 0)


def test_heston_variance_paths_stay_non_negative():
    params = HestonParams()

    _, _, variance_paths = simulate_heston_paths(
        params=params,
        maturity=1.0,
        steps=252,
        paths=500,
        seed=42,
    )

    assert np.all(variance_paths >= 0)


def test_heston_simulation_is_reproducible_with_same_seed():
    params = HestonParams()

    _, stock_paths_1, variance_paths_1 = simulate_heston_paths(
        params=params,
        maturity=1.0,
        steps=252,
        paths=200,
        seed=123,
    )

    _, stock_paths_2, variance_paths_2 = simulate_heston_paths(
        params=params,
        maturity=1.0,
        steps=252,
        paths=200,
        seed=123,
    )

    assert np.allclose(stock_paths_1, stock_paths_2)
    assert np.allclose(variance_paths_1, variance_paths_2)


def test_invalid_heston_parameters_raise_error():
    params = HestonParams(kappa=-1.0)

    with pytest.raises(ValueError):
        params.validate()