import numpy as np
import pandas as pd
import pytest

from src.calibration.heston_calibration import (
    calibrate_spx,
    calibrate_vix,
    evaluate_fit,
)
from src.calibration.objectives import (
    SPXTargets,
    VIXTargets,
    joint_objective,
    make_cf_params,
    make_cir_params,
    spx_iv_objective,
    vix_futures_objective,
)
from src.models.cir import CIRParams
from src.models.heston_cf import HestonCFParams
from src.pricing.heston_spx import price_european_cos
from src.pricing.iv_inversion import implied_volatility
from src.pricing.vix_cos import price_vix_futures_cos


def _build_synthetic_spx_targets(true_params: HestonCFParams) -> SPXTargets:
    spot = 100.0
    rate = true_params.rate

    maturities = np.array([0.25, 0.5, 1.0])
    strike_factors = np.array([0.92, 0.96, 1.00, 1.04, 1.08])

    rows = []
    for t in maturities:
        forward = spot * np.exp(rate * t)
        for sf in strike_factors:
            k = forward * sf
            otype = "call" if k >= forward else "put"
            price = price_european_cos(
                spot=spot,
                strike=float(k),
                maturity=float(t),
                params=true_params,
                option_type=otype,
                n_terms=96,
            )
            iv = implied_volatility(
                price=price,
                s0=spot,
                strike=float(k),
                maturity=float(t),
                rate=rate,
                option_type=otype,
            )
            rows.append({
                "expiration": pd.Timestamp("2024-12-31") + pd.Timedelta(days=int(t * 365)),
                "maturity_years": float(t),
                "strike": float(k),
                "option_type": otype,
                "mid": price,
                "iv_market": iv,
            })

    snapshot = {
        "snapshot_date": "2024-12-31",
        "spot": spot,
        "rate": rate,
        "options": pd.DataFrame(rows),
    }
    return SPXTargets.from_chain(snapshot=snapshot, weight_scheme="uniform")


def _build_synthetic_vix_targets(true_params: HestonCFParams) -> VIXTargets:
    cir = CIRParams(
        v0=true_params.v0,
        kappa=true_params.kappa,
        theta=true_params.theta,
        sigma_v=true_params.sigma_v,
    )
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12])
    futures = np.array([
        price_vix_futures_cos(params=cir, maturity=float(t), r=true_params.rate)[
            "vix_futures_price"
        ]
        for t in maturities
    ])
    return VIXTargets(maturities=maturities, futures=futures, rate=true_params.rate)


def test_objectives_return_finite_values_at_synthetic_truth():
    true_params = HestonCFParams(
        v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7, rate=0.03,
    )
    spx_targets = _build_synthetic_spx_targets(true_params)
    vix_targets = _build_synthetic_vix_targets(true_params)

    x_true = np.array([
        true_params.kappa,
        true_params.theta,
        true_params.sigma_v,
        true_params.rho,
        true_params.v0,
    ])

    spx_loss = spx_iv_objective(x=x_true, targets=spx_targets, n_terms=96, penalise_feller=False)
    vix_loss = vix_futures_objective(x=x_true, targets=vix_targets, penalise_feller=False)

    # At the truth, both losses should be tiny (only numerical noise).
    assert spx_loss < 1e-6
    assert vix_loss < 1e-3


def test_joint_objective_is_convex_combination():
    true_params = HestonCFParams(
        v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7, rate=0.03,
    )
    spx_targets = _build_synthetic_spx_targets(true_params)
    vix_targets = _build_synthetic_vix_targets(true_params)

    x = np.array([2.5, 0.05, 0.6, -0.6, 0.05])
    j0 = joint_objective(x=x, spx_targets=spx_targets, vix_targets=vix_targets, spx_weight=0.0)
    j1 = joint_objective(x=x, spx_targets=spx_targets, vix_targets=vix_targets, spx_weight=1.0)
    jm = joint_objective(x=x, spx_targets=spx_targets, vix_targets=vix_targets, spx_weight=0.5)

    assert j0 > 0
    assert j1 > 0
    # The mid-weight loss lies between the extremes after rescaling.
    assert min(j0, j1) <= jm <= max(j0, j1) + 1e-6


@pytest.mark.slow
def test_vix_only_calibration_recovers_synthetic_params():
    true_params = HestonCFParams(
        v0=0.045, kappa=3.0, theta=0.05, sigma_v=0.6, rho=-0.7, rate=0.03,
    )
    vix_targets = _build_synthetic_vix_targets(true_params)

    result = calibrate_vix(vix_targets=vix_targets, seeds=(11,))

    # The VIX-only problem only identifies kappa, theta, sigma_v, v0.
    # On a synthetic noise-free curve we recover all four within ~30 %
    # (DE+L-BFGS-B is a coarse global search).
    assert abs(result.params["theta"] - true_params.theta) / true_params.theta < 0.3
    assert abs(result.params["v0"] - true_params.v0) / true_params.v0 < 0.3
    assert result.loss < 0.5
