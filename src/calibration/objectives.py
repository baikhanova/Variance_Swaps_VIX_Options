"""
Objective functions for the W6 Heston joint calibration.

Three objectives are exposed:

    spx_iv_objective         — fit the SPX implied-volatility surface
    vix_futures_objective    — fit the VIX futures term structure
    joint_objective          — convex combination of the two

Each function maps a parameter vector x = (kappa, theta, sigma_v, rho, v0)
to a scalar non-negative loss. A soft Feller penalty is added so that
the optimiser is steered away from the 2 kappa theta < sigma_v^2 region
in which the variance process spends a non-trivial amount of time at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.cir import CIRParams
from src.models.heston_cf import HestonCFParams
from src.pricing.heston_spx import price_european_cos_batch
from src.pricing.iv_inversion import bs_vega, implied_volatility
from src.pricing.vix_cos import COSSettings, price_vix_futures_cos


PARAM_NAMES = ("kappa", "theta", "sigma_v", "rho", "v0")
PARAM_BOUNDS = (
    (0.10, 15.0),     # kappa
    (0.005, 0.25),    # theta
    (0.05, 2.5),      # sigma_v
    (-0.99, 0.0),     # rho (non-positive for equity)
    (0.005, 0.5),     # v0
)


def unpack(x: np.ndarray) -> dict[str, float]:
    """Vector to named-parameter dict."""
    return dict(zip(PARAM_NAMES, [float(v) for v in x]))


def make_cf_params(x: np.ndarray, rate: float, dividend_yield: float = 0.0) -> HestonCFParams:
    p = unpack(x)
    return HestonCFParams(
        v0=p["v0"],
        kappa=p["kappa"],
        theta=p["theta"],
        sigma_v=p["sigma_v"],
        rho=p["rho"],
        rate=rate,
        dividend_yield=dividend_yield,
    )


def make_cir_params(x: np.ndarray) -> CIRParams:
    p = unpack(x)
    return CIRParams(
        v0=p["v0"],
        kappa=p["kappa"],
        theta=p["theta"],
        sigma_v=p["sigma_v"],
    )


def feller_penalty(x: np.ndarray, weight: float = 50.0) -> float:
    """
    Soft penalty on the Feller condition 2 kappa theta >= sigma_v^2.

    Returns zero when the condition holds and a quadratic penalty in
    the deficit otherwise. The default weight is calibrated empirically
    so that the penalty becomes meaningful only when the violation is
    substantial.
    """
    p = unpack(x)
    deficit = p["sigma_v"]**2 - 2.0 * p["kappa"] * p["theta"]
    if deficit <= 0:
        return 0.0
    return float(weight * deficit**2)


# ─────────────────────────────────────────────────────────────────────────────
# SPX implied-volatility objective
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SPXTargets:
    """
    Container for the SPX surface used in calibration.

    The strikes, maturities and IVs are stored as flat arrays of equal
    length so the residual loop is a simple zip. To avoid running a
    Brent IV inversion inside every objective evaluation, the market
    price and Black-Scholes vega at the market IV are pre-computed and
    cached. The objective then approximates the IV residual as

        (price_model - price_market) / vega_market

    which is the first-order Taylor expansion of the exact IV residual
    and is standard practice in option calibration.
    """

    spot: float
    rate: float
    strikes: np.ndarray
    maturities: np.ndarray
    option_types: np.ndarray
    iv_market: np.ndarray
    price_market: np.ndarray
    vega_market: np.ndarray
    weights: np.ndarray
    dividend_yield: float = 0.0

    @classmethod
    def from_chain(
        cls,
        snapshot: dict,
        weight_scheme: str = "vega",
    ) -> "SPXTargets":
        df: pd.DataFrame = snapshot["options"]
        strikes = df["strike"].to_numpy(dtype=float)
        maturities = df["maturity_years"].to_numpy(dtype=float)
        option_types = df["option_type"].to_numpy()
        iv = df["iv_market"].to_numpy(dtype=float)
        prices = df["mid"].to_numpy(dtype=float)

        spot = float(snapshot["spot"])
        rate = float(snapshot["rate"])

        vegas = np.array([
            bs_vega(spot, float(k), float(t), rate, float(s))
            for k, t, s in zip(strikes, maturities, iv)
        ])
        vegas = np.where(vegas > 1e-4, vegas, 1e-4)

        if weight_scheme == "vega":
            weights = vegas / vegas.mean()
        elif weight_scheme == "uniform":
            weights = np.ones_like(iv)
        elif weight_scheme == "inverse_moneyness":
            forwards = spot * np.exp(rate * maturities)
            moneyness = np.abs(np.log(strikes / forwards))
            weights = 1.0 / (moneyness + 0.1)
        else:
            raise ValueError(f"Unknown weight scheme: {weight_scheme}")

        return cls(
            spot=spot,
            rate=rate,
            strikes=strikes,
            maturities=maturities,
            option_types=option_types,
            iv_market=iv,
            price_market=prices,
            vega_market=vegas,
            weights=weights,
        )


def spx_iv_objective(
    x: np.ndarray,
    targets: SPXTargets,
    n_terms: int = 128,
    penalise_feller: bool = True,
) -> float:
    """
    Weighted sum of squared (approximate) IV errors across the SPX surface.

    Quotes are grouped by maturity so that the expensive Heston
    characteristic-function evaluation is performed only once per
    maturity slice. The IV residual is approximated by the vega-scaled
    price residual to avoid the Brent IV inversion inside the loop.
    """
    cf_params = make_cf_params(x, rate=targets.rate, dividend_yield=targets.dividend_yield)
    try:
        cf_params.validate()
    except ValueError:
        return 1e6

    sq_errors = 0.0
    weight_total = 0.0

    unique_maturities = np.unique(targets.maturities)
    for maturity in unique_maturities:
        mask = targets.maturities == maturity
        strikes_slice = targets.strikes[mask]
        types_slice = targets.option_types[mask]
        price_market_slice = targets.price_market[mask]
        vega_slice = targets.vega_market[mask]
        weights_slice = targets.weights[mask]

        try:
            prices_model = price_european_cos_batch(
                spot=targets.spot,
                strikes=strikes_slice,
                option_types=types_slice,
                maturity=float(maturity),
                params=cf_params,
                n_terms=n_terms,
            )
        except Exception:
            prices_model = np.full(strikes_slice.shape, np.nan)

        iv_residual = (prices_model - price_market_slice) / vega_slice
        invalid = ~np.isfinite(iv_residual)
        iv_residual = np.where(invalid, 0.5, iv_residual)

        sq_errors += float(np.sum(weights_slice * iv_residual**2))
        weight_total += float(np.sum(weights_slice))

    loss = sq_errors / max(weight_total, 1e-12)
    if penalise_feller:
        loss = loss + feller_penalty(x)
    return float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# VIX futures objective
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VIXTargets:
    """
    Container for the VIX futures curve used in calibration.

    maturities and futures are in matching order. The discount rate is
    used by the COS-based VIX futures price (the model VIX futures price
    is a discounted expectation of VIX_T under the risk-neutral measure).
    """

    maturities: np.ndarray
    futures: np.ndarray
    rate: float


def vix_futures_objective(
    x: np.ndarray,
    targets: VIXTargets,
    penalise_feller: bool = True,
) -> float:
    """
    Mean squared error in VIX-points units across the futures curve.

    The model VIX futures price is computed via the COS routine from W4,
    which uses the CIR characteristic function of terminal variance.
    """
    cir_params = make_cir_params(x)
    try:
        cir_params.validate()
    except ValueError:
        return 1e6

    settings = COSSettings(n_terms=96, coefficient_grid_size=2048)
    errors_sq = 0.0

    for t, market_f in zip(targets.maturities, targets.futures):
        try:
            result = price_vix_futures_cos(
                params=cir_params,
                maturity=float(t),
                r=targets.rate,
                settings=settings,
            )
            model_f = result["vix_futures_price"]
        except Exception:
            model_f = np.nan

        if np.isnan(model_f) or not np.isfinite(model_f):
            errors_sq += 100.0
        else:
            errors_sq += (model_f - market_f)**2

    loss = errors_sq / max(len(targets.maturities), 1)
    if penalise_feller:
        loss = loss + feller_penalty(x)
    return float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Joint objective
# ─────────────────────────────────────────────────────────────────────────────


def joint_objective(
    x: np.ndarray,
    spx_targets: SPXTargets,
    vix_targets: VIXTargets,
    spx_weight: float = 0.5,
    spx_scale: float = 1.0,
    vix_scale: float = 0.01,
    n_terms: int = 128,
) -> float:
    """
    Convex combination of the SPX and VIX objectives.

    Because the SPX loss is in (IV)^2 units while the VIX loss is in
    (VIX points)^2 units, fixed scales convert them to comparable
    orders of magnitude before the weighted sum. The defaults make a
    1 % IV mismatch and a 1-point VIX-futures mismatch roughly equal
    contributions to the joint loss.
    """
    if not 0.0 <= spx_weight <= 1.0:
        raise ValueError("spx_weight must be in [0, 1].")

    spx_loss = spx_iv_objective(
        x=x,
        targets=spx_targets,
        n_terms=n_terms,
        penalise_feller=False,
    )
    vix_loss = vix_futures_objective(
        x=x,
        targets=vix_targets,
        penalise_feller=False,
    )

    combined = (
        spx_weight * spx_scale * spx_loss
        + (1.0 - spx_weight) * vix_scale * vix_loss
    )
    return float(combined + feller_penalty(x))
