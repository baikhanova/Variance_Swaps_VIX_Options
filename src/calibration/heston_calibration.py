"""
Calibration drivers for the W6 Heston joint study.

Three drivers are provided:

    calibrate_spx   — minimise the SPX IV objective only
    calibrate_vix   — minimise the VIX futures objective only
    calibrate_joint — minimise a weighted sum of the two

All three run a coarse `differential_evolution` global search and refine
the best point with `L-BFGS-B`. Multi-start is implemented by varying the
DE seed across a small list. The function returns the calibrated
parameter vector, the in-sample loss, and a small audit dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize

from src.calibration.objectives import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    SPXTargets,
    VIXTargets,
    joint_objective,
    spx_iv_objective,
    unpack,
    vix_futures_objective,
)


@dataclass
class CalibrationResult:
    """
    Output of a single calibration run.
    """
    label: str
    params: dict[str, float]
    loss: float
    convergence: bool
    de_iterations: int
    refinement_message: str
    x: np.ndarray = field(repr=False)


def _run_de_lbfgs(
    objective: Callable[[np.ndarray], float],
    bounds: tuple[tuple[float, float], ...] = PARAM_BOUNDS,
    seeds: tuple[int, ...] = (11, 23, 47),
    de_popsize: int = 18,
    de_maxiter: int = 40,
    de_tol: float = 5e-3,
) -> tuple[np.ndarray, float, dict]:
    """
    Multi-start DE plus L-BFGS-B refinement.

    Each seed seeds the differential evolution population, the best
    DE point is refined by L-BFGS-B, and the cheapest (lowest-loss)
    refined point across the three seeds is returned.
    """
    best_x = None
    best_loss = np.inf
    audit = {"seeds": [], "iterations": 0, "refinement_message": ""}

    for seed in seeds:
        de_result = differential_evolution(
            func=objective,
            bounds=list(bounds),
            popsize=de_popsize,
            maxiter=de_maxiter,
            tol=de_tol,
            seed=seed,
            polish=False,
            updating="deferred",
            workers=1,
        )

        refine = minimize(
            fun=objective,
            x0=de_result.x,
            method="L-BFGS-B",
            bounds=list(bounds),
            options={"maxiter": 80, "ftol": 1e-9},
        )

        audit["seeds"].append({
            "seed": seed,
            "de_loss": float(de_result.fun),
            "refined_loss": float(refine.fun),
            "iterations": int(de_result.nit),
        })

        if refine.fun < best_loss:
            best_loss = float(refine.fun)
            best_x = refine.x.copy()
            audit["iterations"] = int(de_result.nit)
            audit["refinement_message"] = str(refine.message)

    return best_x, best_loss, audit


def calibrate_spx(
    spx_targets: SPXTargets,
    n_terms: int = 96,
    seeds: tuple[int, ...] = (11, 23, 47),
) -> CalibrationResult:
    """
    Calibrate Heston to the SPX IV surface only.
    """
    def objective(x: np.ndarray) -> float:
        return spx_iv_objective(x=x, targets=spx_targets, n_terms=n_terms)

    x_star, loss, audit = _run_de_lbfgs(objective=objective, seeds=seeds)
    return CalibrationResult(
        label="SPX-only",
        params=unpack(x_star),
        loss=loss,
        convergence=np.isfinite(loss),
        de_iterations=audit["iterations"],
        refinement_message=audit["refinement_message"],
        x=x_star,
    )


def calibrate_vix(
    vix_targets: VIXTargets,
    seeds: tuple[int, ...] = (11, 23, 47),
) -> CalibrationResult:
    """
    Calibrate Heston (CIR variance leg) to the VIX futures curve only.

    Only kappa, theta, sigma_v, v0 enter the VIX-side problem. The
    correlation rho cannot be identified from VIX futures alone, so it
    is fixed at its lower-bound value -0.7 in the returned parameter
    dict (any value of rho gives the same VIX futures price).
    """
    def objective(x: np.ndarray) -> float:
        return vix_futures_objective(x=x, targets=vix_targets)

    x_star, loss, audit = _run_de_lbfgs(objective=objective, seeds=seeds)
    p = unpack(x_star)
    p["rho"] = -0.7  # not identified from VIX futures
    return CalibrationResult(
        label="VIX-only",
        params=p,
        loss=loss,
        convergence=np.isfinite(loss),
        de_iterations=audit["iterations"],
        refinement_message=audit["refinement_message"],
        x=x_star,
    )


def calibrate_joint(
    spx_targets: SPXTargets,
    vix_targets: VIXTargets,
    spx_weight: float = 0.5,
    n_terms: int = 96,
    seeds: tuple[int, ...] = (11, 23, 47),
) -> CalibrationResult:
    """
    Joint SPX + VIX calibration with a tunable mixing weight.
    """
    def objective(x: np.ndarray) -> float:
        return joint_objective(
            x=x,
            spx_targets=spx_targets,
            vix_targets=vix_targets,
            spx_weight=spx_weight,
            n_terms=n_terms,
        )

    x_star, loss, audit = _run_de_lbfgs(objective=objective, seeds=seeds)
    return CalibrationResult(
        label=f"Joint (alpha={spx_weight:.2f})",
        params=unpack(x_star),
        loss=loss,
        convergence=np.isfinite(loss),
        de_iterations=audit["iterations"],
        refinement_message=audit["refinement_message"],
        x=x_star,
    )


def evaluate_fit(
    result: CalibrationResult,
    spx_targets: SPXTargets,
    vix_targets: VIXTargets,
    n_terms: int = 96,
) -> dict[str, float]:
    """
    Recompute SPX IV RMSE and VIX futures RMSE at the calibrated x.

    These per-market diagnostics are what the W6 results table reports,
    independently of which loss the calibration actually minimised.
    """
    spx_loss = spx_iv_objective(
        x=result.x,
        targets=spx_targets,
        n_terms=n_terms,
        penalise_feller=False,
    )
    vix_loss = vix_futures_objective(
        x=result.x,
        targets=vix_targets,
        penalise_feller=False,
    )
    return {
        "spx_iv_rmse": float(np.sqrt(spx_loss)),
        "vix_futures_rmse": float(np.sqrt(vix_loss)),
    }
