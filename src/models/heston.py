from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    """
    Parameters used for the Heston simulation.

    The variance part follows a CIR-type process:
    dv_t = kappa(theta - v_t)dt + sigma_v sqrt(v_t)dW_t.
    """

    s0: float = 100.0
    v0: float = 0.04
    r: float = 0.03
    kappa: float = 2.0
    theta: float = 0.04
    sigma_v: float = 0.5
    rho: float = -0.7

    def validate(self) -> None:
        """Check that the parameters can be used in the simulation."""
        if self.s0 <= 0:
            raise ValueError("s0 must be positive.")
        if self.v0 < 0:
            raise ValueError("v0 must be non-negative.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.theta < 0:
            raise ValueError("theta must be non-negative.")
        if self.sigma_v < 0:
            raise ValueError("sigma_v must be non-negative.")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must be between -1 and 1.")


def simulate_heston_paths(
    params: HestonParams,
    maturity: float,
    steps: int,
    paths: int,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate stock and variance paths under the Heston model.

    The stock process is simulated under the risk-neutral measure:

        dS_t / S_t = r dt + sqrt(v_t)dW_t^S

    The variance process is:

        dv_t = kappa(theta - v_t)dt + sigma_v sqrt(v_t)dW_t^v

    The two Brownian motions have correlation rho.

    Returns
    -------
    times:
        Time grid with shape (steps + 1,).

    stock_paths:
        Simulated stock paths with shape (paths, steps + 1).

    variance_paths:
        Simulated variance paths with shape (paths, steps + 1).
    """
    params.validate()

    if maturity <= 0:
        raise ValueError("maturity must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if paths <= 0:
        raise ValueError("paths must be positive.")

    rng = np.random.default_rng(seed)

    dt = maturity / steps
    sqrt_dt = np.sqrt(dt)

    times = np.linspace(0.0, maturity, steps + 1)

    stock_paths = np.empty((paths, steps + 1))
    variance_paths = np.empty((paths, steps + 1))

    stock_paths[:, 0] = params.s0
    variance_paths[:, 0] = params.v0

    for step in range(steps):
        z_v = rng.standard_normal(paths)
        z_independent = rng.standard_normal(paths)

        z_s = params.rho * z_v + np.sqrt(1.0 - params.rho**2) * z_independent

        current_variance = np.maximum(variance_paths[:, step], 0.0)

        variance_drift = params.kappa * (params.theta - current_variance) * dt
        variance_diffusion = (
            params.sigma_v * np.sqrt(current_variance) * sqrt_dt * z_v
        )

        next_variance = variance_paths[:, step] + variance_drift + variance_diffusion
        variance_paths[:, step + 1] = np.maximum(next_variance, 0.0)

        stock_drift = (params.r - 0.5 * current_variance) * dt
        stock_diffusion = np.sqrt(current_variance) * sqrt_dt * z_s

        stock_paths[:, step + 1] = stock_paths[:, step] * np.exp(
            stock_drift + stock_diffusion
        )

    return times, stock_paths, variance_paths