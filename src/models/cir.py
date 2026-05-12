from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIRParams:
    """
    Parameters for the CIR variance process.

    This is the variance part of the Heston model:

        dv_t = kappa(theta - v_t)dt + sigma_v sqrt(v_t)dW_t

    The same process is later used in the W4 COS pricing block.
    """

    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    sigma_v: float = 0.5

    def validate(self) -> None:
        """Check that the parameters can be used in the CIR formulas."""
        if self.v0 < 0:
            raise ValueError("v0 must be non-negative.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.theta < 0:
            raise ValueError("theta must be non-negative.")
        if self.sigma_v <= 0:
            raise ValueError("sigma_v must be positive.")


def vix_affine_coefficients(
    params: CIRParams,
    delta: float = 30 / 365,
) -> tuple[float, float]:
    """
    Return A and B in the Heston approximation:

        VIX_t^2 = A + B * v_t

    where:

        B = (1 - exp(-kappa * Delta)) / (kappa * Delta)
        A = theta * (1 - B)
    """
    params.validate()

    if delta <= 0:
        raise ValueError("delta must be positive.")

    b = (1.0 - np.exp(-params.kappa * delta)) / (params.kappa * delta)
    a = params.theta * (1.0 - b)

    return float(a), float(b)


def vix_level_from_variance(
    variance: np.ndarray | float,
    params: CIRParams,
    delta: float = 30 / 365,
) -> np.ndarray | float:
    """
    Convert variance into VIX index points.

    The input variance is first converted into VIX squared, then into:

        VIX = 100 * sqrt(VIX^2)
    """
    a, b = vix_affine_coefficients(params=params, delta=delta)

    vix_squared = a + b * np.asarray(variance)
    vix_squared = np.maximum(vix_squared, 0.0)

    vix_level = 100.0 * np.sqrt(vix_squared)

    if np.isscalar(variance):
        return float(vix_level)

    return vix_level


def cir_terminal_moments(
    params: CIRParams,
    maturity: float,
) -> tuple[float, float]:
    """
    Return mean and variance of v_T under the CIR process.

    These moments are useful for choosing the truncation interval in COS.
    """
    params.validate()

    if maturity < 0:
        raise ValueError("maturity must be non-negative.")

    if maturity == 0:
        return params.v0, 0.0

    exp_term = np.exp(-params.kappa * maturity)

    mean = params.theta + (params.v0 - params.theta) * exp_term

    variance = (
        params.v0
        * params.sigma_v**2
        * exp_term
        * (1.0 - exp_term)
        / params.kappa
        + params.theta
        * params.sigma_v**2
        * (1.0 - exp_term) ** 2
        / (2.0 * params.kappa)
    )

    return float(mean), float(max(variance, 0.0))


def cir_truncation_interval(
    params: CIRParams,
    maturity: float,
    std_width: float = 8.0,
) -> tuple[float, float]:
    """
    Choose a simple truncation interval for the COS method.

    The CIR variance is non-negative, so the lower bound is set to zero.
    The upper bound is based on mean plus several standard deviations.
    """
    if std_width <= 0:
        raise ValueError("std_width must be positive.")

    mean, variance = cir_terminal_moments(params=params, maturity=maturity)
    std = np.sqrt(variance)

    lower = 0.0
    upper = mean + std_width * std

    return lower, float(max(upper, 1e-8))


def cir_terminal_variance_cf(
    u: np.ndarray | float,
    params: CIRParams,
    maturity: float,
) -> np.ndarray | complex:
    """
    Characteristic function of terminal CIR variance v_T.

    The formula uses the noncentral chi-square distribution of the CIR process.
    It gives:

        E[exp(iu v_T) | v_0]

    This is different from the integrated variance transform. For VIX options,
    the payoff depends on terminal VIX, which is linked to terminal variance.
    """
    params.validate()

    if maturity < 0:
        raise ValueError("maturity must be non-negative.")

    u_array = np.asarray(u, dtype=np.complex128)

    if maturity == 0:
        result = np.exp(1j * u_array * params.v0)
        return result.item() if result.ndim == 0 else result

    exp_term = np.exp(-params.kappa * maturity)

    c = params.sigma_v**2 * (1.0 - exp_term) / (4.0 * params.kappa)
    degrees = 4.0 * params.kappa * params.theta / params.sigma_v**2

    noncentrality = (
        4.0
        * params.kappa
        * exp_term
        * params.v0
        / (params.sigma_v**2 * (1.0 - exp_term))
    )

    denominator = 1.0 - 2.0j * c * u_array

    result = denominator ** (-degrees / 2.0) * np.exp(
        noncentrality * (1.0j * c * u_array) / denominator
    )

    return result.item() if result.ndim == 0 else result