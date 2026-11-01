from abc import ABC, abstractmethod

import numpy as np

from bayes_sse.inference.stochastic_models.covariance_type import is_sqrt_like
from bayes_sse.inference.stochastic_models.stochastic_models import StochasticModel


class CovarianceMatrixAdaptation(ABC):

    @abstractmethod
    def reconcile(self, noise: StochasticModel, cov: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply covariance adaptation method to covariance matrix of stochastic process (ie to noise) based on
        covariance matrix (cov) of a stochastic vector (from variational space).
        Variational space described by stochastic process.
        :param noise: covariance matrix of stochastic process (which describes).
        :param cov: covariance matrix of stochastic vector from a variational space.
        :return: reconciled (adapted) covariance matrix of stochastic vector.
        """
        pass


class AnnealAdaptation(CovarianceMatrixAdaptation):
    """
    Annealing noise covariance adaptation method
    """

    def __init__(self, annealing_factor: float, minimum_allowed_variance: float):
        self._annealing_factor = annealing_factor
        self._minimum_allowed_variance = minimum_allowed_variance

    def reconcile(self, noise: StochasticModel, cov: np.ndarray, **kwargs) -> np.ndarray:
        if is_sqrt_like(noise.covariance_type):
            return np.diag(
                -np.diag(cov)
                + np.sqrt(
                    np.maximum(
                        self._annealing_factor * (np.diag(noise.covariance) ** 2),
                        self._minimum_allowed_variance
                    )
                    + np.diag(cov) ** 2
                )
            )

        return np.diag(np.maximum(
            self._annealing_factor * np.diag(noise.covariance), self._minimum_allowed_variance)
        )


class LambdaDecayAdaptation(CovarianceMatrixAdaptation):
    """
    RLS like lambda decay noise covariance adaptation method
    """

    def __init__(self, lambda_factor: float):
        self._lambda_factor = lambda_factor

    def reconcile(self, noise: StochasticModel, cov: np.ndarray, **kwargs) -> np.ndarray:
        if is_sqrt_like(noise.covariance_type):
            return noise.covariance

        return (1 / self._lambda_factor - 1) * cov


class RobbinsMonroAdaptation(CovarianceMatrixAdaptation):
    """
    Robbins-Monro stochastic approximation.
    """

    def __init__(self, nu_initial_inv: float, nu_final_inv: float):
        self._nu_initial_inv = nu_initial_inv  # equal to 1 / nu_initial
        self._nu_final_inv = nu_final_inv  # equal to 1 / nu_final

    def reconcile(self, noise: StochasticModel, cov: np.ndarray, **kwargs) -> np.ndarray:
        nu = 1 / self._nu_initial_inv

        kalman_gain = kwargs["kalman_gain"]
        innovation = kwargs["innovation"]

        kgi = kalman_gain @ (kalman_gain @ innovation @ innovation.T).T

        if is_sqrt_like(noise.covariance_type):
            rec_cov = np.diag(
                -np.diag(cov)
                + np.sqrt(
                    (1 - nu) * (noise.covariance ** 2) + nu * np.diag(kgi)
                    + np.diag(cov) ** 2
                )
            )
        else:
            rec_cov = (1 - nu) * noise.covariance + nu * kgi

        self._nu_initial_inv = min(self._nu_initial_inv + 1, self._nu_final_inv)

        return rec_cov
