from typing import cast

import numpy as np
from scipy.linalg import block_diag

from bayes_sse._utils.matrix_utils import cholesky_update, divide_inv
from bayes_sse.inference.gssm import StateSpaceModel
from bayes_sse.typing import (
    CovarianceMatrix,
    Float64Matrix,
    Float64Vector,
    ObservationBatch,
    StateBatch,
)

from ._ukf import UKF


class SRUKF(UKF):
    """
    Square-root unscented Kalman filter.

    ``estimate`` accepts a state covariance factor and noise covariance
    factors. It uses QR factorization and Cholesky updates to compute the
    predicted and updated covariance factors.

    The state-space equations for a single step are::

        x(k) = f(x(k-1), v(k-1), u1(k-1))
        z(k) = h(x(k), n(k), u2(k))

    Here ``f`` and ``h`` are the model's ``transition_func`` and ``observation_func``.
    The noises are ``v`` and ``n``. The controls ``u1`` and ``u2`` are
    ``transition_input`` and ``observation_input``, respectively.

    :param alpha: Primary scaling parameter of the unscented transform.
    :param beta: Correction to the central covariance weight.
    :param kappa: Secondary scaling parameter of the unscented transform.
    """

    def __init__(self, alpha: float, beta: float, kappa: float):
        super().__init__(alpha, beta, kappa)

    def __str__(self) -> str:
        return f"SR-UKF: alpha: {self._alpha}; beta: {self._beta}; kappa: {self._kappa}"

    def predict_observation_covariance(
        self,
        model: StateSpaceModel,
        weights: Float64Vector,
        observation_deviations: ObservationBatch,
    ) -> CovarianceMatrix:
        return self._predict_covariance(weights, observation_deviations).T

    def predict_state_covariance(
        self,
        weights: Float64Vector,
        state_deviations: StateBatch,
    ) -> CovarianceMatrix:
        return self._predict_covariance(weights, state_deviations)

    def correct_state_covariance(
        self,
        filter_gain: Float64Matrix,
        predicted_state_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
        model: StateSpaceModel,
    ) -> CovarianceMatrix:
        downdate_vectors = filter_gain @ predicted_observation_covariance
        upper_factor = predicted_state_covariance

        for j in range(downdate_vectors.shape[1]):
            upper_factor = cholesky_update(
                upper_factor, downdate_vectors[:, j : j + 1], "-"
            )

        return upper_factor.T

    def eval_filter_gain(
        self,
        cross_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        return divide_inv(
            divide_inv(cross_covariance, predicted_observation_covariance.T),
            predicted_observation_covariance,
        )

    def eval_sigma_points_offset(
        self,
        augment_dim: int,
        state_covariance: CovarianceMatrix,
        model: StateSpaceModel,
    ) -> Float64Matrix:
        return cast(
            Float64Matrix,
            block_diag(
                state_covariance,
                model.state_noise.covariance,
                model.observation_noise.covariance,
            ),
        )

    def _predict_covariance(
        self,
        weights: Float64Vector,
        deviations: Float64Matrix,
    ) -> CovarianceMatrix:
        weighted_deviations = np.sqrt(weights[1]) * deviations[:, :1]
        _, upper_covariance_factor = np.linalg.qr(
            weighted_deviations.T,
            mode="reduced",
        )

        return cholesky_update(
            upper_covariance_factor,
            np.sqrt(abs(weights[2])) * deviations[:, 0],
            "+" if weights[2] >= 0.0 else "-",
        )
