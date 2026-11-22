from typing import cast

import numpy as np
from scipy.linalg import block_diag

from bayes_sse._utils.matrix_utils import divide_inv, generate_symmetric_sigma_points
from bayes_sse.inference.gssm import StateSpaceModel
from bayes_sse.typing import (
    ControlVector,
    CovarianceMatrix,
    Float64Array,
    Float64Matrix,
    Float64Vector,
    ObservationBatch,
    ObservationVector,
    StateBatch,
    StateVector,
)

from ._base import KalmanFilterResult, KalmanFilterState, UnscentedTransformFilter


class UKF(UnscentedTransformFilter):
    """
    Unscented Kalman filter for a nonlinear state-space model.

    The sigma points include the state, process noise, and observation noise.
    ``estimate`` propagates them through ``model.transition_func`` and
    ``model.observation_func``.

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
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

    def __str__(self) -> str:
        return f"UKF: alpha: {self._alpha}; beta: {self._beta}; kappa: {self._kappa}"

    def estimate(
        self,
        state: StateVector,
        state_covariance: CovarianceMatrix,
        observation: ObservationVector,
        model: StateSpaceModel,
        transition_input: ControlVector,
        observation_input: ControlVector,
    ) -> KalmanFilterResult:
        """Predict and update the state with the unscented transform.

        :param state: Posterior state mean at k-1.
        :param state_covariance: State covariance at k-1, or its square-root factor for ``SRUKF``.
        :param observation: Measurement at k.
        :param model: State-space model supplying the transition and observation
            functions and their noise distributions.
        :param transition_input: Control for the transition from k-1 to k, or ``None``.
        :param observation_input: Control for the measurement at k, or ``None``.
        :return: The updated state mean, updated state covariance,
            and a ``KalmanFilterState`` containing the predictions, innovation, and gain.
            ``SRUKF`` returns covariance factors for the updated state and predictions.
        """
        state_dim = model.state_dim
        state_noise_dim = model.state_noise.dim

        augmented_dim = state_dim + state_noise_dim + model.observation_noise.dim
        n_sigma_points = 2 * augmented_dim + 1
        _lambda = self._alpha**2 * (augmented_dim + self._kappa) - augmented_dim

        weights = np.asarray([_lambda, 0.5, 0]) / (augmented_dim + _lambda)
        weights[2] = weights[0] + (1 - self._alpha**2) + self._beta

        # Generate sigma points
        sigma_point_offsets = self.eval_sigma_points_offset(
            augmented_dim, state_covariance, model
        )

        augmented_mean = np.concatenate(
            (
                state,
                model.state_noise.mean,
                model.observation_noise.mean,
            )
        )

        sigma_points = generate_symmetric_sigma_points(
            augmented_mean, sigma_point_offsets, np.sqrt(augmented_dim + _lambda)
        )

        # Propagate sigma points through state model
        predicted_state = model.transition_func(
            sigma_points[:state_dim, :],
            sigma_points[state_dim : state_dim + state_noise_dim, :],
            transition_input,
        )
        predicted_state_mean = weights[0] * predicted_state[:, :1] + weights[
            1
        ] * predicted_state[:, 1:].sum(axis=1, keepdims=True)
        state_deviations = predicted_state - predicted_state_mean
        predicted_state_covariance = self.predict_state_covariance(
            weights, state_deviations
        )

        # Propagate sigma points through observation model
        predicted_observation = model.observation_func(
            predicted_state,
            sigma_points[state_dim + state_noise_dim :, :],
            observation_input,
        )
        predicted_observation_mean = weights[0] * predicted_observation[:, 0] + weights[
            1
        ] * np.sum(predicted_observation[:, 1:], axis=1)
        observation_deviations = predicted_observation - np.tile(
            predicted_observation_mean, n_sigma_points
        )
        predicted_observation_covariance = self.predict_observation_covariance(
            model, weights, observation_deviations
        )

        # Correction (measurement update)
        cross_covariance = weights[2] * np.outer(
            state_deviations[:, 0], observation_deviations[:, 0]
        ) + weights[1] * (state_deviations[:, 1:] @ observation_deviations[:, 1:].T)
        filter_gain = self.eval_filter_gain(
            cross_covariance, predicted_observation_covariance
        )
        innovation = model.innovation(observation, predicted_observation_mean)

        updated_state = predicted_state_mean + filter_gain @ innovation
        updated_state_covariance = self.correct_state_covariance(
            filter_gain,
            predicted_state_covariance,
            predicted_observation_covariance,
            model,
        )

        filter_state = KalmanFilterState(
            predicted_state_mean,
            predicted_state_covariance,
            predicted_observation_mean,
            predicted_observation_covariance,
            innovation,
            filter_gain,
        )

        return updated_state, updated_state_covariance, filter_state

    def predict_observation_covariance(
        self,
        model: StateSpaceModel,
        weights: Float64Array,
        observation_deviations: ObservationBatch,
    ) -> CovarianceMatrix:
        return (
            weights[2] * observation_deviations[:, 0] @ observation_deviations[:, 0].T
            + weights[1]
            * observation_deviations[:, 1:]
            @ observation_deviations[:, 1:].T
        )

    def predict_state_covariance(
        self,
        weights: Float64Vector,
        state_deviations: StateBatch,
    ) -> CovarianceMatrix:
        return (
            weights[2] * state_deviations[:, 0] @ state_deviations[:, 0].T
            + weights[1] * state_deviations[:, 1:] @ state_deviations[:, 1:].T
        )

    def correct_state_covariance(
        self,
        filter_gain: Float64Matrix,
        predicted_state_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
        model: StateSpaceModel,
    ) -> CovarianceMatrix:
        return (
            predicted_state_covariance
            - filter_gain @ predicted_observation_covariance @ filter_gain.T
        )

    def eval_filter_gain(
        self,
        cross_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        return divide_inv(cross_covariance, predicted_observation_covariance)

    def eval_sigma_points_offset(
        self,
        augment_dim: int,
        state_covariance: CovarianceMatrix,
        model: StateSpaceModel,
    ) -> Float64Matrix:
        return cast(
            Float64Matrix,
            block_diag(
                np.linalg.cholesky(state_covariance),
                np.linalg.cholesky(model.state_noise.covariance),
                np.linalg.cholesky(model.observation_noise.covariance),
            ),
        )
