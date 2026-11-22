from bayes_sse._utils.matrix_utils import divide_inv
from bayes_sse.inference.gssm import StateSpaceModel
from bayes_sse.typing import (
    ControlVector,
    CovarianceMatrix,
    ObservationVector,
    StateVector,
)

from ._base import KalmanFilterResult, KalmanFilterState, LocalApproximationKalmanFilter


class KF(LocalApproximationKalmanFilter):
    """Kalman filter for linear state transitions and observations.

    The model's ``transition_func`` and ``observation_func`` supply the
    predicted means. The filter propagates covariances using the state and
    noise matrices from ``transition_jacobians`` and ``observation_jacobians``.

    The state-space equations for a single step are::

        x(k) = f(x(k-1), v(k-1), u1(k-1))
        z(k) = h(x(k), n(k), u2(k))

    Here ``f`` and ``h`` are the model's ``transition_func`` and ``observation_func``.
    The noises are ``v`` and ``n``. The controls ``u1`` and ``u2`` are
    ``transition_input`` and ``observation_input``, respectively.
    """

    def __str__(self) -> str:
        return "Kalman Filter"

    def estimate(
        self,
        state: StateVector,
        state_covariance: CovarianceMatrix,
        observation: ObservationVector,
        model: StateSpaceModel,
        transition_input: ControlVector,
        observation_input: ControlVector,
    ) -> KalmanFilterResult:
        """Predict from time k-1 and correct the prediction with the observation at k.

        :param state: Posterior state mean at k-1, as a column vector.
        :param state_covariance: Posterior state covariance at k-1.
        :param observation: Measurement at k, as a column vector.
        :param model: State-space model supplying the transition and observation
            functions, their Jacobians, and the means and covariances of the
            Gaussian or combined Gaussian noise sources.
        :param transition_input: Control for the transition from k-1 to k, or ``None``.
        :param observation_input: Control for the measurement at k, or ``None``.
        :return: The posterior state mean, posterior state covariance, and a
            ``KalmanFilterState`` with ``predicted_state_mean``, ``predicted_state_covariance``,
            ``predicted_observation_mean``, ``predicted_observation_covariance`` (the
            innovation covariance), ``innovation``, and ``filter_gain``.
        """
        # Time update (prediction)
        f, g = model.transition_jacobians(
            state, state_noise=model.state_noise.mean, control_x=transition_input
        )

        predicted_state = model.transition_func(
            state, model.state_noise.mean, transition_input
        )
        predicted_state_covariance = (
            f @ state_covariance @ f.T + g @ model.state_noise.covariance @ g.T
        )

        # Observation / measurement update (correction)
        c, h = model.observation_jacobians(
            predicted_state,
            observation_noise=model.observation_noise.mean,
            control_z=observation_input,
        )

        predicted_observation_covariance = (
            c @ predicted_state_covariance @ c.T
            + h @ model.observation_noise.covariance @ h.T
        )
        filter_gain = divide_inv(
            predicted_state_covariance @ c.T, predicted_observation_covariance
        )

        predicted_observation = model.observation_func(
            predicted_state, model.observation_noise.mean, observation_input
        )
        innovation = model.innovation(observation, predicted_observation)

        updated_state_mean = predicted_state + filter_gain @ innovation
        updated_state_covariance = (
            predicted_state_covariance
            - filter_gain @ predicted_observation_covariance @ filter_gain.T
        )

        filter_state = KalmanFilterState(
            predicted_state,
            predicted_state_covariance,
            predicted_observation,
            predicted_observation_covariance,
            innovation,
            filter_gain,
        )

        return updated_state_mean, updated_state_covariance, filter_state
