from bayes_sse._utils.matrix_utils import divide_inv
from bayes_sse.inference.gssm import StateSpaceModel
from bayes_sse.typing import (
    ControlVector,
    CovarianceMatrix,
    ObservationVector,
    StateVector,
)

from ._base import KalmanFilterResult, KalmanFilterState, LocalApproximationKalmanFilter


class EKF(LocalApproximationKalmanFilter):
    """Extended Kalman filter for nonlinear state transitions and observations.

    The model's ``transition_func`` and ``observation_func`` supply the
    predicted means. For first-order covariance propagation, the filter calls
    ``transition_jacobians`` at the previous state mean and
    ``observation_jacobians`` at the predicted state mean.

    The state-space equations for a single step are::

        x(k) = f(x(k-1), v(k-1), u1(k-1))
        z(k) = h(x(k), n(k), u2(k))

    Here ``f`` and ``h`` are the model's ``transition_func`` and ``observation_func``.
    The noises are ``v`` and ``n``. The controls ``u1`` and ``u2`` are
    ``transition_input`` and ``observation_input``, respectively.
    """

    def __str__(self) -> str:
        return "Extended Kalman Filter"

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
            state,
            state_noise=model.state_noise.mean,
            observation_noise=model.observation_noise.mean,
            transition_input=transition_input,
            observation_input=observation_input,
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
            state_noise=model.state_noise.mean,
            observation_noise=model.observation_noise.mean,
            transition_input=transition_input,
            observation_input=observation_input,
        )

        r = model.observation_noise.covariance
        cross_cov = predicted_state_covariance @ c.T
        predicted_observation_covariance = c @ cross_cov + h @ r @ h.T
        filter_gain = divide_inv(cross_cov, predicted_observation_covariance)

        predicted_observation = model.observation_func(
            predicted_state, model.observation_noise.mean, observation_input
        )
        innovation = model.innovation(observation, predicted_observation)

        updated_state = predicted_state + filter_gain @ innovation
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

        return updated_state, updated_state_covariance, filter_state
