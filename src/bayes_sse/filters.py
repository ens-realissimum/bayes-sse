from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, cast

import numpy as np
from numpy.random import random

import bayes_sse._utils.matrix_utils as m_utils
import bayes_sse.core.numerical_computations as num_computations
from bayes_sse.core.gaussian_mixture_info import GaussianMixture
from bayes_sse.core.linearization_type import LinearizationType
from bayes_sse.inference.gssm import StateSpaceModel
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType

from .typing import (
    ControlVector,
    CovarianceFactor,
    CovarianceMatrix,
    Float64Array,
    Float64Matrix,
    Float64Vector,
    IntegrationPoints,
    IntegrationWeights,
    ObservationBatch,
    ObservationVector,
    ParticleIndices,
    ParticleMatrix,
    ParticleWeights,
    StateBatch,
    StateVector,
)


def init_gmi(
    initial_state: StateVector,
    n_particles: int,
    n_gmm_componets: int,
) -> GaussianMixture:
    state_dim = initial_state.shape[0]

    samples = np.random.randn(n_particles, state_dim) + initial_state
    return GaussianMixture.init(n_gmm_componets, samples)


class KalmanFilterState:
    def __init__(
        self,
        predicted_state_mean: StateVector,
        predicted_state_covariance: CovarianceMatrix,
        predicted_observation_mean: ObservationVector,
        predicted_observation_covariance: CovarianceMatrix,
        innovation: Float64Vector,
        filter_gain: Float64Matrix,
    ):
        self._predicted_state_mean = predicted_state_mean
        self._predicted_state_covariance = predicted_state_covariance
        self._predicted_observation_mean = predicted_observation_mean
        self._predicted_observation_covariance = predicted_observation_covariance
        self._innovation = innovation
        self._filter_gain = filter_gain

    @property
    def predicted_state_mean(self) -> StateVector:
        return self._predicted_state_mean

    @property
    def predicted_state_covariance(self) -> CovarianceMatrix:
        return self._predicted_state_covariance

    @property
    def predicted_observation_mean(self) -> ObservationVector:
        return self._predicted_observation_mean

    @property
    def predicted_observation_covariance(self) -> CovarianceMatrix:
        return self._predicted_observation_covariance

    @property
    def innovation(self) -> Float64Vector:
        return self._innovation

    @property
    def filter_gain(self) -> Float64Matrix:
        return self._filter_gain


KalmanFilterResult = tuple[StateVector, CovarianceMatrix, KalmanFilterState]


class PointEstimateMethod(Enum):
    mean = 0
    median = 1


class LocalApproximationKalmanFilter(Protocol):
    """
    Base class (interface) for linear form of Kalman filter and all filters
    which are based on approximation of non-linear transformations like sigma-point filters,
    extended Kalman filter, e.t.c.
    Another way to estimate is approximate probability density function (pdf).
    This is used in particle filters and their variations.
    """

    @abstractmethod
    def estimate(
        self,
        state: StateVector,
        state_covariance: CovarianceMatrix,
        observation: ObservationVector,
        model: StateSpaceModel,
        transition_input: ControlVector,
        observation_input: ControlVector,
    ) -> KalmanFilterResult: ...


type PdfApproximationFilterResult[ApproximationT] = tuple[
    StateVector,
    ApproximationT,
]


class PdfApproximationFilter[ApproximationT](ABC):
    """
    Base class (interface) for Kalman filter (and all other filters,
    e.g. particle filters family) which are based on approximation of
    predictive distributions (probability density function (pdf))
    after a non-linear transformations, e.t.c.
    """

    def __init__(
        self,
        point_estimate_method: PointEstimateMethod,
    ):
        self._point_estimate_method = point_estimate_method

    @abstractmethod
    def estimate(
        self,
        approximation: ApproximationT,
        observation: ObservationVector,
        model: StateSpaceModel,
        transition_input: ControlVector,
        observation_input: ControlVector,
    ) -> PdfApproximationFilterResult[ApproximationT]:
        """
        Estimates state mean (x(k)) of dynamic system at time k based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)].

        :param data_set: BootstrapDataSet, particle filter data structure
            (contains set of particles as well as their corresponding weights).
        :param observation:
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue
            (evolution of state, relation between state and
            observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            estimate: np array, estimates of state at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            data_set: BootstrapDataSet, updated Particle filter data structure.
                Contains set of particles as well as their corresponding weights.
        """
        ...


class KF(LocalApproximationKalmanFilter):
    """
    KF - Kalman Filter (linear).

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)],

    The filter is a linear estimator, i.e. the filter works under assumption
    that the dynamic state space system is a linear.
    The filter is used linearize function of dynamic state space system
    to reduce corresponding model to linear model.
    Therefore, the filter can be used with nonlinear dynamic systems.
    In this case the filter will work as linear estimator of nonlinear dynamic system.

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
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
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)],

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue
            (evolution of state, relation between state and
        observation, state noise, observation noise, e.t.c.); state noise must be of type 'gaussian' or 'combo-gaussian';
        observation noise must be of type 'gaussian' or 'combo-gaussian'.
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        # Time update (prediction)
        f = model.linearize(
            LinearizationType.F,
            state,
            model.state_noise.mean,
            None,
            transition_input,
            None,
        )

        g = model.linearize(
            LinearizationType.G,
            state,
            model.state_noise.mean,
            None,
            transition_input,
            None,
        )

        predicted_state = model.transition_func(
            state, model.state_noise.mean, transition_input
        )
        predicted_state_covariance = (
            f @ state_covariance @ f.T + g @ model.state_noise.covariance @ g.T
        )

        # Observation / measurement update (correction)
        c = model.linearize(
            LinearizationType.C,
            predicted_state,
            None,
            model.observation_noise.mean,
            None,
            observation_input,
        )

        h = model.linearize(
            LinearizationType.H,
            predicted_state,
            None,
            model.observation_noise.mean,
            None,
            observation_input,
        )

        predicted_observation_covariance = (
            c @ predicted_state_covariance @ c.T
            + h @ model.observation_noise.covariance @ h.T
        )
        filter_gain = m_utils.divide_inv(
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


class EKF(LocalApproximationKalmanFilter):
    """
    EKF - Extended Kalman Filter.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)],

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
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
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)],

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue
            (evolution of state, relation between state and
        observation, state noise, observation noise, e.t.c.);
            state noise must be of type 'gaussian' or 'combo-gaussian';
        observation noise must be of type 'gaussian' or 'combo-gaussian'.
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        # Time update (prediction)
        f = model.linearize(
            LinearizationType.F,
            state,
            model.state_noise.mean,
            model.observation_noise.mean,
            transition_input,
            observation_input,
        )

        g = model.linearize(
            LinearizationType.G,
            state,
            model.state_noise.mean,
            model.observation_noise.mean,
            transition_input,
            observation_input,
        )

        predicted_state = model.transition_func(
            state, model.state_noise.mean, transition_input
        )
        predicted_state_covariance = (
            f @ state_covariance @ f.T + g @ model.state_noise.covariance @ g.T
        )

        # Observation / measurement update (correction)
        c = model.linearize(
            LinearizationType.C,
            predicted_state,
            model.state_noise.mean,
            model.observation_noise.mean,
            transition_input,
            observation_input,
        )

        h = model.linearize(
            LinearizationType.H,
            predicted_state,
            model.state_noise.mean,
            model.observation_noise.mean,
            transition_input,
            observation_input,
        )

        predicted_observation_covariance = (
            c @ predicted_state_covariance @ c.T
            + h @ model.observation_noise.covariance @ h.T
        )
        filter_gain = m_utils.divide_inv(
            predicted_state_covariance @ c.T, predicted_observation_covariance
        )

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


class UnscentedTransformFilter(LocalApproximationKalmanFilter):
    @abstractmethod
    def predict_observation_covariance(
        self,
        model: StateSpaceModel,
        weights: Float64Vector,
        observation_deviations: ObservationBatch,
    ) -> CovarianceMatrix: ...

    @abstractmethod
    def predict_state_covariance(
        self,
        weights: Float64Vector,
        state_deviations: StateBatch,
    ) -> CovarianceMatrix: ...

    @abstractmethod
    def correct_state_covariance(
        self,
        filter_gain: Float64Matrix,
        predicted_state_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
        model: StateSpaceModel,
    ) -> CovarianceMatrix: ...

    @abstractmethod
    def eval_filter_gain(
        self,
        cross_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
    ) -> Float64Matrix: ...

    @abstractmethod
    def eval_sigma_points_offset(
        self,
        augment_dim: int,
        state_covariance: CovarianceMatrix,
        model: StateSpaceModel,
    ) -> Float64Matrix: ...


class UKF(UnscentedTransformFilter):
    """
    UKF - Unscented Kalman Filter. Sigma-Point Kalman Filter variant.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)],

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        alpha - UKF scale factor;
        beta - UKF covariance correction factor;
        kappa - UKF secondary scaling parameter.
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
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)],

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue
            (evolution of state, relation between state and observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterResult`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
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

        sigma_points = augmented_mean[:, None] + sigma_point_offsets

        # Propagate sigma points through state model
        predicted_state = model.transition_func(
            sigma_points[:state_dim, :],
            sigma_points[state_dim : state_dim + state_noise_dim, :],
            transition_input,
        )
        predicted_state_mean = weights[0] * predicted_state[:, 0] + weights[1] * np.sum(
            predicted_state[:, 1:], axis=1
        )
        state_deviations = predicted_state - np.tile(
            predicted_state_mean, n_sigma_points
        )
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
        cross_covariance = (
            weights[2] * state_deviations[:, 0] @ observation_deviations[:, 0].T
            + weights[1] * state_deviations[:, 1:] @ observation_deviations[:, 1:].T
        )
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
            + model.observation_noise.covariance
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
        return m_utils.divide_inv(cross_covariance, predicted_observation_covariance)

    def eval_sigma_points_offset(
        self,
        augment_dim: int,
        state_covariance: CovarianceMatrix,
        model: StateSpaceModel,
    ) -> Float64Matrix:
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            augment_dim,
            [
                cast(CovarianceFactor, np.linalg.cholesky(state_covariance)),
                cast(
                    CovarianceFactor,
                    np.linalg.cholesky(model.state_noise.covariance),
                ),
                cast(
                    CovarianceFactor,
                    np.linalg.cholesky(model.observation_noise.covariance),
                ),
            ],
        )


class SRUKF(UKF):
    """
    SRUKF - Square Root Unscented Kalman Filter. Sigma-Point Kalman Filter variant.

    This is a square root form of UKF.
    The square-root Unscented Kalman filter which is also O(L ** 3) for general state estimation
    and O(L ** 2) for parameter estimation
    (note, the original formulation of the UKF for parameter-estimation was O(L ** 3)).
    In addition, the square-root forms have the added benefit of numerical stability and
    guaranteed positive semi-definiteness of the state covariances.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        alpha - UKF scale factor;
        beta - UKF covariance correction factor;
        kappa - UKF secondary scaling parameter.
    """

    def __init__(self, alpha: float, beta: float, kappa: float):
        super().__init__(alpha, beta, kappa)

    def __str__(self) -> str:
        return f"SR UKF: alpha: {self._alpha}; beta: {self._beta}; kappa: {self._kappa}"

    def predict_observation_covariance(
        self,
        model: StateSpaceModel,
        weights: Float64Vector,
        observation_deviations: ObservationBatch,
    ) -> CovarianceMatrix:
        sqrt_weights = np.sqrt(np.abs(weights))
        _, predicted_observation_sqrt_covariance = np.linalg.qr(
            np.vstack(
                (
                    sqrt_weights[1] * observation_deviations[:, 1:].T,
                    model.observation_noise.covariance,
                )
            )
        )

        upper_obseration_covariance_factor = m_utils.cholesky_update(
            predicted_observation_sqrt_covariance,
            sqrt_weights[2] * observation_deviations[:, 0],
            "+" if weights[2] > 0 else "-",
        )

        return upper_obseration_covariance_factor.T

    def predict_state_covariance(
        self,
        weights: Float64Vector,
        state_deviations: StateBatch,
    ) -> CovarianceMatrix:
        noncentral_weight = weights[1]
        central_weight = weights[2]

        weighted_deviations = np.sqrt(noncentral_weight) * state_deviations[:, 1:]

        _, upper_covariance_factor = np.linalg.qr(
            weighted_deviations.T,
            mode="reduced",
        )

        # QR does not guarantee a positive diagonal, but cholesky_update
        # expects a conventional upper Cholesky factor.
        row_signs = np.where(
            np.diag(upper_covariance_factor) < 0.0,
            -1.0,
            1.0,
        )
        upper_covariance_factor *= row_signs[:, None]

        return m_utils.cholesky_update(
            upper_covariance_factor,
            np.sqrt(abs(central_weight)) * state_deviations[:, 0],
            "+" if central_weight >= 0.0 else "-",
        )

    def correct_state_covariance(
        self,
        filter_gain: Float64Matrix,
        predicted_state_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
        model: StateSpaceModel,
    ) -> CovarianceMatrix:
        downdate_vectors = filter_gain @ predicted_observation_covariance
        updated_state_sqrt_covariance_upper = predicted_state_covariance

        for downdate_vector in downdate_vectors.T:
            updated_state_sqrt_covariance_upper = m_utils.cholesky_update(
                updated_state_sqrt_covariance_upper,
                downdate_vector,
                "-",
            )

        return updated_state_sqrt_covariance_upper.T

    def eval_filter_gain(
        self,
        cross_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        return m_utils.divide_inv(
            m_utils.divide_inv(cross_covariance, predicted_observation_covariance.T),
            predicted_observation_covariance,
        )

    def eval_sigma_points_offset(
        self,
        augment_dim: int,
        state_covariance: CovarianceMatrix,
        model: StateSpaceModel,
    ) -> Float64Matrix:
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            augment_dim,
            [
                state_covariance,
                model.state_noise.covariance,
                model.observation_noise.covariance,
            ],
        )


class CentralDiffTransformFilter(LocalApproximationKalmanFilter):
    @abstractmethod
    def correct_state_covariance(
        self,
        state_sqrt_covariance: CovarianceMatrix,
    ) -> CovarianceMatrix: ...

    @abstractmethod
    def eval_observation_sigma_points_offsets(
        self,
        model: StateSpaceModel,
        predicted_state_sqrt_covariance: CovarianceMatrix,
    ) -> Float64Matrix: ...

    @abstractmethod
    def eval_state_sigma_points_offsets(
        self,
        model: StateSpaceModel,
        state_covariance: CovarianceMatrix,
    ) -> Float64Matrix: ...


class CDKF(CentralDiffTransformFilter):
    """
    CDKF - Central Difference Kalman Filter. Sigma-Point Kalman Filter variant.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        scale_factor - scale factor / difference step size.
    """

    def __init__(self, scale_factor: float):
        self._scale_factor = scale_factor

    def __str__(self) -> str:
        return f"CDKF. scale factor: {self._scale_factor}"

    def estimate(
        self,
        state: StateVector,
        state_covariance: CovarianceMatrix,
        observation: ObservationVector,
        model: StateSpaceModel,
        transition_input: ControlVector,
        observation_input: ControlVector,
    ) -> KalmanFilterResult:
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)],

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue
            (evolution of state, relation between state and
        observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: sqrt version of predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: sqrt version of innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        state_dim = model.state_dim
        state_noise_dim = model.state_noise_dim
        observation_noise_dim = model.observation_noise_dim
        scale_factor_squared = self._scale_factor**2

        state_weights = np.array(
            [
                [2 * (scale_factor_squared - state_dim - state_noise_dim), 1],
                [self._scale_factor, np.sqrt(scale_factor_squared - 1)],
            ]
        ) / (2 * scale_factor_squared)

        observation_weights = state_weights.copy()
        observation_weights[0, 0] = (
            scale_factor_squared - state_dim - observation_noise_dim
        ) / scale_factor_squared

        # Generate sigma points for the state
        unscaled_state_offsets = self.eval_state_sigma_points_offsets(
            model,
            state_covariance,
        )
        state_sigma_points = m_utils.generate_symmetric_sigma_points(
            (np.vstack((state, model.state_noise.mean))),
            unscaled_state_offsets,
            self._scale_factor,
        )

        # Propagate state sigma points through state model
        predicted_state = model.transition_func(
            state_sigma_points[:state_dim, :],
            state_sigma_points[state_dim:, :],
            transition_input,
        )
        predicted_state_mean, a, b = self._compute_central_difference_terms(
            predicted_state,
            state_weights,
        )

        _, predicted_state_sqrt_covariance = np.linalg.qr(np.hstack((a, b)).T)
        predicted_state_sqrt_covariance = predicted_state_sqrt_covariance.T

        # Generate sigma points for the observations
        unscaled_observation_offsets = self.eval_observation_sigma_points_offsets(
            model,
            predicted_state_sqrt_covariance,
        )
        observation_sigma_points = m_utils.generate_symmetric_sigma_points(
            np.vstack((predicted_state_mean, model.observation_noise.mean)),
            unscaled_observation_offsets,
            self._scale_factor,
        )

        # Propagate observation sigma points through observation model
        predicted_observation = model.observation_func(
            observation_sigma_points[:state_dim, :],
            observation_sigma_points[state_dim:, :],
            observation_input,
        )
        predicted_observation_mean, c, d = self._compute_central_difference_terms(
            predicted_observation,
            observation_weights,
        )

        _, upper_observation_covariance_factor = np.linalg.qr(
            (np.hstack((c, d, model.observation_noise.covariance))).T
        )
        lower_observation_covariance_factor = upper_observation_covariance_factor.T

        # Correction (measurement update)
        cross_covariance = predicted_state_sqrt_covariance @ c[:, :state_dim].T

        filter_gain = m_utils.divide_inv(
            m_utils.divide_inv(cross_covariance, upper_observation_covariance_factor),
            lower_observation_covariance_factor,
        )

        innovation = model.innovation(observation, predicted_observation_mean)

        updated_state = predicted_state_mean + filter_gain @ innovation

        _, upper_state_covariance_factor = np.linalg.qr(
            np.hstack(
                (
                    predicted_state_sqrt_covariance - filter_gain @ c[:, :state_dim],
                    filter_gain @ c[:, state_dim:],
                    filter_gain @ d,
                )
            ).T
        )
        lower_state_covariance_factor = upper_state_covariance_factor.T
        updated_state_covariance = self.correct_state_covariance(
            lower_state_covariance_factor
        )

        filter_state = KalmanFilterState(
            predicted_state_mean,
            predicted_state_sqrt_covariance,
            predicted_observation_mean,
            upper_observation_covariance_factor,
            innovation,
            filter_gain,
        )

        return updated_state, updated_state_covariance, filter_state

    @staticmethod
    def _compute_central_difference_terms(
        transformed_sigma_points: Float64Matrix,
        weights: Float64Array,
    ) -> tuple[Float64Vector, Float64Matrix, Float64Matrix]:
        sigma_point_count = transformed_sigma_points.shape[1]

        if sigma_point_count < 3 or sigma_point_count % 2 == 0:
            raise ValueError(
                "Expected one central point followed by symmetric point pairs"
            )

        offset_count = (sigma_point_count - 1) // 2
        negative_points_start = 1 + offset_count

        central_point = transformed_sigma_points[:, :1]
        positive_points = transformed_sigma_points[:, 1:negative_points_start]
        negative_points = transformed_sigma_points[:, negative_points_start:]

        predicted_mean = weights[0, 0] * central_point[:, 0] + (
            weights[0, 1] * transformed_sigma_points[:, 1:].sum(axis=1)
        )

        weighted_first_order_differences = weights[1, 0] * (
            positive_points - negative_points
        )

        weighted_second_order_differences = weights[1, 1] * (
            positive_points + negative_points - 2 * central_point
        )

        return (
            predicted_mean,
            weighted_first_order_differences,
            weighted_second_order_differences,
        )

    def correct_state_covariance(
        self,
        state_sqrt_covariance: CovarianceMatrix,
    ) -> CovarianceMatrix:
        return state_sqrt_covariance @ state_sqrt_covariance.T

    def eval_observation_sigma_points_offsets(
        self,
        model: StateSpaceModel,
        predicted_state_sqrt_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        observation_noise_sqrt_covariane = np.linalg.cholesky(
            model.observation_noise.covariance
        )
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            model.state_dim + model.observation_noise_dim,
            [predicted_state_sqrt_covariance, observation_noise_sqrt_covariane],
        )

    def eval_state_sigma_points_offsets(
        self,
        model: StateSpaceModel,
        state_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        state_noise_sqrt_covariance = cast(
            CovarianceFactor, np.linalg.cholesky(model.state_noise.covariance)
        )
        state_sqrt_covariance = cast(
            CovarianceFactor, np.linalg.cholesky(state_covariance)
        )
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            model.state_dim + model.state_noise_dim,
            [state_sqrt_covariance, state_noise_sqrt_covariance],
        )


class SRCDKF(CDKF):
    """
    SR CDKF - Square Root Central Difference Kalman Filter. Sigma-Point Kalman Filter variant.

    This is a square root form of CDKF. The square-root Central Difference Kalman filter
    which is also O(L ** 3) for general state estimation and O(L ** 2) for parameter estimation.
    In addition, the square-root forms have the added benefit of numerical stability
    and guaranteed positive semi-definiteness of the state covariances.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        scale_factor - scale factor / difference step size.
    """

    def __init__(self, scale_factor: float):
        super().__init__(scale_factor)

    def __str__(self) -> str:
        return f"SR CDKF. Scale factor: {self._scale_factor}"

    def correct_state_covariance(
        self,
        state_sqrt_covariance: CovarianceMatrix,
    ) -> CovarianceMatrix:
        return state_sqrt_covariance

    def eval_observation_sigma_points_offsets(
        self,
        model: StateSpaceModel,
        predicted_state_sqrt_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            model.state_dim + model.observation_noise_dim,
            [predicted_state_sqrt_covariance, model.observation_noise.covariance],
        )

    def eval_state_sigma_points_offsets(
        self,
        model: StateSpaceModel,
        state_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            model.state_dim + model.state_noise_dim,
            [state_covariance, model.state_noise.covariance],
        )


class CubatureTransformFilter(LocalApproximationKalmanFilter):
    @abstractmethod
    def predict_observation_covariance(
        self,
        model: StateSpaceModel,
        observations: ObservationBatch,
    ) -> CovarianceMatrix: ...

    @abstractmethod
    def predict_state_covariance(
        self,
        n_cubature_points: int,
        model: StateSpaceModel,
        predicted_state_mean: StateVector,
        predicted_cubature_points: StateBatch,
    ) -> CovarianceMatrix: ...

    @abstractmethod
    def correct_state_covariance(
        self,
        filter_gain: Float64Matrix,
        predicted_state_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
        model: StateSpaceModel,
        state: StateBatch,
        observation: ObservationBatch,
    ) -> CovarianceMatrix: ...

    @abstractmethod
    def eval_filter_gain(
        self,
        cross_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
    ) -> Float64Matrix: ...

    @abstractmethod
    def eval_observation_cubature_points_offsets(
        self,
        n_cubature_points: int,
        predicted_state_covariance: CovarianceMatrix,
    ) -> Float64Matrix: ...

    @abstractmethod
    def eval_state_cubature_points_offsets(
        self,
        state_covariance: CovarianceMatrix,
    ) -> Float64Matrix: ...


class CKF(CubatureTransformFilter):
    """
    CKF - Cubature Kalman Filter. Sigma-Point Kalman Filter variant.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        return "Cubature Kalman Filter (3rd degree)"

    def estimate(
        self,
        state: StateVector,
        state_covariance: CovarianceMatrix,
        observation: ObservationVector,
        model: StateSpaceModel,
        transition_input: ControlVector,
        observation_input: ControlVector,
    ) -> KalmanFilterResult:
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)].

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue
            (evolution of state, relation between state and
            observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        state_dimension = model.state_dim
        n_cubature_points = 2 * state_dimension

        # Generate cubature points for the state
        state_cubature_points = m_utils.generate_third_degree_cubature_points(
            state,
            self.eval_state_cubature_points_offsets(state_covariance),
        )

        # Propagate state cubature points through state model
        predicted_state = model.transition_func(
            state_cubature_points,
            np.tile(model.state_noise.mean, (1, n_cubature_points)),
            transition_input,
        )
        predicted_state_mean = np.sum(predicted_state, axis=1) / n_cubature_points
        predicted_state_covariance = self.predict_state_covariance(
            n_cubature_points,
            model,
            predicted_state_mean,
            predicted_state,
        )

        # Generate cubature points for the observations
        observatiob_cubature_points = m_utils.generate_third_degree_cubature_points(
            observation,
            self.eval_observation_cubature_points_offsets(
                n_cubature_points,
                predicted_state_covariance,
            ),
        )

        # Propagate observation cubature points through observation model
        predicted_observations = model.observation_func(
            observatiob_cubature_points,
            np.tile(model.observation_noise.mean, (1, n_cubature_points)),
            observation_input,
        )
        predicted_observation_mean = (
            np.sum(predicted_observations, axis=1) / n_cubature_points
        )

        # Correction (measurement update)
        normalization = np.sqrt(n_cubature_points)
        state_deviations = (
            observatiob_cubature_points - predicted_state_mean
        ) / normalization
        observation_deviations = (
            predicted_observations - predicted_observation_mean
        ) / normalization

        predicted_oservation_covariance = self.predict_observation_covariance(
            model,
            observation_deviations,
        )
        cross_covariance = state_deviations @ observation_deviations.T
        filter_gain = self.eval_filter_gain(
            cross_covariance,
            predicted_oservation_covariance,
        )

        innovation = model.innovation(observation, predicted_observation_mean)

        updated_state = predicted_state_mean + filter_gain @ innovation
        updated_state_covariance = self.correct_state_covariance(
            filter_gain,
            predicted_state_covariance,
            predicted_oservation_covariance,
            model,
            state_deviations,
            observation_deviations,
        )

        filter_state = KalmanFilterState(
            predicted_state_mean,
            predicted_state_covariance,
            predicted_observation_mean,
            predicted_oservation_covariance,
            innovation,
            filter_gain,
        )

        return updated_state, updated_state_covariance, filter_state

    def predict_observation_covariance(
        self,
        model: StateSpaceModel,
        observations: ObservationBatch,
    ) -> CovarianceMatrix:
        return observations @ observations.T + model.observation_noise.covariance

    def predict_state_covariance(
        self,
        n_cubature_points: int,
        model: StateSpaceModel,
        predicted_state_mean: StateVector,
        predicted_cubature_points: StateBatch,
    ) -> CovarianceMatrix:
        state_deviations = (predicted_cubature_points - predicted_state_mean) / (
            np.sqrt(n_cubature_points)
        )
        return state_deviations @ state_deviations.T + model.state_noise.covariance

    def correct_state_covariance(
        self,
        filter_gain: Float64Matrix,
        predicted_state_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
        model: StateSpaceModel,
        state: StateBatch,
        observation: ObservationBatch,
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
        return cross_covariance @ np.linalg.pinv(predicted_observation_covariance)

    def eval_observation_cubature_points_offsets(
        self,
        n_cubature_points: int,
        predicted_state_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        return m_utils.svd_sqrt(predicted_state_covariance) * np.sqrt(
            n_cubature_points / 2
        )

    def eval_state_cubature_points_offsets(
        self,
        state_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        state_dim = state_covariance.shape[0]
        return m_utils.svd_sqrt(state_covariance) * np.sqrt(state_dim)


class SRCKF(CKF):
    """
    SR CKF - Square Root Cubature Kalman Filter. Sigma-Point Kalman Filter variant.

    This is a square root form of CKF. The square-root Cubature Kalman filter
    which is also O(L ** 3) for general state estimation and O(L ** 2) for parameter estimation.
    In addition, the square-root forms have the added benefit of numerical stability
    and guaranteed positive semi-definiteness of the state covariances.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def __init__(self):
        super().__init__()

    def __str__(self) -> str:
        return "SR Cubature Kalman Filter (3rd degree)"

    def predict_observation_covariance(
        self,
        model: StateSpaceModel,
        observations: ObservationBatch,
    ) -> CovarianceMatrix:
        _, updated_state_covariance_upper_factor = np.linalg.qr(
            np.hstack((observations, model.observation_noise.covariance)).T
        )
        return updated_state_covariance_upper_factor.T

    def correct_state_covariance(
        self,
        filter_gain: Float64Matrix,
        predicted_state_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
        model: StateSpaceModel,
        state: StateBatch,
        observation: ObservationBatch,
    ) -> CovarianceMatrix:
        _, state_sqrt_covariance_upper_factor = np.linalg.qr(
            np.hstack(
                (
                    state - filter_gain @ observation,
                    filter_gain @ model.observation_noise.covariance,
                )
            ).T
        )
        return state_sqrt_covariance_upper_factor.T

    def eval_filter_gain(
        self,
        cross_covariance: CovarianceMatrix,
        predicted_observation_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        return m_utils.divide_inv(
            (m_utils.divide_inv(cross_covariance, predicted_observation_covariance.T)),
            predicted_observation_covariance,
        )

    def eval_observation_cubature_points_offsets(
        self,
        n_cubature_points: int,
        predicted_state_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        return predicted_state_covariance * np.sqrt(n_cubature_points / 2)

    def predict_state_covariance(
        self,
        n_cubature_points: int,
        model: StateSpaceModel,
        predicted_state_mean: StateVector,
        predicted_cubature_points: StateBatch,
    ) -> CovarianceFactor:
        scaled_state_deviations = (
            predicted_cubature_points
            - np.tile(predicted_state_mean, (1, n_cubature_points))
        ) / np.sqrt(n_cubature_points)

        _, predicted_state_covariance_upper_factor = np.linalg.qr(
            np.hstack((scaled_state_deviations, model.state_noise.covariance)).T
        )
        return predicted_state_covariance_upper_factor.T

    def eval_state_cubature_points_offsets(
        self,
        state_covariance: CovarianceMatrix,
    ) -> Float64Matrix:
        state_dim = state_covariance.shape[0]
        return state_covariance * np.sqrt(state_dim)


class HDLKF(LocalApproximationKalmanFilter, ABC):
    """
    HDLKF - High Degree 'Local Approximation' Kalman Filter.
    Base class that encapsulate filtration process for 'Local approximation' Kalman filters,
    e.g. Cubature filters with degree more than 3, Gauss-Hermite Quadrature Filter.
    The class specifies abstract methods for cubature points and weights calculation.
    These methods must be implemented in the inherited classes.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def estimate(
        self,
        state: StateVector,
        state_covariance: CovarianceMatrix,
        observation: ObservationVector,
        model: StateSpaceModel,
        transition_input: ControlVector,
        observation_input: ControlVector,
    ) -> KalmanFilterResult:
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)].

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue
            (evolution of state, relation between state and observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        standard_points, weights = self.eval_integration_rule(model)

        # Generate state points
        state_covariance_factor = np.linalg.cholesky(state_covariance)
        state_points = state + state_covariance_factor @ standard_points

        # Predict the state
        predicted_state_points = model.transition_func(
            state_points,
            model.state_noise.mean.reshape(-1, 1),
            transition_input,
        )
        predicted_state_mean = predicted_state_points @ weights[:, None]
        state_deviations = predicted_state_points - predicted_state_mean
        predicted_state_covariance = (
            state_deviations * weights
        ) @ state_deviations.T + model.state_noise.covariance

        predicted_state_covariance_factor = np.linalg.cholesky(
            predicted_state_covariance
        )
        predicted_state_deviations = predicted_state_covariance_factor @ standard_points
        predicted_state_points = predicted_state_mean + predicted_state_deviations

        # Predict the observations
        predicted_observations = model.observation_func(
            predicted_state_points,
            model.observation_noise.mean.reshape(-1, 1),
            observation_input,
        )
        predicted_observation_mean = predicted_observations @ weights

        # Correction
        observation_deviations = predicted_observations - predicted_observation_mean
        predicted_observation_covariance = (
            observation_deviations * weights
        ) @ observation_deviations.T + model.observation_noise.covariance
        cross_covariance = (
            predicted_state_deviations * weights @ observation_deviations.T
        )
        filter_gain = cross_covariance @ np.linalg.pinv(
            predicted_observation_covariance
        )

        innovation = model.innovation(observation, predicted_observation_mean)

        updated_state = predicted_state_mean + filter_gain @ innovation
        updated_state_covariance = (
            predicted_state_covariance
            - filter_gain @ predicted_observation_covariance @ filter_gain.T
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

    @abstractmethod
    def eval_integration_rule(
        self,
        model: StateSpaceModel,
    ) -> tuple[IntegrationPoints, IntegrationWeights]: ...


class FDCKF(HDLKF):
    """
    FDCKF - Fifth Degree Cubature Kalman Filter. Fifth degree Spherical Simplex-Radial Cubature Kalman Filter.
    Sigma Point Kalman Filter variant.
    Approximation points (sigma-point) calculated via Spherical Simplex-Radial Rule.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def __str__(self) -> str:
        return "Fifth degree Cubature Kalman Filter"

    def eval_integration_rule(
        self,
        model: StateSpaceModel,
    ) -> tuple[IntegrationPoints, IntegrationWeights]:
        return num_computations.eval_fifth_degree_cubature_rule(model.state_dim)


class CQKF(HDLKF):
    """
    CQKF - Cubature Quadrature Kalman Filter (some kind of High-degree Cubature Kalman Filter).
    Sigma Point Kalman Filter variant.
    Quadrature points are calculated based on Gauss-Laguerre quadrature rule.
    Cubature points are calculated based on intersection with unit hypersphere.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        order - order of Gauss-Laguerre rule.
    """

    def __init__(self, order: int):
        self._order = order

    def __str__(self) -> str:
        return f"Cubature Quadrature Kalman Filter. order: {self._order}"

    def eval_integration_rule(
        self,
        model: StateSpaceModel,
    ) -> tuple[IntegrationPoints, IntegrationWeights]:
        return num_computations.eval_cubature_quadrature_points(
            model.state_dim, self._order
        )


class GHQF(HDLKF):
    """
    GHQF - Gauss-Hermite Quadrature Filter. Sigma Point Kalman Filter variant.
    Estimates state and state covariance for nonlinear systems with
    additive Gaussian noise by linearizing the process and observation functions using
    statistical linear regression (SLR) through a set of
    Gauss-Hermite quadrature points that parameterize the Gaussian density.

    For more details see "Discrete-Time Nonlinear Filtering Algorithms Using Gauss-Hermite Quadrature"
    by Ienkaran Arasaratnam, Simon Haykin and Robert J. Elliott.
    Vol. 95, No. 5, May 2007 | Proceedings of the IEEE

    This filter assumes the following standard state-space model:

    x(k) = f[x(k-1), v(k-1), u1(k-1)];
    z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        order - order of Gauss-Hermite rule.
    """

    def __init__(self, order: int):
        self._order = order

    def __str__(self) -> str:
        return f"Gauss-Hermite Quadrature Filter. order: {self._order}"

    def eval_integration_rule(
        self,
        model: StateSpaceModel,
    ) -> tuple[IntegrationPoints, IntegrationWeights]:
        return num_computations.eval_gauss_hermite_rule(self._order, model.state_dim)


#  todo: not completed yet. there is a bug in points & weights calculation. fix it
class SGHQF(HDLKF):
    """
    SGHQF. Sparse Gauss-Hermite quadrature filter is proposed using a sparse-grid method for multidimensional
    numerical integration in the Bayesian estimation framework.
    The conventional Gauss-Hermite quadrature filter is
    computationally expensive for multidimensional problems,
    because the number of Gauss-Hermite quadrature
    points increases exponentially with the dimension.
    The number of sparse-grid points of the computationally efficient
    sparse Gauss-Hermite quadrature filter, however, increases only polynomially with the dimension.
    In addition, it is proven in this paper that the unscented Kalman filter using
    the suggested optimal parameter is a subset of the sparse
    Gauss-Hermite quadrature filter.

    For more details see:
    "Sparse Gauss-Hermite Quadrature Filter with Application to Spacecraft Attitude Estimation"
    By Bin Jia, Ming Xin and Yang Cheng.
    JOURNAL OF GUIDANCE, CONTROL, AND DYNAMICS Vol. 34, No. 2, March-April 2011

    This filter assumes the following standard state-space model:

    x(k) = f[x(k-1), v(k-1), u1(k-1)];
    z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        order - order of Gauss-Hermite rule;
        manner - increase manner: L, 2*L-1, 2^L-1.
    """

    def __init__(
        self,
        order: int,
        manner: int,
    ):
        """
        Parameters
        ----------
        order : int
        manner : int
        """
        self._order = order
        self._manner = manner

    def __str__(self) -> str:
        return f"Sparse Gauss-Hermite Quadrature Filter. order: {self._order}; manner: {self._manner}"

    def eval_integration_rule(
        self,
        model: StateSpaceModel,
    ) -> tuple[IntegrationPoints, IntegrationWeights]:
        raise ValueError("Not implemented yet")


@dataclass
class WeightedParticleSet:
    __slots__ = ("_particles", "_weights")

    def __init__(
        self,
        particles: ParticleMatrix,
        weights: ParticleWeights,
    ) -> None:
        if particles.shape[0] == 0 or particles.shape[1] == 0:
            raise ValueError(
                "particles must contain at least one state dimension and one particle"
            )

        if particles.shape[1] != weights.size:
            raise ValueError(
                f"Expected {particles.shape[1]} weights, got {weights.size}"
            )

        self._particles = particles.copy()
        self._weights = weights.copy()

        self._particles.setflags(write=False)
        self._weights.setflags(write=False)

    @property
    def particles(self) -> ParticleMatrix:
        return self._particles.view()

    @property
    def weights(self) -> ParticleWeights:
        return self._weights.view()

    @property
    def n_particles(self) -> int:
        return self.particles.shape[1]


class ResamplingMethod(Enum):
    unknown = 0
    residual = 1
    residual2 = 2
    multinomial = 3
    stratified = 4
    systematic = 5


class Resampler(ABC):
    @abstractmethod
    def resample(
        self,
        weights: ParticleWeights,
    ) -> ParticleIndices:
        """
        Performs the resampling algorithm used by particle filters
        :param weights: list-nike of float list of weights as floats
        :return: np array of ints, array of indexes into the weights defining the resample.
        """

    @staticmethod
    def resolve(method: ResamplingMethod) -> Resampler:
        match method:
            case ResamplingMethod.unknown:
                raise ValueError(f"{ResamplingMethod.unknown} is not supported.")
            case ResamplingMethod.residual:
                return ResidualResampler()
            case ResamplingMethod.residual2:
                return ResidualResampler2()
            case ResamplingMethod.multinomial:
                return MultinomialResampler()
            case ResamplingMethod.stratified:
                return StratifiedResampler()
            case ResamplingMethod.systematic:
                return SystematicResampler()
            case _:
                raise ValueError(f"Resampling method [{method}] is not supported.")


class ResidualResampler(Resampler):
    def __str__(self) -> str:
        return "Residual resampler"

    def resample(
        self,
        weights: ParticleWeights,
    ) -> ParticleIndices:
        n_particles = weights.size
        input_indices = range(n_particles)

        ancestor_indices = np.zeros(n_particles, dtype=int)

        # first integer part
        residual_weights = n_particles * weights
        offspring_counts = np.fix(residual_weights)

        n_residual_samples = int(n_particles - np.sum(offspring_counts))

        if n_residual_samples > 0:
            residual_weights = (
                residual_weights - offspring_counts
            ) / n_residual_samples
            cum_dist = np.cumsum(residual_weights)

            # generate N (N = residual_particles_count) ordered random variables uniformly distributed in [0, 1]
            exponents = 1.0 / np.arange(n_residual_samples, 0, -1)
            sorted_uniforms = np.cumprod(
                np.random.rand(n_residual_samples) ** exponents
            )[::-1]

            j = 0
            for i in range(n_residual_samples):
                while sorted_uniforms[i] > cum_dist[j]:
                    j += 1

                offspring_counts[j] += 1

        idx = 0
        for i in range(n_particles):
            if offspring_counts[i] > 0:
                upper_index = int(idx + offspring_counts[i])
                for j in range(idx, upper_index):
                    ancestor_indices[j] = int(input_indices[i])

            idx = int(idx + offspring_counts[i])

        return ancestor_indices


class ResidualResampler2(Resampler):
    def __str__(self) -> str:
        return "Residual resampler 2"

    """
    Residual resample algorithm.
    J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic systems.
    Journal of the American Statistical Association, 93(443):1032–1044, 1998.
    """

    def resample(
        self,
        weights: ParticleWeights,
    ) -> ParticleIndices:
        n_particles = len(weights)

        # take bool(N*w) copies of each weight to find out what particle will survive after re-sampling,
        # particle will survive if weights is 'big enough'.
        # the procedure  ensures particles with the same weight are drawn uniformly
        deterministic_counts = np.floor(n_particles * np.asarray(weights)).astype(int)
        ancestor_indices = np.zeros(n_particles, dtype=int)
        k = 0
        for i in range(n_particles):
            for _ in range(deterministic_counts[i]):
                ancestor_indices[k] = i
                k += 1

        if k == n_particles:
            return ancestor_indices

        residual_weights = n_particles * weights - deterministic_counts
        residual_weights /= residual_weights.sum()

        cumulative_weights = m_utils.cum_sum_with_last_equal_to_one(residual_weights)
        ancestor_indices[k:] = np.searchsorted(
            cumulative_weights,
            random(n_particles - k),
            side="right",
        )

        return ancestor_indices


class MultinomialResampler(Resampler):
    """
    The naive form of roulette sampling where we compute the cumulative sum of the weights and then use binary search
    to select the resampled point based on a uniformly distributed random number.
    """

    def __str__(self) -> str:
        return "Multinomial resampler"

    def resample(
        self,
        weights: ParticleWeights,
    ) -> ParticleIndices:
        cumulative_weights = m_utils.cum_sum_with_last_equal_to_one(weights)
        return np.searchsorted(cumulative_weights, random(len(weights)))


class StratifiedResampler(Resampler):
    """
    Performs the stratified resampling algorithm used by particle filters.
    This algorithm makes selections relatively uniformly across the particles.
    It divides the cumulative sum of the weights into N equal divisions,
    and then selects one particle randomly from each division.
    This guarantees that each sample is between 0 and 2/N apart.
    """

    def __str__(self) -> str:
        return "Stratified resampler"

    def resample(
        self,
        weights: ParticleWeights,
    ) -> ParticleIndices:
        n_particles = len(weights)
        # make N subdivisions, and chose a random position within each one
        cdf_positions = (random(n_particles) + range(n_particles)) / n_particles
        return _inverse_cdf_indices(weights, cdf_positions)


class SystematicResampler(Resampler):
    """
    Performs the systematic resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions.
    A single random offset is used to choose where to sample from for all divisions.
    This guarantees that every sample is exactly 1/N apart.
    """

    def __str__(self) -> str:
        return "Systemic resampler"

    def resample(
        self,
        weights: ParticleWeights,
    ) -> ParticleIndices:
        n_particles = len(weights)

        cdf_positions = (random() + np.arange(n_particles)) / n_particles
        return _inverse_cdf_indices(weights, cdf_positions)


def _inverse_cdf_indices(
    weights: ParticleWeights,
    positions: Float64Matrix,
) -> ParticleIndices:
    n_particles = len(weights)
    ancestor_indices = np.zeros(n_particles, dtype=int)
    cumulative_weights = np.cumsum(weights)

    i = 0
    j = 0
    while i < n_particles:
        if positions[i] < cumulative_weights[j]:
            ancestor_indices[i] = j
            i += 1
        else:
            j += 1

    return ancestor_indices


type ParticleFilterResult = PdfApproximationFilterResult[WeightedParticleSet]


class BPF(PdfApproximationFilter[WeightedParticleSet]):
    """
    Generic Bootstrap Particle filter / Particle Filter.
    This filter is also known as the 'Bootstrap Particle Filter' or the 'Condensation Algorithm'.
    Particle filters implement the prediction-updating updates in an approximate manner.
    The samples from the distribution are represented by a set of particles;
    each particle has a likelihood weight assigned to it that represents the probability of that particle being sampled from
    the probability density function.

    This filter assumes the following standard state-space model:

    x(k) = f[x(k-1), v(k-1), u1(k-1)];
    z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        estimate_type - EstimateType, estimate type : 'mean', 'median', etc;
        resample_threshold - float, if the ratio of the 'effective particle set size'
            to the total number of particles drop below this threshold  i.e.,
            (effective size / total number of particles) < resample threshold the particles will be re-sampled
            ("effective size" is always less than or equal to the total number of particles).
        resample_strategy - ResampleStrategy, resampling strategy.
    """

    def __init__(
        self,
        ess_threshold_ratio: float,
        point_estimate_method: PointEstimateMethod = PointEstimateMethod.mean,
        resampler: Resampler | None = None,
    ):
        super().__init__(point_estimate_method)
        self._ess_threshold_ratio = ess_threshold_ratio
        self._resampler = resampler if resampler is not None else ResidualResampler()

    def __str__(self) -> str:
        return (
            f"Bootstrap Particle Filter: "
            f"ESS threshold ratio: {self._ess_threshold_ratio}; "
            f"Point Estimate Method: {self._point_estimate_method}; "
            f"Resampler: {self._resampler}"
        )

    def estimate(
        self,
        approximation: WeightedParticleSet,
        observation: ObservationVector,
        model: StateSpaceModel,
        transition_input: ControlVector,
        observation_input: ControlVector,
    ) -> tuple[StateVector, WeightedParticleSet]:
        """
        Estimates state mean (x(k))of dynamic system at time k based on
        noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)].

        """
        state_noise_samples = model.state_noise.sample(approximation.n_particles)
        predicted_particles = model.transition_func(
            approximation.particles,
            state_noise_samples,
            transition_input,
        )

        observation_likelihoods = model.likelihood(
            observation,
            predicted_particles,
            observation_input,
        )
        weights = np.multiply(approximation.weights, observation_likelihoods)
        weights /= np.sum(weights, axis=1)

        particles_set = self._resample_if_needed(
            WeightedParticleSet(predicted_particles, weights)
        )
        updated_state = self._estimate_state_from_particles(particles_set)

        return updated_state, particles_set

    def _estimate_state_from_particles(
        self,
        particle_set: WeightedParticleSet,
    ) -> StateVector:
        if self._point_estimate_method is PointEstimateMethod.mean:
            return np.average(
                particle_set.particles,
                axis=1,
                weights=particle_set.weights,
            )

        if self._point_estimate_method is PointEstimateMethod.median:
            return np.quantile(
                particle_set.particles,
                0.5,
                axis=1,
                weights=particle_set.weights,
                method="inverted_cdf",
            )

        raise ValueError(f"Unknown estimate type: {self._point_estimate_method}")

    def _resample_if_needed(
        self,
        particle_set: WeightedParticleSet,
    ) -> WeightedParticleSet:
        ess_threshold = np.round(self._ess_threshold_ratio * particle_set.n_particles)
        effective_sample_size = 1 / np.sum(np.power(particle_set.weights, 2), axis=1)

        if effective_sample_size >= ess_threshold:
            return WeightedParticleSet(particle_set.particles, particle_set.weights)

        ancestor_indices = self._resampler.resample(
            np.array(particle_set.weights).flatten()
        )
        particles = particle_set.particles[:, ancestor_indices]

        weights = np.full((particle_set.n_particles), 1.0 / (particle_set.n_particles))

        return WeightedParticleSet(particles, weights)


type GaussianSumParticleFilterResult = PdfApproximationFilterResult[GaussianMixture]


class GSPF(PdfApproximationFilter[GaussianMixture]):
    """
    Gaussian Sum Particle Filter.
    The filter approximate the filtering and predictive distributions
    by weighted Gaussian mixtures and are basically banks of Gaussian particle filters.
    Then, we extend the use of Gaussian particle filters and Gaussian sum particle filters to
    dynamic state space (DSS) models with non-Gaussian noise.
    With non-Gaussian noise approximated by Gaussian mixtures, the non-Gaussian
    noise models are approximated by banks of Gaussian noise models,
    and Gaussian mixture filters are developed using
    algorithms developed for Gaussian noise models.

    For more details please see:
        Jayesh H. Kotecha and Petar M. Djuric, "Gaussian Sum Particle Filtering for Dynamic State Space Models",
        Proceedings of ICASSP-2001, Salt Lake City, Utah, May 2001.

    This filter assumes the following standard state-space model:

    x(k) = f[x(k-1), v(k-1), u1(k-1)];
    z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        estimate_type - EstimateType, estimate type : 'mean', 'median', etc;
        resample_threshold - float, if the ratio of the 'effective particle set size' to the total number of particles
            drop below this threshold  i.e.
            (effective size / total number of particles) < resample threshold
            the particles will be re-sampled.
            ("effective size" is always less than or equal to the total number of particles).
        resample_strategy - ResampleStrategy, resampling strategy.
    """

    def __init__(
        self,
        n_particles_per_component: int,
        point_estimate_method: PointEstimateMethod = PointEstimateMethod.mean,
        resampler: Resampler | None = None,
    ):
        super().__init__(point_estimate_method)
        self._n_particles_per_component = n_particles_per_component
        self._resampler = resampler if resampler is not None else ResidualResampler()

    def __str__(self) -> str:
        return (
            f"Gaussian Sum Particle Filter: "
            f"Point Estimate Method: {self._point_estimate_method}; "
            f"Resampler: {self._resampler}; "
            f"Particles per component: {self._n_particles_per_component}"
        )

    def estimate(
        self,
        approximation: GaussianMixture,
        observation: ObservationVector,
        model: StateSpaceModel,
        transition_input: ControlVector,
        observation_input: ControlVector,
    ) -> GaussianSumParticleFilterResult:
        n_particles = self._n_particles_per_component
        state_noise = model.state_noise
        n_components = approximation.n_components * state_noise.n_components

        # Time update (prediction).
        component_particles = approximation.sqrt_covariances @ np.random.randn(
            approximation.n_components,
            model.state_dim,
            n_particles,
        )
        component_particles += approximation.means[..., None]

        standard_normal_samples = np.random.randn(
            state_noise.n_components,
            approximation.n_components,
            state_noise.dim,
            n_particles,
        )
        process_noise_samples = (
            state_noise.covariance[:, None] @ standard_normal_samples
        )
        process_noise_samples += state_noise.mean[:, None, :, None]

        predicted_component_particles = np.stack(
            [
                model.transition_func(
                    component_particles[g],
                    process_noise_samples[k, g],
                    transition_input,
                )
                for k in range(state_noise.n_components)
                for g in range(approximation.n_components)
            ]
        )

        component_weights = np.outer(state_noise.weights, approximation.weights).ravel()
        component_weights /= component_weights.sum()

        component_means = predicted_component_particles.mean(axis=-1)
        predicted_state_deviations = (
            predicted_component_particles - component_means[..., None]
        )

        qr_upper_factors = cast(
            Float64Matrix,
            np.linalg.qr(
                predicted_state_deviations.swapaxes(-1, -2),
                mode="r",
            ),
        )
        component_covariance_factors = qr_upper_factors.swapaxes(-1, -2)
        component_covariance_factors /= np.sqrt(n_particles - 1)

        # Measurement update (correction).
        standard_normal_samples = np.random.randn(
            n_components, model.state_dim, n_particles
        )
        proposal_particles = component_covariance_factors @ standard_normal_samples
        proposal_particles += component_means[..., None]

        observation_batch = np.broadcast_to(
            observation, (model.observation_dim, n_particles)
        )
        likelihoods = np.stack(
            [
                model.likelihood(
                    observation_batch,
                    proposal_particles[g],
                    observation_input,
                ).A1
                for g in range(n_components)
            ]
        )

        component_likelihood_sums = likelihoods.sum(axis=-1)
        particle_weights = likelihoods / component_likelihood_sums[:, None]

        component_means = (proposal_particles @ particle_weights[..., None])[..., 0]

        sqrt_particle_weights = np.sqrt(particle_weights)
        weighted_state_deviations = proposal_particles - component_means[..., None]
        weighted_state_deviations *= sqrt_particle_weights[:, None, :]

        qr_upper_factors = cast(
            Float64Matrix,
            np.linalg.qr(
                weighted_state_deviations.swapaxes(-1, -2),
                mode="r",
            ),
        )
        component_covariance_factors = qr_upper_factors.swapaxes(-1, -2)

        component_weights *= component_likelihood_sums
        component_weights /= component_weights.sum()

        if self._point_estimate_method is not PointEstimateMethod.mean:
            raise RuntimeError(
                f"Unsupported point estimate method: {self._point_estimate_method}"
            )

        estimate = component_means.T @ component_weights[:, None]

        idx = self._resample_mixture_components_if_needed(
            n_components,
            approximation.n_components,
            component_weights,
        )
        approximation.set_params(
            component_means[idx],
            component_covariance_factors[idx],
            np.full(
                approximation.n_components,
                1.0 / approximation.n_components,
                dtype=np.float64,
            ),
            CovarianceType.sqrt,
        )

        return estimate, approximation

    def _resample_mixture_components_if_needed(
        self,
        n_components: int,
        n_x_components: int,
        weights: ParticleWeights,
    ) -> ParticleIndices:
        resample_idx = self._resampler.resample(weights)
        r_idx = np.argsort(np.random.rand(n_components))[:n_x_components]
        return resample_idx[r_idx]
