from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol

from bayes_sse.inference.gssm import StateSpaceModel
from bayes_sse.typing import (
    ControlVector,
    CovarianceMatrix,
    Float64Matrix,
    Float64Vector,
    ObservationBatch,
    ObservationVector,
    StateBatch,
    StateVector,
)


class KalmanFilterState:
    """Predictions and correction terms from one call to ``estimate``.

    :param predicted_state_mean: State mean before the measurement update.
    :param predicted_state_covariance: Predicted state covariance or its factor,
        depending on the filter.
    :param predicted_observation_mean: Measurement predicted from the prior state estimate.
    :param predicted_observation_covariance: Innovation covariance or its factor,
        depending on the filter.
    :param innovation: Measurement residual returned by the model's ``innovation`` method.
    :param filter_gain: Kalman gain applied to the innovation.
    """

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
    """Interface for linear and local approximation Kalman filters.

    This includes filters that approximate nonlinear transformations,
    such as sigma-point filters and the extended Kalman filter.

    Another way to estimate the state is to approximate its probability density
    function (PDF). This is used in particle filters and their variations.
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
    ) -> KalmanFilterResult:
        """Advance the state estimate from time k-1 to k.

        :param state: Posterior state mean at k-1.
        :param state_covariance: State covariance at k-1, or its square-root
            factor for square-root filters.
        :param observation: Measurement at k.
        :param model: State-space model for the transition and measurement.
        :param transition_input: Control for the transition from k-1 to k, or ``None``.
        :param observation_input: Control for the measurement at k, or ``None``.
        :return: The updated state mean, updated covariance or its factor, and
            a ``KalmanFilterState`` containing this step's predictions, innovation, and gain.
        """
        ...


type PdfApproximationFilterResult[ApproximationT] = tuple[
    StateVector,
    ApproximationT,
]


class PdfApproximationFilter[ApproximationT](ABC):
    """Base class for filters that approximate predictive distributions.

    These filters approximate probability density functions (PDFs) after
    nonlinear transformations. Particle filters are an example.
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
        """Estimate the state x(k) of a dynamic system at time k.

        The filter uses the noisy observation z(k) and assumes the following
        standard state-space model::

            x(k) = f[x(k-1), v(k-1), u1(k-1)]
            z(k) = h[x(k), n(k), u2(k)]

        :param approximation: Approximation of the state distribution at time k-1.
        :param observation: Noisy observation at time k, i.e., z(k).
        :param model: Inference state-space model, which describes state evolution,
            the relationship between the state and the observation, and the state
            and observation noise.
        :param transition_input: Exogenous input to the state transition function
            at time k-1, i.e., u1(k-1).
        :param observation_input: Exogenous input to the observation function
            at time k, i.e., u2(k).
        :return: A tuple containing the state point estimate at time k and the
            updated distribution approximation.
        """
        ...


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
