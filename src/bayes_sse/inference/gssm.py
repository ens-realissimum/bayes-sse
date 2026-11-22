from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import overload

import numpy as np

import bayes_sse.inference.stochastic_models.stochastic_models as sm
from bayes_sse.core.covariance_adapdation import CovarianceMatrixAdaptation
from bayes_sse.core.linearization_type import LinearizationType
from bayes_sse.inference.stochastic_models.stochastic_models import (
    Gaussian,
    StochasticModel,
)
from bayes_sse.typing import (
    ControlVector,
    Float64FlatVector,
    JacobianMatrix,
    NoiseBatch,
    NoiseVector,
    ObservationBatch,
    ObservationJacobians,
    ObservationVector,
    StateBatch,
    StateVector,
    TransitionJacobians,
)


class StateSpaceModel(ABC):
    """
    Model which allows to formulate filtration issue in bayesian estimation terminology.
    """

    def __init__(
        self,
        state_noise: StochasticModel,
        observation_noise: StochasticModel,
        _covariance_adaptation: CovarianceMatrixAdaptation | None = None,
    ):
        self._state_noise = deepcopy(state_noise)
        self._observation_noise = deepcopy(observation_noise)
        self._covariance_adaptation = _covariance_adaptation

    @property
    @abstractmethod
    def type(self) -> str:
        return "General state space model"

    @property
    @abstractmethod
    def tag(self) -> str:
        return ""

    @property
    @abstractmethod
    def state_dim(self) -> int: ...

    @property
    @abstractmethod
    def observation_dim(self) -> int: ...

    @property
    @abstractmethod
    def transition_input_dim(self) -> int | None:
        return None

    @property
    @abstractmethod
    def observation_input_dim(self) -> int | None:
        return None

    @property
    def state_noise(self) -> StochasticModel:
        return self._state_noise

    @property
    def observation_noise(self) -> StochasticModel:
        return self._observation_noise

    @property
    def covariance_adaptation(self) -> CovarianceMatrixAdaptation | None:
        return self._covariance_adaptation

    @overload
    @abstractmethod
    def transition_func(
        self,
        state: StateVector,
        state_noise: NoiseVector | None = None,
        transition_input: ControlVector = None,
    ) -> StateVector: ...

    @overload
    @abstractmethod
    def transition_func(
        self,
        state: StateBatch,
        state_noise: NoiseBatch | None = None,
        transition_input: ControlVector = None,
    ) -> StateBatch:
        """
        State transition function. Calculate (predict) value of state space vector at time = k based on value of
        state space vector at time = k-1.
        :return: value of state space vector
        """
        ...

    @overload
    @abstractmethod
    def observation_func(
        self,
        state: StateVector,
        observation_noise: NoiseVector | None = None,
        observation_input: ControlVector = None,
    ) -> ObservationVector: ...

    @overload
    @abstractmethod
    def observation_func(
        self,
        state: StateBatch,
        observation_noise: NoiseBatch | None = None,
        observation_input: ControlVector = None,
    ) -> ObservationBatch:
        """
        Observation function (mapping state to observation)
        :return: observation calculated based on value of state space vector at time = k.
        """
        ...

    @abstractmethod
    def prior(
        self,
        next_state: StateBatch,
        state: StateBatch,
        transition_input: ControlVector = None,
    ) -> Float64FlatVector:
        """
        Function to calculate that calculates P(x(k)|x(k-1)), ie probability that value of state space vector
        at time = k equal to 'x(k)' if at previous time the value has been equal to x(k-1).
        :return: probability that value of state space vector at time = k equal to 'x(k)' if at previous time
        the value has been equal to x(k-1).
        """

    @abstractmethod
    def likelihood(
        self,
        observation: ObservationBatch,
        state: StateBatch,
        observation_input: ControlVector = None,
    ) -> Float64FlatVector:
        """
        Function to calculate the observation likelihood function that calculates p(z(k)|x(k))
        :return: calculated likelihood.
        """

    @staticmethod
    def innovation(
        observation: ObservationVector,
        predicted_observation: ObservationVector,
    ) -> ObservationVector:
        """
        Function to calculate the innovation model function that calculates the difference between the output
        of the observation function (observation_func) and the actual 'real-world' measurement/observation
        of that signal.
        :param observation: 'real-world' measurement/observation of that signal at time = k
        :param observation_predicted: the output of the observation function (observation_func) at time = k.
        :return: difference between the output of the observation function and the actual 'real-world' observation.
        """
        return observation - predicted_observation

    @abstractmethod
    def linearize(
        self,
        lin_type: LinearizationType,
        state: StateVector | None,
        *,
        state_noise: NoiseVector | None = None,
        observation_noise: NoiseVector | None = None,
        transition_input: ControlVector = None,
        observation_input: ControlVector = None,
    ) -> JacobianMatrix:
        """Compute the requested Jacobian at the supplied state and inputs."""
        ...

    def transition_jacobians(
        self,
        state: StateVector | None,
        *,
        state_noise: NoiseVector | None = None,
        observation_noise: NoiseVector | None = None,
        transition_input: ControlVector = None,
        observation_input: ControlVector = None,
    ) -> TransitionJacobians:
        """Return F and G for the prediction step."""
        linearize_at_state = partial(
            self.linearize,
            state=state,
            state_noise=state_noise,
            observation_noise=observation_noise,
            transition_input=transition_input,
            observation_input=observation_input,
        )
        return (
            linearize_at_state(LinearizationType.F),
            linearize_at_state(LinearizationType.G),
        )

    def observation_jacobians(
        self,
        state: StateVector | None,
        *,
        state_noise: NoiseVector | None = None,
        observation_noise: NoiseVector | None = None,
        transition_input: ControlVector = None,
        observation_input: ControlVector = None,
    ) -> ObservationJacobians:
        """Return C and H for the measurement update."""
        linearize_at_state = partial(
            self.linearize,
            state=state,
            state_noise=state_noise,
            observation_noise=observation_noise,
            transition_input=transition_input,
            observation_input=observation_input,
        )
        return (
            linearize_at_state(LinearizationType.C),
            linearize_at_state(LinearizationType.H),
        )

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """
        Allow to update specific model parameters. Do nothing by default
        :return:
        """
        ...

    @abstractmethod
    def clone(
        self,
        state_noise: StochasticModel,
        observation_noise: StochasticModel,
        reconcile_strategy: CovarianceMatrixAdaptation | None,
    ) -> StateSpaceModel: ...

    def adapt_covariance(self, filter_cov: np.ndarray, **kwargs) -> None:
        if self._covariance_adaptation is None:
            return

        if not isinstance(self._state_noise, Gaussian):
            raise TypeError(
                "Adaptation is allowed only for Gaussian (not mixture) stochastic process"
            )

        cov = self._covariance_adaptation.adapt(self._state_noise, filter_cov, **kwargs)

        self._state_noise = Gaussian(
            mean=self._state_noise.mean,
            covariance=cov,
            covariance_type=self._state_noise.covariance_type,
        )

    @property
    def observation_noise_dim(self) -> int:
        return self.observation_noise.dim

    @property
    def state_noise_dim(self) -> int:
        return self.state_noise.dim

    def set_state_noise(self, state_noise: sm.StochasticModel) -> StateSpaceModel:
        return self.clone(
            state_noise, self.observation_noise, self.covariance_adaptation
        )

    def set_observation_noise(
        self,
        observation_noise: sm.StochasticModel,
    ) -> StateSpaceModel:
        return self.clone(
            self.state_noise, observation_noise, self.covariance_adaptation
        )
