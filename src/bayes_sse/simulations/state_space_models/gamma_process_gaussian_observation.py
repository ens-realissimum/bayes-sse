from typing import overload

import numpy as np

import bayes_sse.inference.stochastic_models.stochastic_models as sm
from bayes_sse.core.covariance_adapdation import CovarianceMatrixAdaptation
from bayes_sse.core.linearization_type import LinearizationType
from bayes_sse.inference.gssm import StateSpaceModel
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType
from bayes_sse.inference.stochastic_models.noise_type import NoiseType

from ...typing import (
    ControlVector,
    Float64FlatVector,
    JacobianMatrix,
    NoiseBatch,
    NoiseVector,
    ObservationBatch,
    ObservationVector,
    StateBatch,
    StateVector,
)


class GammaProcessWithGaussianObservationGssm(StateSpaceModel):
    """
    Describe general state space model of gamma distributed state vector and
    observation with Gaussian noise.
    Relation between state and observation is non-linear.
    State space function is non-linear.
    """

    def __init__(
        self,
        omega: float,
        phi: float,
        state_noise: sm.StochasticModel,
        observation_noise: sm.StochasticModel,
    ):
        super().__init__(state_noise, observation_noise)

        self._omega = omega
        self._phi = phi

    def __str__(self):
        return (
            f"Gamma distributed state vector (dim: {self.state_dim}) and "
            f"observation (dim: {self.observation_dim}) with Gaussian noise"
        )

    @property
    def type(self) -> str:
        return "Model is a scalar nonlinear system with Gamma process and observation with Gaussian noise."

    @property
    def tag(self) -> str:
        return "Nonlinear system with Gamma process and Gaussian observation noise"

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def observation_dim(self) -> int:
        return 1

    @property
    def control_state_dim(self) -> int | None:
        return 1

    @property
    def control_observation_dim(self) -> int | None:
        return 1

    @overload
    def transition_func(
        self,
        state: StateVector,
        state_noise: NoiseVector | None = None,
        transition_input: ControlVector = None,
    ) -> StateVector: ...

    @overload
    def transition_func(
        self,
        state: StateBatch,
        state_noise: NoiseBatch | None = None,
        transition_input: ControlVector = None,
    ) -> StateBatch: ...

    def transition_func(
        self,
        state: StateBatch,
        state_noise: NoiseBatch | None = None,
        transition_input: ControlVector = None,
    ) -> StateBatch:
        u = 1 if transition_input is None else transition_input
        v = 0 if state_noise is None else state_noise
        return 1 + np.sin(self._omega * np.pi * u) + self._phi * state + v

    @overload
    def observation_func(
        self,
        state: StateVector,
        observation_noise: NoiseVector | None = None,
        observation_input: ControlVector = None,
    ) -> ObservationVector: ...

    @overload
    def observation_func(
        self,
        state: StateBatch,
        observation_noise: NoiseBatch | None = None,
        observation_input: ControlVector = None,
    ) -> ObservationBatch: ...

    def observation_func(
        self,
        state: StateBatch,
        observation_noise: NoiseBatch | None = None,
        observation_input: ControlVector = None,
    ) -> ObservationBatch:
        x = np.atleast_2d(state)
        u = 0 if observation_input is None else observation_input

        z = self._phi * np.power(x, 2) + u

        if observation_noise is not None:
            z += observation_noise

        return z

    def prior(
        self,
        next_state: StateBatch,
        state: StateBatch,
        transition_input: ControlVector = None,
    ) -> Float64FlatVector:
        x_deviation = next_state - self.transition_func(state, None, transition_input)
        return self.state_noise.likelihood(x_deviation)

    def likelihood(
        self,
        observation: ObservationBatch,
        state: StateBatch,
        observation_input: ControlVector = None,
    ) -> Float64FlatVector:
        z_deviation = observation - self.observation_func(
            state, None, observation_input
        )
        return np.asarray(self.observation_noise.likelihood(z_deviation))

    def linearize(
        self,
        lin_type: LinearizationType,
        state: StateVector | None,
        *,
        state_noise: NoiseVector | None = None,
        observation_noise: NoiseVector | None = None,
        control_x: ControlVector = None,
        control_z: ControlVector = None,
    ) -> JacobianMatrix:
        if lin_type is LinearizationType.F:
            return np.array([[self._phi]], dtype=np.float64)

        if lin_type in (LinearizationType.D, LinearizationType.G, LinearizationType.H):
            return np.ones((1, 1), dtype=np.float64)

        if lin_type in (LinearizationType.B, LinearizationType.JFW):
            u = 1.0 if control_x is None else control_x.item()
            slope = np.pi * np.cos(self._omega * np.pi * u)

            if lin_type is LinearizationType.B:
                return np.array([[self._omega * slope]], dtype=np.float64)

            if state is None:
                raise ValueError("JFW requires a state.")
            return np.array([[u * slope, state.item()]], dtype=np.float64)

        if lin_type in (LinearizationType.C, LinearizationType.JHW):
            if state is None:
                raise ValueError(f"{lin_type.name} requires a state.")

            x = state.item()
            if lin_type is LinearizationType.C:
                return np.array([[2 * self._phi * x]], dtype=np.float64)
            return np.array([[0.0, x**2]], dtype=np.float64)

        raise ValueError(f"Unknown linearization type: {lin_type}")

    def set_params(self, **kwargs) -> None:
        if "omega" in kwargs:
            self._omega = kwargs["omega"]

        if "phi" in kwargs:
            self._phi = kwargs["phi"]

    def clone(
        self,
        state_noise: sm.StochasticModel,
        observation_noise: sm.StochasticModel,
        reconcile_strategy: CovarianceMatrixAdaptation | None,
    ) -> StateSpaceModel:
        return GammaProcessWithGaussianObservationGssm(
            self._omega, self._phi, state_noise, observation_noise
        )


def build(
    omega: float,
    phi: float,
    shape: float,
    scale: float,
    n_mean: float,
    n_cov: float,
    n_cov_type: CovarianceType,
) -> GammaProcessWithGaussianObservationGssm:
    x_noise = sm.build_stochastic_process(NoiseType.gamma, shape=shape, scale=scale)
    z_noise = sm.build_stochastic_process(
        NoiseType.gaussian,
        mean=np.asarray(n_mean),
        covariance=np.atleast_2d(n_cov),
        covariance_type=n_cov_type,
    )

    return GammaProcessWithGaussianObservationGssm(omega, phi, x_noise, z_noise)
