from typing import NoReturn, Optional

import numpy as np

import bayes_sse.inference.stochastic_models.stochastic_models as sm
from bayes_sse.inference.gssm import StateSpaceModel
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType
from bayes_sse.inference.stochastic_models.noise_type import NoiseType
from bayes_sse.core.covariance_adapdation import CovarianceMatrixAdaptation
from bayes_sse.core.linearization_type import LinearizationType


class GammaProcessWithGaussianObservationGssm(StateSpaceModel):
    """
    Describe general state space model of gamma distributed state vector and
    observation with Gaussian noise.
    Relation between state and observation is non-linear.
    State space function is non-linear.
    """

    def __init__(self, omega: float, phi: float, state_noise: sm.StochasticModel,
                 observation_noise: sm.StochasticModel):
        super().__init__(state_noise, observation_noise)

        self._omega = omega
        self._phi = phi

    def __str__(self):
        return f"Gamma distributed state vector (dim: {self.state_dim}) and " \
               f"observation (dim: {self.observation_dim}) with Gaussian noise"

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
    def control_state_dim(self) -> Optional[int]:
        return 1

    @property
    def control_observation_dim(self) -> Optional[int]:
        return 1

    def transition_func(
            self,
            state: np.ndarray,
            state_noise: Optional[np.ndarray] = None,
            control_vect: Optional[np.ndarray] = None
    ) -> np.ndarray:
        u = 1 if control_vect is None else control_vect
        v = 0 if state_noise is None else state_noise

        return 1 + np.sin(self._omega * np.pi * u) + self._phi * state + v

    def observation_func(
            self,
            state: np.ndarray,
            observation_noise: Optional[np.ndarray] = None,
            control_vect: Optional[np.ndarray] = None
    ) -> np.ndarray:
        x = np.atleast_2d(state)
        u = 0 if control_vect is None else control_vect

        z = self._phi * np.power(x, 2) + u

        if observation_noise is not None:
            z += observation_noise

        return z

    def prior(
            self,
            next_state: np.ndarray,
            state: np.ndarray,
            control_x: Optional[np.ndarray] = None
    ) -> np.ndarray:
        x_deviation = next_state - self.transition_func(state, None, control_x)
        return self.state_noise.likelihood(x_deviation)

    def likelihood(
            self,
            observation: np.ndarray,
            state: np.ndarray,
            control_z: Optional[np.ndarray] = None
    ) -> np.matrix:
        z_deviation = observation - self.observation_func(state, None, control_z)
        return np.matrix(self.observation_noise.likelihood(z_deviation))

    def linearize(
            self,
            lin_type: LinearizationType,
            state: Optional[np.ndarray],
            state_noise: Optional[np.ndarray],
            observation_noise: Optional[np.ndarray],
            control_x: Optional[np.ndarray],
            control_z: Optional[np.ndarray]
    ) -> np.ndarray:
        if lin_type == LinearizationType.F:
            return np.atleast_2d([self._phi])
        elif lin_type in (LinearizationType.B, LinearizationType.D):
            return np.atleast_2d([])
        elif lin_type == LinearizationType.C:
            return np.atleast_2d([2 * self._phi * state if control_z <= 30 else self._phi])
        elif lin_type in (LinearizationType.G, LinearizationType.H):
            return np.atleast_2d([1])
        elif lin_type == LinearizationType.JFW:
            return np.atleast_2d([np.cos(self._omega * np.pi * control_x) * np.pi * control_x, state])
        elif lin_type == LinearizationType.JHW:
            jhw = [0, state ** 2] if control_z <= 30 else [0, state]
            return np.atleast_2d(jhw)
        else:
            raise Exception(f"Unknown value of lin_type: {lin_type.name}")

    def set_params(self, **kwargs) -> NoReturn:
        if "omega" in kwargs:
            self._omega = kwargs["omega"]

        if "phi" in kwargs:
            self._phi = kwargs["phi"]

    def clone(
            self,
            state_noise: sm.StochasticModel,
            observation_noise: sm.StochasticModel,
            reconcile_strategy: CovarianceMatrixAdaptation
    ) -> StateSpaceModel:
        return GammaProcessWithGaussianObservationGssm(self._omega, self._phi, state_noise, observation_noise)


def build(
        omega: float,
        phi: float,
        shape: float,
        scale: float,
        n_mean: float,
        n_cov: float,
        n_cov_type: CovarianceType
) -> GammaProcessWithGaussianObservationGssm:
    x_noise = sm.build_stochastic_process(NoiseType.gamma, shape=shape, scale=scale)
    z_noise = sm.build_stochastic_process(
        NoiseType.gaussian,
        mean=np.asarray(n_mean),
        covariance=np.atleast_2d(n_cov),
        covariance_type=n_cov_type
    )

    return GammaProcessWithGaussianObservationGssm(omega, phi, x_noise, z_noise)
