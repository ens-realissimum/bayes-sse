from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from bayes_sse.inference.stochastic_models.stochastic_processes import (
    WienerProcessValueProvider,
)


class AngularVelocityProvider(ABC):
    @abstractmethod
    def eval(self, k: int) -> np.ndarray:
        """
        Evaluate angular velocity at discrete time = k.
        :param k: discrete time, int, [-].
        :return: the angular velocity vector in 3-space at time = k; numpy array of size (3,), rad.
        """
        pass


class WienerModelAngularVelocityProvider(AngularVelocityProvider):
    """
    Model of angular velocity where angular velocity is standard Wiener Process (Brownian motion).
    Angular velocity for each axis (in 3D space) are independent.
    """

    def __init__(
        self, initial_angular_velocity: np.ndarray, delta: float, dt: float, n: int
    ):
        """
        Initialize instance of WienerAccelerationModelProvider
        :param initial_angular_velocity: initial angular velocity
        :param delta: 'velocity' of Wiener process.
        :param dt: float, The time step.
        :param n: int, The number of steps to take.
        """
        self._values_provider = WienerProcessValueProvider(
            initial_angular_velocity, delta, dt, n
        )

    def eval(self, k: int) -> np.ndarray:
        """
        Evaluate angular velocity at discrete time = k.
        :param k: discrete time, int, [-].
        :return: the angular velocity vector in 3-space at time = k; numpy array of size (3,), rad.
        """
        return self._values_provider.eval(k)
