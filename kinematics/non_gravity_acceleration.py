from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from bayes_sse.inference.stochastic_models.stochastic_processes import (
    WienerProcessValueProvider,
)


class NonGravityAccelerationProvider(ABC):
    @abstractmethod
    def eval(self, k: int) -> np.ndarray:
        """
        Evaluate non-gravity acceleration (acceleration that caused by all non-gravity forces) at discrete time = k.
        :param k: discrete time, int, [-].
        :return: the acceleration vector in 3-space at time = k; numpy array of size (3,), [km/sec^2].
        """
        pass


class WienerAccelerationModelProvider(NonGravityAccelerationProvider):
    """
    Model of acceleration where acceleration is standard Wiener Process (Brownian motion).
    Acceleration for each axis (in 3D space) are independent.
    """

    def __init__(
        self, initial_acceleration: np.ndarray, delta: float, dt: float, n: int
    ):
        """
        Initialize instance of WienerAccelerationModelProvider
        :param initial_acceleration: initial acceleration
        :param delta: 'velocity' of Wiener process.
        :param dt: float, The time step.
        :param n: int, The number of steps to take.
        """
        self._values_provider = WienerProcessValueProvider(
            initial_acceleration, delta, dt, n
        )

    def eval(self, k: int) -> np.ndarray:
        """
        Evaluate non-gravity acceleration (acceleration that caused by all non-gravity forces) at discrete time = k.
        :param k: discrete time, int, [-].
        :return: the acceleration vector in 3-space at time = k; numpy array of size (3,), [km/sec^2].
        """
        return self._values_provider.eval(k)
