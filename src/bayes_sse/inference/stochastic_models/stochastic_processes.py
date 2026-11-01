from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy.stats import norm


class WienerProcess:
    """
    Represents the Wiener process (i.e. Brownian motion) with shift (initial position) - 'shift' and
    speed (std variance) of the Brownian motion - 'delta'.
    """

    def __init__(self, shift: Union[float, np.ndarray], delta: float):
        """
        Initialize instance of BrownianProcess.
        :param shift: float or numpy array (or something that can be converted to a numpy array
            using numpy.asarray(x0)). The initial condition(s) (i.e. position(s)) of the Brownian motion.
        :param delta: float, sigma determines the "speed" of the Brownian motion. The random variable
            of the position at time t, X(t), has a normal distribution whose mean is
            the position at time t=0 and whose variance is delta**2*t.
        """
        self._shift = np.asarray(shift).copy()
        self._delta = delta

    def __str__(self):
        return f"Shift: {self._shift}; Delta: {self._delta}"

    def eval(self, dt: float, n: int) -> np.ndarray:
        """
        Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

        where N(mu, sigma; t0, t1) is a normally distributed random variable with mean mu and
        variance sigma. The parameters t0 and t1 make explicit the statistical
        independence of N on different time intervals; that is, if [t0, t1) and
        [t2, t3) are disjoint intervals, then N(mu, sigma; t0, t1) and N(mu, sigma; t2, t3) are independent.

        Written as an iteration scheme,

            X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)

        If `x0`, i.e. shift, is an array (or array-like), each value in `x0` is treated as
        an initial condition, and the value returned is a numpy array with one
        more dimension than `x0`.

        :param dt: float, The time step.
        :param n: int, The number of steps to take.
        :return: np.ndarray, array of generated process values
        """
        # For each element of x0, generate a sample of n numbers from a normal distribution.
        r = norm.rvs(size=self._shift.shape + (n,), scale=self._delta * np.sqrt(dt))
        out = np.empty(r.shape)

        # This computes the Brownian motion by forming the cumulative sum of the random samples.
        np.cumsum(r, axis=-1, out=out)

        # Add the initial condition.
        out += np.expand_dims(self._shift, axis=-1)

        return out


class ProcessValueProvider(ABC):
    @abstractmethod
    def eval(self, k: int) -> np.ndarray:
        """
        Evaluate value of process at time k according to process distribution.
        :param k: discrete time k.
        :return: numpy array, random value of process at time k generated according to process distribution.
        """
        pass


class WienerProcessValueProvider(ProcessValueProvider):
    """
    Provider of values of Wiener Process (Brownian motion).
    """

    def __init__(self, shift: np.ndarray, delta: float, dt: float, n: int):
        """
        Initialize instance of WienerAccelerationModelProvider
        :param shift: shift of Brownian motion.
        :param delta: 'velocity' of Wiener process.
        :param dt: float, The time step.
        :param n: int, The number of steps to take.
        """
        wiener_process = WienerProcess(shift, delta)
        self._values = wiener_process.eval(dt, n)

    def eval(self, k: int) -> np.ndarray:
        """
        Evaluate angular velocity at discrete time = k.
        :param k: discrete time, int, [-].
        :return: the angular velocity vector in 3-space at time = k; numpy array of size (3,), rad.
        """
        return self._values[:, k]


class WienerProcessIterative(ProcessValueProvider):
    """
    Represents the Wiener process (i.e. Brownian motion) with shift (initial position) - 'shift' and
    speed (std variance) of the Brownian motion - 'delta'.
    The class is designed to work in iterative way.
    """

    def __init__(self, shift: Union[float, np.ndarray], delta: float, dt: float):
        """
        Initialize instance of BrownianProcess.
        :param shift: float or numpy array (or something that can be converted to a numpy array
            using numpy.asarray(x0)). The initial condition(s) (i.e. position(s)) of the Brownian motion.
        :param delta: float, sigma determines the "speed" of the Brownian motion. The random variable
            of the position at time t, X(t), has a normal distribution whose mean is
            the position at time t=0 and whose variance is delta**2*t.
        :param dt: float, The time step.
        """
        self._current_value = np.asarray(shift, dtype=float).copy()
        self._shift = np.asarray(shift, dtype=float).copy()
        self._delta = delta
        self._dt = dt
        self._last_generated = np.nan

    def __str__(self):
        return f"Shift: {self._shift}; Delta: {self._delta}"

    def eval(self, k: int) -> np.ndarray:
        """
        Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

        where N(mu, sigma; t0, t1) is a normally distributed random variable with mean mu and
        variance sigma. The parameters t0 and t1 make explicit the statistical
        independence of N on different time intervals; that is, if [t0, t1) and
        [t2, t3) are disjoint intervals, then N(mu, sigma; t0, t1) and N(mu, sigma; t2, t3) are independent.

        Written as an iteration scheme,

            X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)

        If `x0`, i.e. shift, is an array (or array-like), each value in `x0` is treated as
        an initial condition, and the value returned is a numpy array with one
        more dimension than `x0`.

        :param k: float, the time step.
        :return: np array, array of generated process values.
        """
        if self._last_generated > k:
            raise Exception("unable to generate value for time in the past")

        if self._last_generated == k:
            return self._current_value

        self._last_generated = k
        self._current_value += norm.rvs(size=self._current_value.shape, scale=self._delta * np.sqrt(self._dt))
        return self._current_value
