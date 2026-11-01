from __future__ import annotations

from abc import ABC, abstractmethod
from os import linesep

import numpy as np
from scipy import constants as sc_const
from scipy.integrate import solve_ivp

from kinematics.models import KinematicState
from kinematics.motion_equations import uniform_acceleration_motion_equation


class CelestialBodyGravityInfo:
    """
    Contains information that required to calculate gravitation acceleration:
        - mass of object;
        - position vector of object (vector at 3D space), [km].
        - velocity vector of object (vector at 3D space), [km/sec].
    """

    def __init__(self, kinematic_state: KinematicState, mass: float):
        """
        Creates instance of GravityInfo.
        :param kinematic_state: object kinematic state, KinematicState.
        :param mass: object mass, [kg].
        """
        self._state = kinematic_state
        self._mass = mass

    def __str__(self):
        return f"Mass: {self._mass} kg;{linesep}Kinematic state: {self._state}"

    @property
    def kinematic_state(self) -> KinematicState:
        return self._state

    @property
    def mass(self) -> float:
        return self._mass


class GravityAccelerationProvider(ABC):
    @abstractmethod
    def eval_acceleration(
        self, gravity_info: CelestialBodyGravityInfo, dt: float
    ) -> np.ndarray:
        """
        Evaluate gravity acceleration at discrete time = k.
        :param gravity_info: information about position vector of celestial body at time k and object mass.
        :param dt: sampling time, float, [sec].
        :return: the acceleration vector in 3-space at time = k; numpy of size (3,), [km/sec^2].
        """
        pass


class NBodyProblemGravityAccelerationProvider(GravityAccelerationProvider):
    """
    Evaluate gravity acceleration at discrete time = k for a body that was not part of initial system (n-1-body system).
    The class is initialized by n-1 body system.
    During runtime it allows to add another one body and calculate acceleration for n-body system.
    The class allows to solve following problem. Object initialized with solar system planets, Sun and other massive body.
    This is n-1 system.
    We interested in acceleration of a spaceship.
    So, during runtime the spaceship will be included to n-body system and n-body problem will be solved.
    """

    def __init__(self, *celestial_body: CelestialBodyGravityInfo):
        self._bodies = list(celestial_body).copy()

    def __str__(self) -> str:
        state = linesep.join(
            map(
                lambda x: f"Mass: {x.mass}; Kinematic State: {x.kinematic_state}",
                self._bodies,
            )
        )
        return f"N-body model with following bodies:{linesep}{state}"

    def eval_acceleration(
        self, gravity_info: CelestialBodyGravityInfo, dt: float
    ) -> np.ndarray:
        accelerations = eval_n_body_gravity_acceleration(
            np.vstack(
                (
                    np.asarray(
                        list(map(lambda x: x.kinematic_state.position, self._bodies))
                    ),
                    gravity_info.kinematic_state.position,
                )
            ),
            np.append(
                np.asarray(list(map(lambda x: x.mass, self._bodies))), gravity_info.mass
            ),
        )

        self._bodies = list(
            map(
                lambda _: _update_gravity_info(_[0], _[1], dt),
                [(x, accelerations[i, :]) for i, x in enumerate(self._bodies)],
            )
        )

        #  return last object as acceleration, coz gravity_info of input object were added as last element
        return accelerations[-1, :]


def _update_gravity_info(
    g_info: CelestialBodyGravityInfo, acceleration: np.ndarray, dt: float
) -> CelestialBodyGravityInfo:
    """
    Calculate updated position and velocity vectors of g_info based on acceleration at the end of interval [to; t0 + dt].
    Adams/BDF method is used to solve ode.
    Quaternion (angular position) and mass remain same as on previous step.
    :param g_info: gravity information: kinematic vector and mass.
    :param acceleration: acceleration at this point of time, [km/sec^2].
    :param dt: sample time, float, [sec].
    :return: Gravity info with updated velocity and position.
    """
    x_new = solve_ivp(
        lambda t, x: uniform_acceleration_motion_equation(x, acceleration),
        [0, dt],
        np.hstack((g_info.kinematic_state.position, g_info.kinematic_state.velocity)),
        method="LSODA",
    )
    # todo check and uncomment if it works fine
    # velocity = g_info.kinematic_state.velocity + acceleration * dt
    # position = g_info.kinematic_state.position + velocity * dt
    state = KinematicState(
        x_new.y[:3, -1], x_new.y[3:, -1], g_info.kinematic_state.quaternion
    )

    return CelestialBodyGravityInfo(state, g_info.mass)


def eval_n_body_gravity_acceleration(
    positions: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """
    Compute the accelerations due to gravity for an n-body problem in 3D-space
    :param positions: positions of celestial bodies,
    which should be considered when solving the n-body problem; numpy array of size (n,3), [km].
    :return: the acceleration vector in 3-space at time = k; numpy array of size (n,3), [km/sec^2].
    """
    positions_m = np.asarray(positions * 1e3)  # position in [m]
    displacements_positions = positions_m.reshape((1, -1, 3)) - positions_m.reshape(
        (-1, 1, 3)
    )  # displacements
    dists = np.linalg.norm(displacements_positions, axis=2)
    dists[dists == 0] = 1  # to avoid divide by zero

    mass_matrix = masses.reshape((1, -1, 1)) * masses.reshape((-1, 1, 1))
    forces = (
        sc_const.gravitational_constant
        * displacements_positions
        * mass_matrix
        / np.expand_dims(dists, 2) ** 3
    )

    acceleration = forces.sum(axis=1) / masses.reshape(-1, 1)  # [m/sec^2]
    return acceleration / 1e3  # convert to [km/sec^2]
