from typing import Iterable

import numpy as np
from numpy.testing import assert_array_almost_equal

from kinematics.gravity_forces import (
    CelestialBodyGravityInfo,
    NBodyProblemGravityAccelerationProvider,
)
from kinematics.models import KinematicState
from bayes_sse._utils.quaternion_utils import from_float_array


class RigidBodyData:
    def __init__(self, position, mass) -> None:
        super().__init__()
        self.position = position
        self.mass = mass


def build_rigid_body(data: RigidBodyData) -> CelestialBodyGravityInfo:
    return CelestialBodyGravityInfo(
        KinematicState(
            position=data.position,
            velocity=np.zeros((3,)),
            quaternion=from_float_array(np.random.random(4)),
        ),
        data.mass,
    )


def build_acceleration_provider(
    rigid_bodies: Iterable[RigidBodyData],
) -> NBodyProblemGravityAccelerationProvider:
    gi_list = map(lambda x: build_rigid_body(x), rigid_bodies)
    return NBodyProblemGravityAccelerationProvider(*gi_list)


class TestNBodyProblemGravityAccelerationProvider:
    def test_eval_acceleration(self):
        gravity_acceleration_provider = build_acceleration_provider(
            [
                RigidBodyData(1 * np.ones((3,)), 1e20),
                RigidBodyData(2 * np.ones((3,)), 2e20),
                RigidBodyData(3 * np.ones((3,)), 3e20),
                RigidBodyData(4 * np.ones((3,)), 4e20),
                RigidBodyData(6 * np.ones((3,)), 5e20),
            ]
        )
        gi = build_rigid_body(RigidBodyData(np.zeros((3,)), 6e6))
        dt = 5e3

        expected_acc = [2.85428288, 2.85428288, 2.85428288]
        gravity_acceleration = gravity_acceleration_provider.eval_acceleration(gi, dt)

        assert_array_almost_equal(gravity_acceleration, expected_acc, decimal=4)
