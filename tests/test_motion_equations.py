import numpy as np
from numpy.testing import assert_array_almost_equal

from kinematics.models import KinematicState
from kinematics.motion_equations import solve_kinematic_motion_equation
from bayes_sse._utils.quaternion_utils import from_float_array


def test_solve_kinematic_motion_equation():
    x0 = KinematicState(
        position=[1, 1, 1],
        velocity=[2, 2, 2],
        quaternion=from_float_array([1, 0, 0, 0]),
    )
    x1 = solve_kinematic_motion_equation(
        time_span=(0, 100),
        state=x0,
        acceleration=np.asarray([1, 1, 1]),
        gravity_acceleration=np.asarray([0.0, 9.8, 0]),
        angular_velocity=np.asarray([0.25, 0.5, 0.75]),
    )

    assert_array_almost_equal(x1.position, [5201.0, 54201.0, 5201.0], decimal=1)
    assert_array_almost_equal(x1.velocity, [102.0, 1082.0, 102.0], decimal=1)

    expected_quat = np.array([-0.94307022, -0.0888894, -0.1777788, -0.2666682])
    actual_quat = x1.quaternion.components

    assert_array_almost_equal(actual_quat, expected_quat, decimal=3)
