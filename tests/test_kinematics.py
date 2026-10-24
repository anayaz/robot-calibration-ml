import numpy as np

from src.kinematics import forward_kinematics


def test_robot_straight():
    """Both joints at zero should place the robot along the x-axis."""
    q = [0, 0]
    L = [1.0, 0.8]

    position = forward_kinematics(q, L)

    np.testing.assert_allclose(position, [1.8, 0], atol=1e-10)


def test_robot_vertical():
    """A 90-degree first joint should point both links upward."""
    q = [np.pi / 2, 0]
    L = [1.0, 0.8]

    position = forward_kinematics(q, L)

    np.testing.assert_allclose(position, [0, 1.8], atol=1e-10)


def test_robot_folded():
    """A 180-degree second joint should fold the second link backward."""
    q = [0, np.pi]
    L = [1.0, 0.8]

    position = forward_kinematics(q, L)

    np.testing.assert_allclose(position, [0.2, 0], atol=1e-10)