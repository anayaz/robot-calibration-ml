import numpy as np


def forward_kinematics(q, L):
    """
    Calculate the end-effector position of a 2-link planar robot.

    q: Joint angles in radians [q1, q2].
    L: Link lengths in meters [L1, L2].

    Returns the end-effector position [x, y] in meters.
    """
    q1, q2 = q
    L1, L2 = L

    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)

    return np.array([x, y])