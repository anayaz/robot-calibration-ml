import numpy as np
from scipy.optimize import least_squares

from src.kinematics import forward_kinematics

"""Calculate the differences between measured and predicted robot positions.""" 
def residuals(L, angles, measurements):
    errors = []

    for q, measured in zip(angles, measurements):
        predicted = forward_kinematics(q, L)
        error = measured - predicted
        errors.extend(error)

    return np.array(errors)

"""Estimate link lengths by minimizing positioning errors across measurements."""
def calibrate(angles, measurements, L_initial):
    result = least_squares(
        residuals,
        x0=L_initial,
        args=(angles, measurements),
    )

    return result.x

"""Calculate positioning errors while accounting for link lengths and joint offsets."""
def residuals_full(params, angles, measurements):
    L = params[:2]
    joint_offsets = params[2:]

    errors = []

    for q, measured in zip(angles, measurements):
        predicted = forward_kinematics(q + joint_offsets, L)
        errors.extend(measured - predicted)

    return np.array(errors)

"""Estimate link lengths and joint offsets from measured robot positions."""
def calibrate_full(angles, measurements, L_initial):
    initial_params = np.array([*L_initial, 0.0, 0.0])

    result = least_squares(
        residuals_full,
        x0=initial_params,
        args=(angles, measurements),
    )

    return result.x