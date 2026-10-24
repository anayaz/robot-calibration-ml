import numpy as np

from src.kinematics import forward_kinematics
from src.calibration import calibrate, calibrate_full


def test_link_length_calibration():
    """Check that calibration recovers the true link lengths from exact measurements."""

    rng = np.random.default_rng(42)

    L_true = np.array([1.005, 0.797])
    L_initial = np.array([1.0, 0.8])

    # Generate measurements at different robot configurations.
    angles = rng.uniform(-np.pi, np.pi, size=(100, 2))

    measurements = np.array([
        forward_kinematics(q, L_true)
        for q in angles
    ])

    L_estimated = calibrate(angles, measurements, L_initial)

    np.testing.assert_allclose(
        L_estimated,
        L_true,
        atol=1e-5
    )


def test_full_calibration():
    """Check that calibration recovers link lengths and joint offsets."""

    rng = np.random.default_rng(42)

    L_true = np.array([1.005, 0.797])
    offsets_true = np.deg2rad([0.2, -0.3])

    L_initial = np.array([1.0, 0.8])

    angles = rng.uniform(-np.pi, np.pi, size=(100, 2))

    # Simulate a robot whose actual joint angles have small offsets.
    measurements = np.array([
        forward_kinematics(q + offsets_true, L_true)
        for q in angles
    ])

    estimated = calibrate_full(angles, measurements, L_initial)

    np.testing.assert_allclose(
        estimated[:2],
        L_true,
        atol=1e-5
    )

    np.testing.assert_allclose(
        estimated[2:],
        offsets_true,
        atol=1e-5
    )

def test_calibration_with_noise():
    """Check that calibration recovers link lengths despite measurement noise."""

    rng = np.random.default_rng(42)

    L_true = np.array([1.005, 0.797])
    L_initial = np.array([1.0, 0.8])

    angles = rng.uniform(-np.pi, np.pi, size=(100, 2))

    # Simulate measurements with 0.2 mm Gaussian noise.
    measurements = np.array([
        forward_kinematics(q, L_true)
        + rng.normal(0, 0.0002, size=2)
        for q in angles
    ])

    L_estimated = calibrate(angles, measurements, L_initial)

    # With noisy measurements, we allow a 0.1 mm parameter error.
    np.testing.assert_allclose(
        L_estimated,
        L_true,
        atol=0.0001,
        rtol=0
    )