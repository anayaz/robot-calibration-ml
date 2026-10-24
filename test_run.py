import numpy as np
from src.kinematics import forward_kinematics

q = np.deg2rad([30, 60])

L_nominal = [1.0, 0.8]
L_actual = [1.005, 0.797]

p_nominal = forward_kinematics(q, L_nominal)
p_actual = forward_kinematics(q, L_actual)

error = p_actual - p_nominal

print("expected position (manufacturer model):", p_nominal)
print("True position (real robot):", p_actual)
print("Positioning error (mm):", error * 1000)

# Generate measurements at different joint configurations
np.random.seed(42)

L_actual = [1.005, 0.797]

angles = np.deg2rad([
    [0, 30],
    [20, 60],
    [45, 90],
    [60, -45],
    [90, 30],
])

# Simulate actual positions with measurement noise (0.5 mm)
measurements = []

for q in angles:
    position = forward_kinematics(q, L_actual)
    noise = np.random.normal(0, 0.0005, size=2)
    measurements.append(position + noise)

measurements = np.array(measurements)

print("Measured positions:")
print(measurements)

from src.calibration import calibrate

L_estimated = calibrate(angles, measurements, L_nominal)

print("Manufacturer lengths:", L_nominal)
print("Estimated lengths:   ", L_estimated)
print("True lengths:        ", L_actual)


# Compare positioning errors before and after calibration
errors_before = []
errors_after = []

for q in angles:
    p_true = forward_kinematics(q, L_actual)

    # Before calibration: manufacturer's model
    p_before = forward_kinematics(q, L_nominal)

    # After calibration: updated model
    p_after = forward_kinematics(q, L_estimated)

    errors_before.append(np.linalg.norm(p_true - p_before))
    errors_after.append(np.linalg.norm(p_true - p_after))

# Convert to millimeters
errors_before = np.array(errors_before) * 1000
errors_after = np.array(errors_after) * 1000

print("Mean error before calibration:", np.mean(errors_before), "mm")
print("Mean error after calibration: ", np.mean(errors_after), "mm")


# Generate 100 new joint configurations
rng = np.random.default_rng(123)
test_angles = rng.uniform(-np.pi, np.pi, size=(100, 2))

errors_before = []
errors_after = []

for q in test_angles:
    p_true = forward_kinematics(q, L_actual)

    p_before = forward_kinematics(q, L_nominal)
    p_after = forward_kinematics(q, L_estimated)

    errors_before.append(np.linalg.norm(p_true - p_before))
    errors_after.append(np.linalg.norm(p_true - p_after))

print("TEST RESULTS (100 new configurations)")
print("Before:", np.mean(errors_before) * 1000, "mm")
print("After: ", np.mean(errors_after) * 1000, "mm")


# Joint-angle offsets (degrees)
joint_offsets = np.deg2rad([0.2, -0.3])

# Manufacturer's commanded angles
q_commanded = np.deg2rad([30, 60])

# Actual physical joint angles
q_true = q_commanded + joint_offsets

# Expected position from manufacturer model
p_expected = forward_kinematics(q_commanded, L_nominal)

# True position with both link-length and joint-angle errors
p_true = forward_kinematics(q_true, L_actual)

error = np.linalg.norm(p_true - p_expected) * 1000

print("Expected position:", p_expected)
print("True position:    ", p_true)
print("Positioning error:", error, "mm")

# Simulate a robot with both types of errors
joint_offsets = np.deg2rad([0.2, -0.3])

rng = np.random.default_rng(42)

# 30 configurations for calibration
train_angles = rng.uniform(-np.pi, np.pi, size=(30, 2))

train_measurements = []

for q in train_angles:
    # Physical robot has joint offsets
    p_true = forward_kinematics(q + joint_offsets, L_actual)

    # Simulated measurement noise: 0.5 mm
    noise = rng.normal(0, 0.0005, size=2)

    train_measurements.append(p_true + noise)

train_measurements = np.array(train_measurements)

# Calibrate ONLY link lengths, ignoring joint offsets
L_estimated_2 = calibrate(
    train_angles,
    train_measurements,
    L_nominal
)

print("True lengths:     ", L_actual)
print("Estimated lengths:", L_estimated_2)

test_angles = rng.uniform(-np.pi, np.pi, size=(100, 2))

errors_before = []
errors_after = []

for q in test_angles:
    # True robot includes joint-angle offsets
    p_true = forward_kinematics(q + joint_offsets, L_actual)

    # Manufacturer's model
    p_before = forward_kinematics(q, L_nominal)

    # Calibrated model (length corrections only)
    p_after = forward_kinematics(q, L_estimated_2)

    errors_before.append(np.linalg.norm(p_true - p_before))
    errors_after.append(np.linalg.norm(p_true - p_after))

print("Before calibration:", np.mean(errors_before) * 1000, "mm")
print("After calibration: ", np.mean(errors_after) * 1000, "mm")


from src.calibration import calibrate_full

params_estimated = calibrate_full(
    train_angles,
    train_measurements,
    L_nominal
)

print("Estimated lengths:", params_estimated[:2])
print("True lengths:     ", L_actual)

print("Estimated angle offsets (degrees):",
      np.rad2deg(params_estimated[2:]))

print("True angle offsets (degrees):",
      np.rad2deg(joint_offsets))


errors_full = []

L_full = params_estimated[:2]
offsets_full = params_estimated[2:]

for q in test_angles:
    p_true = forward_kinematics(q + joint_offsets, L_actual)
    p_calibrated = forward_kinematics(q + offsets_full, L_full)

    errors_full.append(np.linalg.norm(p_true - p_calibrated))

print("Before calibration:     ", np.mean(errors_before) * 1000, "mm")
print("Length-only calibration:", np.mean(errors_after) * 1000, "mm")
print("Full calibration:       ", np.mean(errors_full) * 1000, "mm")


# Load-dependent deformation experiment

g = 9.81
tool_mass = 10.0  # kg
force = tool_mass * g

# Assume 0.02 mm deformation per newton
k_true = 0.00002  # meters / newton

# Use a test configuration
q = np.deg2rad([30, 60])

# Position predicted by our calibrated model
p_predicted = forward_kinematics(q + offsets_full, L_full)

# True unloaded position
p_unloaded = forward_kinematics(q + joint_offsets, L_actual)

# True loaded position
p_loaded = p_unloaded.copy()
p_loaded[1] -= k_true * force

# Positioning error under load
error_loaded = np.linalg.norm(p_loaded - p_predicted) * 1000

print("Predicted position:", p_predicted)
print("Loaded position:   ", p_loaded)
print("Error under load:  ", error_loaded, "mm")