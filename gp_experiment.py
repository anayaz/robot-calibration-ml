import numpy as np

from src.kinematics import forward_kinematics

rng = np.random.default_rng(42)

# Robot parameters
L_true = [1.005, 0.797]
L_calibrated = [1.005, 0.797]

# Generate 100 random joint configurations
angles = rng.uniform(-np.pi / 2, np.pi / 2, size=(100, 2))


def deformation(q):
    """Simulated configuration-dependent deformation in meters."""
    q1, q2 = q

    dx = 0.002 * np.sin(q1)
    dy = -0.003 * np.cos(q2)

    return np.array([dx, dy])


X = []
Y = []

for q in angles:
    # Physics model prediction
    p_physics = forward_kinematics(q, L_calibrated)

    # True position includes deformation
    p_true = forward_kinematics(q, L_true) + deformation(q)

    # Simulated measurement noise: 0.2 mm
    noise = rng.normal(0, 0.0002, size=2)
    p_measured = p_true + noise

    # Residual = measured - physics prediction
    residual = p_measured - p_physics

    X.append(q)
    Y.append(residual)

X = np.array(X)
Y = np.array(Y)

print("Input shape:", X.shape)
print("Residual shape:", Y.shape)
print("First residual (mm):", Y[0] * 1000)

from src.gp_correction import train_gp

gp = train_gp(X, Y)

print("GP trained successfully!")
print("Learned kernel:", gp.kernel_)

# Predict the residual at the first configuration
prediction = gp.predict(X[:1])

print("Measured residual (mm):", Y[0] * 1000)
print("GP prediction (mm):    ", prediction[0] * 1000)


# Generate independent test configurations
test_angles = rng.uniform(-np.pi / 2, np.pi / 2, size=(100, 2))

errors_physics = []
errors_hybrid = []

for q in test_angles:
    # True position, including deformation
    p_true = forward_kinematics(q, L_true) + deformation(q)

    # Physics-only prediction
    p_physics = forward_kinematics(q, L_calibrated)

    # GP predicts the remaining positioning error
    correction = gp.predict(q.reshape(1, -1))[0]

    # Hybrid prediction
    p_hybrid = p_physics + correction

    errors_physics.append(np.linalg.norm(p_true - p_physics))
    errors_hybrid.append(np.linalg.norm(p_true - p_hybrid))

print("\nTEST RESULTS (100 unseen configurations)")
print("Physics-only error:", np.mean(errors_physics) * 1000, "mm")
print("Physics + GP error:", np.mean(errors_hybrid) * 1000, "mm")


# Test configurations outside the training range
outside_angles = np.deg2rad([
    [0, 0],       # Inside training range
    [60, 60],     # Inside training range
    [120, 120],   # Outside training range
    [160, 160],   # Far outside training range
])

# Predict residual errors and uncertainty
predictions, std = gp.predict(outside_angles, return_std=True)

for q, pred, uncertainty in zip(outside_angles, predictions, std):
    print(
        f"Angles: {np.rad2deg(q)} | "
        f"Correction: {pred * 1000} mm | "
        f"Uncertainty: {uncertainty * 1000} mm"
    )


# Generate 500 candidate robot configurations
candidate_angles = rng.uniform(
    -np.pi,
    np.pi,
    size=(500, 2)
)

# Predict GP uncertainty at each configuration
_, std = gp.predict(candidate_angles, return_std=True)

# Combine x and y uncertainties into one score
uncertainty_score = np.sqrt(np.sum(std**2, axis=1))

# Find the configuration with maximum uncertainty
best_idx = np.argmax(uncertainty_score)
best_angles = candidate_angles[best_idx]

print("\nNEXT CALIBRATION MEASUREMENT")
print("Joint angles (degrees):", np.rad2deg(best_angles))
print("Predicted uncertainty (mm):", std[best_idx] * 1000)

# 1. Simulate a new measurement
p_true = forward_kinematics(best_angles, L_true) + deformation(best_angles)
p_measured = p_true + rng.normal(0, 0.0002, size=2)

# 2. Calculate its residual
p_physics = forward_kinematics(best_angles, L_calibrated)
new_residual = p_measured - p_physics

# 3. Add the measurement to our training dataset
X_updated = np.vstack([X, best_angles])
Y_updated = np.vstack([Y, new_residual])

# 4. Retrain the GP
gp_updated = train_gp(X_updated, Y_updated)

# 5. Compare uncertainty before and after
_, std_before = gp.predict(best_angles.reshape(1, -1), return_std=True)
_, std_after = gp_updated.predict(best_angles.reshape(1, -1), return_std=True)

print("\nUNCERTAINTY BEFORE AND AFTER")
print("Before (mm):", std_before[0] * 1000)
print("After (mm): ", std_after[0] * 1000)

# Generate test configurations outside the original training range
# test_outside = rng.uniform(
#     np.pi / 2,
#     np.pi,
#     size=(100, 2)
# )

test_outside = best_angles + rng.normal(
    0,
    np.deg2rad(10),
    size=(100, 2)
)

errors_before = []
errors_after = []

for q in test_outside:

    # True position
    p_true = forward_kinematics(q, L_true) + deformation(q)

    # Physics prediction
    p_physics = forward_kinematics(q, L_calibrated)

    # Original GP correction
    correction_before = gp.predict(q.reshape(1, -1))[0]

    # Updated GP correction
    correction_after = gp_updated.predict(q.reshape(1, -1))[0]

    # Positioning errors
    errors_before.append(
        np.linalg.norm(p_true - (p_physics + correction_before))
    )

    errors_after.append(
        np.linalg.norm(p_true - (p_physics + correction_after))
    )

print("\nEXTRAPOLATION TEST")
print("Original GP error (mm):", np.mean(errors_before) * 1000)
print("Updated GP error (mm): ", np.mean(errors_after) * 1000)