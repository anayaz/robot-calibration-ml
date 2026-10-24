import numpy as np

from src.gp_correction import train_gp


def test_gp_prediction():
    """Check that the GP learns a smooth positioning error."""

    rng = np.random.default_rng(42)

    # Generate 100 robot configurations.
    X = rng.uniform(-1, 1, size=(100, 2))

    # Simulate known positioning errors in meters.
    Y = np.column_stack([
        0.002 * np.sin(X[:, 0]),
        -0.003 * np.cos(X[:, 1])
    ])

    # Train the GP.
    # Expected warning: noise-free test data pushes the GP noise level to its lower bound.
    gp = train_gp(X, Y)

    # Test on configurations not used during training.
    X_test = rng.uniform(-1, 1, size=(50, 2))

    Y_true = np.column_stack([
        0.002 * np.sin(X_test[:, 0]),
        -0.003 * np.cos(X_test[:, 1])
    ])

    Y_pred = gp.predict(X_test)

    # Calculate the mean positioning error in meters.
    mean_error = np.mean(
        np.linalg.norm(Y_true - Y_pred, axis=1)
    )

    # Require an average prediction error below 0.2 mm.
    assert mean_error < 0.0002