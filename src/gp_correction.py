from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel,
)


def train_gp(X, residuals):
    """Train a Gaussian Process to predict positioning errors."""

    # Define the GP kernel
    kernel = (
        ConstantKernel(1.0)
        * RBF(length_scale=1.0)
        + WhiteKernel(noise_level=0.01)
    )

    # Create the GP model
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=42,
    )

    # Train using joint angles and measured residuals
    gp.fit(X, residuals)

    return gp