# Robot Calibration with Physics and Machine Learning

A Python-based educational project demonstrating how geometric calibration and machine learning can improve the positioning accuracy of a two-link planar robot. The project uses simplified simulations and was developed for learning and demonstration purposes.

## Overview

The project combines:

- Forward kinematics for a two-link robot
- Nonlinear least-squares calibration of link lengths and joint offsets
- Simulation of configuration-dependent positioning errors
- Gaussian Process regression for residual error correction
- Uncertainty-guided selection of additional calibration measurements

## Results

In our simulated experiment with 100 unseen robot configurations:

| Model | Mean positioning error |
|---|---:|
| Physics-only | 2.620 mm |
| Physics + GP | 0.076 mm |

The hybrid model reduced mean positioning error by approximately 97.1%.

These results are based on synthetic data and do not represent
performance on a real industrial robot.

## Project Structure

- `src/kinematics.py` — Forward kinematics
- `src/calibration.py` — Least-squares parameter estimation
- `src/gp_correction.py` — Gaussian Process residual correction
- `gp_experiment.py` — Hybrid calibration experiment
- `test_run.py` — Initial calibration experiments
- `tests/` — Automated tests

## Installation

Clone the project, then create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Experiment

```bash
python3 gp_experiment.py
```

## Run the Tests

```bash
python3 -m pytest tests/ -v
```

## Limitations

The project uses a simplified planar robot and simulated deformation.
Real industrial robot calibration requires additional considerations,
including measurement uncertainty, load-dependent deformation,
joint compliance, and workspace constraints.

## Future Improvements

This project is a starting point that I plan to extend with more realistic robot models and calibration methods.

- Extend the model to a 3D robot with 6 degrees of freedom.
- Include more realistic errors, such as joint misalignment and load-dependent deformation.
- Explore different machine learning models for error correction.
- Improve calibration efficiency by selecting the most useful measurement points.
- Test the methods on real robot data.
- Add 3D visualization and explore C++ for performance improvements.
