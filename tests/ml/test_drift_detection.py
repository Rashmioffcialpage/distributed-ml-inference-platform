import numpy as np
from drift_detection import population_stability_index


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    psi = population_stability_index(a, a.copy())
    assert psi < 1e-6


def test_psi_positive_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(3, 1, 5000)  # clearly shifted
    psi = population_stability_index(a, b)
    assert psi > 0.25  # comfortably above the "alert" threshold used in the pipeline
