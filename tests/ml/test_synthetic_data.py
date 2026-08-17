import numpy as np
from synthetic_data import generate_driving_events, generate_trajectories, EVENT_LABELS


def test_generate_driving_events_shape():
    X, y = generate_driving_events(500, seed=1)
    assert X.shape == (500, 5)
    assert y.shape == (500,)
    assert set(np.unique(y)).issubset(set(range(len(EVENT_LABELS))))


def test_generate_driving_events_deterministic():
    X1, y1 = generate_driving_events(100, seed=42)
    X2, y2 = generate_driving_events(100, seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_driving_events_all_classes_present():
    _, y = generate_driving_events(5000, seed=1)
    assert set(np.unique(y)) == {0, 1, 2, 3}


def test_generate_trajectories_shape():
    history, target = generate_trajectories(64, seq_len=10, seed=1)
    assert history.shape == (64, 10, 2)
    assert target.shape == (64, 2)
