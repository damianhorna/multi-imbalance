from collections import Counter

import numpy as np
import pytest

from multi_imbalance.resampling.rbo import RBO, MultiClassRBO

X = np.array([
    [0.51916715, 0.46894559],
    [0.42850038, 0.49204451],
    [0.45844347, 0.44231806],
    [0.49862482, 0.61777354],
    [0.55701822, 0.32693741],
    [0.37040839, 0.48617894],
])

y = np.array([
    1,
    0,
    0,
    0,
    1,
    0
])

mc_X = np.vstack(
    [
        np.random.normal(0, 1, (100, 2)),
        np.random.normal(3, 5, (30, 2)),
        np.random.normal(-2, 2, (20, 2)),
    ]
)

mc_y = np.array([1] * 100 + [2] * 30 + [3] * 20)


def test_rbo_resampling_counts():
    clf = RBO(gamma=0.5, step=2, iterations=10, k=3)
    oversampled_X, oversampled_y = clf.fit_resample(X, y)
    cnt = Counter(oversampled_y)
    assert cnt[1] == cnt[0]


def test_mcrbo_resampling_counts():
    clf = MultiClassRBO(gamma=0.5, step=2, iterations=10, k=3)
    oversampled_X, oversampled_y = clf.fit_resample(mc_X, mc_y)
    cnt = Counter(oversampled_y)
    assert cnt[1] == cnt[2]
    assert cnt[1] == cnt[3]
    assert cnt[2] == cnt[3]


def test_mutual_potential_function():
    clf = RBO(gamma=1000, step=2, iterations=10, k=3)

    X_minority = np.array([
        [1, 1],
        [2, 2],
    ])

    X_majority = np.array([
        [-1, -1],
        [-2, 2]
    ])

    x = np.array([0, 0])

    assert clf._mutual_class_potential(x, X_majority, X_minority) == pytest.approx(0)


def test_potential_for_empty_collection_equals_zero():
    clf = RBO(gamma=1, step=2, iterations=10, k=2)

    X_minority = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])

    X_majority = np.array([
        [-1, -1],
        [-2, -2]
    ])

    X = np.vstack([X_minority, X_majority])
    y = np.array([1, 1, 1, 0, 0])

    minority_index = 1
    minority_class = 1

    k_sorted_nearest_neighbours = clf._find_k_sorted_nearest_neighbours(X_minority, X)
    example_nearest_neighbours = k_sorted_nearest_neighbours[minority_index]
    X_majority, X_minority = clf._get_nearest_majority_and_minority_neighbours(example_nearest_neighbours, minority_class, X, y)

    majority_potential = clf._potential(X[minority_index], X_majority)
    minority_potential = clf._potential(X[minority_index], X_minority)
    assert majority_potential == pytest.approx(0)
    assert clf._mutual_class_potential(X[minority_index], X_majority, X_minority) == pytest.approx(-minority_potential)


def test_manhattan_distance_perturbance_step():
    step = 1
    clf = RBO(gamma=0.5, iterations=5, k=5, step=step)

    x = np.array([0, 0])
    new_x = clf._perturb_x(x, x.shape)[np.newaxis, :]
    distance = clf.distance_function(x, new_x)

    assert distance == pytest.approx(step)
