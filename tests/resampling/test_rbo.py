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


def distance(x, y, p_norm=1):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def rbf(d, gamma):
    # print("d:", d)
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def mutual_class_potential(point, majority_points, minority_points, gamma):
    result = 0.0
    weights = []
    for majority_point in majority_points:
        print(rbf(distance(point, majority_point), gamma))
        weights.append(rbf(distance(point, majority_point), gamma))
        result += rbf(distance(point, majority_point), gamma)

    print("orig_weights:", weights)
    print("orig_total:", result)
    for minority_point in minority_points:
        result -= rbf(distance(point, minority_point), gamma)

    return result


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

    assert clf.mutual_class_potential(x, X_majority, X_minority) == pytest.approx(0)
