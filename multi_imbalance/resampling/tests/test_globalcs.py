from collections import Counter, defaultdict

import numpy as np
import pytest

from multi_imbalance.resampling.global_cs import GlobalCS

X = np.array([
    [0.05837771, 0.57543339],
    [0.06153624, 0.99871925],
    [0.14308529, 0.00681144],
    [0.23401697, 0.21188708],
    [0.2418553, 0.02137086],
    [0.32480534, 0.81547632],
    [0.42478482, 0.31995162],
    [0.50726834, 0.72621157],
    [0.54580968, 0.58025914],
    [0.55748531, 0.71866238],
    [0.69208769, 0.63759459],
    [0.70797377, 0.16348051],
    [0.76410615, 0.70451542],
    [0.81680686, 0.50793884],
    [0.8490789, 0.53826627],
    [0.8847505, 0.96856011],
    [0.9287003, 0.97580299],
    [0.9584236, 0.10536541],
    [0.96983103, 0.87666093],
    [0.97352367, 0.78807909],
])

y_balanced = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
y_imb_easy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
y_imb_hard = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
complete_test_data = [
    (X, y_balanced),
    (X, y_imb_easy),
    (X, y_imb_hard),
]


@pytest.fixture()
def global_cs_mock():
    def _get_parametrized_globalcs(X, y):
        clf = GlobalCS()
        clf.quantities = Counter(y)
        return clf

    return _get_parametrized_globalcs


def get_goal_quantity(y):
    quantities = Counter(y)
    return max(quantities.values()) * len(quantities.keys())


def calc_duplicates_quantities(X, y, X_oversampled):
    quantities = dict()
    for label in Counter(y).keys():
        quantities[label] = list()

    for i, row in enumerate(X):
        equal_row_indices = np.where((X_oversampled == row).all(axis=1))[0]
        label = y[i]
        quantities[label].append(len(equal_row_indices))

    return quantities


@pytest.mark.parametrize("X, y", complete_test_data)
def test_output_length_validate(X, y, global_cs_mock):
    clf = global_cs_mock(X, y)
    oversampled_X, oversampled_y = clf.fit_transform(X, y)
    assert len(oversampled_X) == get_goal_quantity(y)
    assert len(oversampled_y) == get_goal_quantity(y)


@pytest.mark.parametrize("X, y", complete_test_data)
def test_output_equal_replication(X, y, global_cs_mock):
    clf = global_cs_mock(X, y)
    oversampled_X, oversampled_y = clf.fit_transform(X, y)
    oversampled_class_quantities = calc_duplicates_quantities(X, y, oversampled_X)

    for label, duplicates_quantities in oversampled_class_quantities.items():
        min_quantity = min(duplicates_quantities)
        max_quantity = max(duplicates_quantities)

        assert max_quantity - min_quantity <= 1


def test_invalid_input_when_not_enough_labels():
    clf = GlobalCS()
    X = np.array([[1, 1], [1, 0]])
    y = np.array([1])

    with pytest.raises(AssertionError):
        _, _ = clf.fit_transform(X, y)


def test_invalid_input_when_one_dimension_X():
    clf = GlobalCS()
    X = np.array([1, 1, 0])
    y = np.array([1])

    with pytest.raises(AssertionError):
        _, _ = clf.fit_transform(X, y)
