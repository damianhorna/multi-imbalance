from collections import Counter, defaultdict

import numpy as np
import pytest

from multi_imbalance.resampling.SOUP import SOUP

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
y_balanced_quantities = Counter({0: 10, 1: 10})
y_balanced_first_sample_safe_level = 1
y_balanced_0_class_safe_levels = defaultdict(float,
                                             {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0,
                                              9: 1.0})
y_balanced_1_class_safe_levels = defaultdict(float,
                                             {10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0,
                                              18: 1.0, 19: 1.0})

y_imb_easy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
y_imb_easy_quantities = Counter({0: 14, 1: 6})
y_imb_easy_first_sample_safe_level = 0.7714285714285714
y_imb_easy_0_class_safe_levels = defaultdict(float,
                                             {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
                                              8: 1.0, 9: 1.0, 10: 0.8857142857142858, 11: 0.7714285714285714,
                                              12: 0.7714285714285714, 17: 0.7714285714285714})
y_imb_easy_1_class_safe_levels = defaultdict(float, {13: 0.6571428571428571, 14: 0.7714285714285714,
                                                     15: 0.8857142857142858, 16: 0.8857142857142858,
                                                     18: 0.8857142857142858, 19: 0.8857142857142858})

y_imb_hard = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
y_imb_hard_quantities = Counter({0: 14, 1: 6})
y_imb_hard_first_sample_safe_level = 0.7714285714285714
y_imb_hard_quantities_0_class_safe_levels = defaultdict(float, {0: 0.8857142857142858, 1: 0.8857142857142858,
                                                                2: 0.8857142857142858, 3: 0.8857142857142858,
                                                                4: 0.8857142857142858, 5: 0.7714285714285714,
                                                                7: 0.7714285714285714, 10: 0.7714285714285714,
                                                                11: 0.7714285714285714, 12: 0.7714285714285714,
                                                                13: 0.7714285714285714, 15: 0.7714285714285714,
                                                                17: 0.7714285714285714, 19: 0.7714285714285714})
y_imb_hard_quantities_1_class_safe_levels = defaultdict(float, {6: 0.6571428571428571, 8: 0.6571428571428571,
                                                                9: 0.6571428571428571, 14: 0.5428571428571429,
                                                                16: 0.6571428571428571, 18: 0.6571428571428571})

complete_test_data = [
    (X, y_balanced, y_balanced_quantities, y_balanced_0_class_safe_levels, y_balanced_1_class_safe_levels,
     y_balanced_first_sample_safe_level),
    (X, y_imb_easy, y_imb_easy_quantities, y_imb_easy_0_class_safe_levels, y_imb_easy_1_class_safe_levels,
     y_imb_easy_first_sample_safe_level),
    (X, y_imb_hard, y_imb_hard_quantities, y_imb_hard_quantities_0_class_safe_levels,
     y_imb_hard_quantities_1_class_safe_levels, y_imb_hard_first_sample_safe_level),
]

safe_levels_test_data = [
    (X, y_balanced, y_balanced_0_class_safe_levels, y_balanced_quantities),
    (X, y_balanced, y_balanced_1_class_safe_levels, y_balanced_quantities),
    (X, y_imb_easy, y_imb_easy_0_class_safe_levels, y_imb_easy_quantities),
    (X, y_imb_easy, y_imb_easy_1_class_safe_levels, y_imb_easy_quantities),
    (X, y_imb_hard, y_imb_hard_quantities_0_class_safe_levels, y_imb_hard_quantities),
    (X, y_imb_hard, y_imb_hard_quantities_1_class_safe_levels, y_imb_hard_quantities),
]


@pytest.fixture()
def soup_mock():
    def _get_parametrized_soup(X, quantities):
        clf = SOUP(k=5)
        clf.neigh_clf.fit(X)
        clf.quantities = quantities
        clf.goal_quantity = 10
        return clf

    return _get_parametrized_soup


@pytest.mark.parametrize("X, y, quantities, zero_safe_levels, one_safe_levels, first_sample_safe", complete_test_data)
def test_calculating_safe_levels_for_sample(X, y, quantities, zero_safe_levels, one_safe_levels, first_sample_safe,
                                            soup_mock):
    clf = soup_mock(X, quantities)
    neighbour_quantities = Counter({0: 3, 1: 2})

    safe_level = clf._calculate_sample_safe_level(0, neighbour_quantities)
    assert safe_level == first_sample_safe


@pytest.mark.parametrize("X, y, quantities, zero_safe_levels, one_safe_levels, first_sample_safe", complete_test_data)
def test_calculating_safe_levels_for_class(X, y, quantities, zero_safe_levels, one_safe_levels, first_sample_safe,
                                           soup_mock):
    clf = soup_mock(X, quantities)

    zero_levels = clf._construct_class_safe_levels(X, y, 0)
    one_levels = clf._construct_class_safe_levels(X, y, 1)

    assert zero_levels == zero_safe_levels
    assert one_levels == one_safe_levels


@pytest.mark.parametrize("X, y, safe_levels, quantities", safe_levels_test_data)
def test_oversample(X, y, safe_levels, quantities, soup_mock):
    clf = soup_mock(X, quantities)
    if len(safe_levels) <= 10:
        oversampled_X, oversampled_y = clf._oversample(X, y, safe_levels)
        assert len(oversampled_X) == 10
        assert len(oversampled_y) == 10
    else:
        with pytest.raises(AttributeError):
            _, _ = clf._oversample(X, y, safe_levels)


@pytest.mark.parametrize("X, y, safe_levels, quantities", safe_levels_test_data)
def test_undersample(X, y, safe_levels, quantities, soup_mock):
    clf = soup_mock(X, quantities)
    if len(safe_levels) >= 10:
        undersampled_X, undersampled_y = clf._undersample(X, y, safe_levels)
        assert len(undersampled_X) == 10
        assert len(undersampled_y) == 10
    else:
        with pytest.raises(AttributeError):
            _, _ = clf._undersample(X, y, safe_levels)


def test_invalid_input_when_not_enough_labels():
    clf = SOUP(k=5)
    X = np.array([[1, 1], [1, 0]])
    y = np.array([1])

    with pytest.raises(AssertionError):
        _, _ = clf.fit_transform(X, y)


def test_invalid_input_when_one_dimension_X():
    clf = SOUP(k=5)
    X = np.array([1, 1, 0])
    y = np.array([1])

    with pytest.raises(AssertionError):
        _, _ = clf.fit_transform(X, y)
