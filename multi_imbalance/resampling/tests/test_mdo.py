from collections import Counter, defaultdict

import numpy as np
import pytest

from multi_imbalance.resampling.MDO import MDO

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
y_balanced_SC_minor = [[0.69208769, 0.63759459],
                       [0.70797377, 0.16348051],
                       [0.76410615, 0.70451542],
                       [0.81680686, 0.50793884],
                       [0.8490789, 0.53826627],
                       [0.8847505, 0.96856011],
                       [0.9287003, 0.97580299],
                       [0.9584236, 0.10536541],
                       [0.96983103, 0.87666093],
                       [0.97352367, 0.78807909]]

y_balanced_weights = [0.0877193, 0.07017544, 0.0877193, 0.0877193, 0.0877193, 0.10526316, 0.12280702, 0.10526316,
                      0.12280702, 0.12280702]

y_imb_easy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
y_imb_easy_SC_minor = [[0.8847505, 0.96856011], [0.9287003, 0.97580299], [0.96983103, 0.87666093],
                       [0.97352367, 0.78807909]]
y_imb_easy_weights = [0.21052632, 0.26315789, 0.26315789, 0.26315789]

y_imb_hard = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
y_imb_hard_SC_minor = y_imb_hard_weights = []

complete_test_data = [
    (X, y_balanced, y_balanced_SC_minor, y_balanced_weights),
    (X, y_imb_easy, y_imb_easy_SC_minor, y_imb_easy_weights),
    (X, y_imb_hard, y_imb_hard_SC_minor, y_imb_hard_weights),
]


@pytest.fixture()
def mdo_mock():
    def _get_parametrized_soup(X, y):
        clf = MDO()
        clf.nn.fit(X)
        return clf

    return _get_parametrized_soup


# @pytest.mark.parametrize("X, y, quantities, zero_safe_levels, one_safe_levels, first_sample_safe", complete_test_data)
# def test_choose_samples(X, y, quantities, zero_safe_levels, one_safe_levels, first_sample_safe, mdo_mock):
#     pass

@pytest.mark.parametrize("X, y, sc_minor_expected, weights_expected", complete_test_data)
def test_oversampling(X, y, sc_minor_expected, weights_expected, mdo_mock):
    clf = mdo_mock(X, y)
    SC_minor, weights = clf._choose_samples(X, y, 1)
    assert SC_minor.all() == np.array(sc_minor_expected).all()
    assert weights.all() == np.array(weights_expected).all()
