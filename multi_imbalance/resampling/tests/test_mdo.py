from collections import Counter, defaultdict

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose, assert_array_almost_equal

from multi_imbalance.resampling.mdo import MDO

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

y_balanced_weights = [0.058824, 0.088235, 0.088235, 0.088235, 0.117647, 0.117647, 0.117647, 0.088235, 0.117647,
                      0.117647]

y_imb_easy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
y_imb_easy_SC_minor = [[0.8490789, 0.53826627], [0.8847505, 0.96856011], [0.9287003, 0.97580299],
                       [0.96983103, 0.87666093], [0.97352367, 0.78807909]]
y_imb_easy_weights = [0.142857, 0.214286, 0.214286, 0.214286, 0.214286]

y_imb_hard = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
y_imb_hard_SC_minor = np.empty(shape=(0, 2))
y_imb_hard_weights = ()

complete_test_data = [
    (X, y_balanced, y_balanced_SC_minor, y_balanced_weights),
    (X, y_imb_easy, y_imb_easy_SC_minor, y_imb_easy_weights),
    (X, y_imb_hard, y_imb_hard_SC_minor, y_imb_hard_weights),
]


@pytest.fixture()
def mdo_mock():
    def _get_parametrized_mdo(X, y):
        clf = MDO(k1_frac=.5)
        clf.knn.fit(X)
        clf.X, clf.y = X, y
        return clf

    return _get_parametrized_mdo


@pytest.mark.parametrize("X, y, sc_minor_expected, weights_expected", complete_test_data)
def test_choose_samples(X, y, sc_minor_expected, weights_expected, mdo_mock):
    clf = mdo_mock(X, y)
    SC_minor, weights = clf._choose_samples(1)
    assert_array_almost_equal(SC_minor, np.array(sc_minor_expected))
    assert_array_almost_equal(weights, weights_expected)


def test_choose_samples_when_correct(mdo_mock):
    clf = mdo_mock(X, list())
    T = np.array([[-2.74e-01, -2.43e-1], [2.74e-01, 2.43e-1]])
    V = np.array([7.53e-02, 5.91e-3])
    oversampling_rate = 2
    weights = [0.3, 0.7]
    expected_result = np.array([[0.391485, 0.230027], [0.631741, 0.183352]])

    S_temp = clf._MDO_oversampling(T, V, oversampling_rate, weights)
    assert_array_almost_equal(S_temp, expected_result)


def test_choose_samples_when_zero_samples_expected(mdo_mock):
    clf = mdo_mock(X, list())
    T = np.array([[-2.74e-01, -2.43e-1], [2.74e-01, 2.43e-1]])
    V = np.array([7.53e-02, 5.91e-3])
    oversampling_rate = -1
    weights = [0.3, 0.7]

    S_temp = clf._MDO_oversampling(T, V, oversampling_rate, weights)
    assert len(S_temp) == 0


def test_zero_variance(mdo_mock):
    clf = mdo_mock(X, list())
    T = np.array([[-2.74e-01, -2.43e-1], [2.74e-01, 2.43e-1]])
    V = np.array([0, 0])
    oversampling_rate = 2
    weights = [0.3, 0.7]
    expected_result = np.array([[0.157618, 0.330578], [0.254349, 0.263499]])

    S_temp = clf._MDO_oversampling(T, V, oversampling_rate, weights)
    assert_array_almost_equal(S_temp, expected_result)


def test_mdo_api(mdo_mock):
    clf = mdo_mock(X, y_imb_hard)
    maj_int_min = {'maj': [0], 'int': [], 'min': [1]}
    clf.k1 = 0
    clf.class_balances = maj_int_min
    X_r, y_r = clf.fit_transform(X, y_imb_hard)
    assert X_r.shape == (28,2)