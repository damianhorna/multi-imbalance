import numpy as np
from numpy.testing import assert_array_equal

from multi_imbalance.resampling.ccr import CCR

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

original_cleaning_results = np.array([
    [0.3279181486665761, 0.5367090144999095],
    [0.3123061206752467, 0.4686048130243289],
    [0.49287499380029176, 0.6594306863002918],
    [0.3248957760286293, 0.4914514685286293],
    [0.51916715, 0.46894559],
    [0.55701822, 0.32693741]
])


def test_compare_cleaning_results_to_original_article_implementation():
    clf = CCR(energy=0.5)
    resampled_X, resampled_y = clf.fit_resample(X, y)
    assert_array_equal(np.sort(resampled_X[:X.shape[0]], axis=0), np.sort(original_cleaning_results, axis=0))
