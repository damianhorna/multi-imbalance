from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from multi_imbalance.resampling.ccr import CCR, MultiClassCCR

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

multiclass_X = np.vstack(
        [
            np.random.normal(0, 1, (100, 2)),
            np.random.normal(3, 5, (30, 2)),
            np.random.normal(-2, 2, (20, 2)),
            np.random.normal(-4, 1, (10, 2)),
            np.random.normal(10, 1, (5, 2)),
        ]
    )

multiclass_y = np.array([1] * 100 + [2] * 30 + [3] * 20 + [4] * 10 + [5] * 5)


def test_compare_cleaning_results_to_original_article_implementation():
    clf = CCR(energy=0.5)
    resampled_X, resampled_y = clf.fit_resample(X, y)
    assert_array_equal(np.sort(resampled_X[:X.shape[0]], axis=0), np.sort(original_cleaning_results, axis=0))


def test_multiclass_ccr_call_count():
    clf = MultiClassCCR(energy=0.5)

    with patch.object(CCR, '_clean_and_generate', wraps=clf.CCR._clean_and_generate) as mock:
        _, _ = clf.fit_resample(multiclass_X, multiclass_y)
        print(mock.call_count)
        assert mock.call_count == 4
