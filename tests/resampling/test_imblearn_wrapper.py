from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

from multi_imbalance.resampling.imblearn_wrapper import ImblearnWrapper

X = np.vstack(
    [
        np.random.normal(0, 1, (100, 2)),
        np.random.normal(3, 5, (30, 2)),
        np.random.normal(-2, 2, (20, 2)),
    ]
)

y = np.array([1] * 100 + [2] * 30 + [3] * 20)


def test_wrapper_static_smote():
    clf = ImblearnWrapper(SMOTE(), mode="static")
    resampled_X, resampled_y = clf.fit_resample(X, y)

    cnt = Counter(resampled_y)

    assert cnt[1] == 100
    assert cnt[2] == 60
    assert cnt[3] == 80


def test_wrapper_default_oversampling():
    clf = ImblearnWrapper(SMOTE())
    resampled_X, resampled_y = clf.fit_resample(X, y)

    cnt = Counter(resampled_y)

    assert cnt[1] == 100
    assert cnt[2] >= cnt[1]
    assert cnt[3] >= cnt[1]


def test_wrapper_default_undersampling():
    clf = ImblearnWrapper(ClusterCentroids())
    resampled_X, resampled_y = clf.fit_resample(X, y)

    cnt = Counter(y)
    resampled_cnt = Counter(resampled_y)

    assert resampled_cnt[1] <= cnt[1]
    assert resampled_cnt[2] <= cnt[2]
    assert resampled_cnt[3] <= cnt[3]
