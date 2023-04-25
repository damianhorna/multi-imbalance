from collections import Counter

import numpy as np

from multi_imbalance.resampling.rbo import RBO

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


def test_rbo():
    clf = RBO(gamma=0.5, step=2, iterations=10, k=3)
    oversampled_X, oversampled_y = clf.fit_resample(X, y)
    cnt = Counter(oversampled_y)
    assert cnt[1] == cnt[0]
