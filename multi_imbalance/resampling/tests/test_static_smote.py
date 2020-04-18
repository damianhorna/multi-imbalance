from collections import Counter

import numpy as np

from multi_imbalance.resampling.static_smote import StaticSMOTE


def test_static_smote():
    X = np.vstack([np.random.normal(0, 1, (100, 2)),
                   np.random.normal(3, 5, (30, 2)),
                   np.random.normal(-2, 2, (20, 2))])

    y = np.array([1] * 100 + [2] * 30 + [3] * 20)
    ssm = StaticSMOTE()
    X_resampled, y_resampled = ssm.fit_transform(X, y)
    cnt = Counter(y_resampled)
    assert cnt[1] == 100
    assert cnt[2] == 60
    assert cnt[3] == 80
