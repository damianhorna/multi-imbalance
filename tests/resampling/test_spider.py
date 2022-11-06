from collections import Counter

import numpy as np

from multi_imbalance.resampling.spider import SPIDER3

cost = np.ones((3, 3))
np.fill_diagonal(cost, 0)
spider = SPIDER3(1, maj_int_min={"maj": ["MAJ"], "int": ["INT"], "min": ["MIN"]}, cost=cost)


def test_knn():
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1], [0, 0]]).astype(object)

    y = np.array(["MIN", "MIN", "MAJ", "MAJ", "MAJ"])

    DS = np.append(X, y.reshape(y.shape[0], 1), axis=1)

    assert (DS[4] == spider._knn(DS[0], DS)).all()


def test_min_cost_classes():
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1], [0, 0]]).astype(object)

    y = np.array(["MIN", "MIN", "MAJ", "MAJ", "MAJ"])

    DS = np.append(X, y.reshape(y.shape[0], 1), axis=1)
    spider._fit_resample(X, y)
    spider._min_cost_classes(DS[0], DS)
    assert (spider._min_cost_classes(DS[0], DS) == ["MAJ"]).all()
    assert (spider._min_cost_classes(DS[4], DS) == ["MIN", "MAJ"]).all()


def test_estimate_cost_matrix():
    y = [0, 1, 1, 2, 2, 2, 2, 2, 2]
    cost = SPIDER3._estimate_cost_matrix(y).ravel().tolist()
    assert cost == [0, 1, 1, 2, 0, 1, 6, 3, 0]


def test_fit_resample():
    np.random.seed(7)
    X = np.vstack(
        [
            np.random.normal(0, 1, (100, 2)),
            np.random.normal(3, 5, (30, 2)),
            np.random.normal(-2, 2, (20, 2)),
        ]
    )

    y = np.array([1] * 100 + [2] * 30 + [3] * 20)
    sp = SPIDER3(5)
    X_resampled, y_resampled = sp.fit_resample(X, y)
    cnt = Counter(y_resampled)
    assert cnt[1] == 72
    assert cnt[2] == 57
    assert cnt[3] == 30
