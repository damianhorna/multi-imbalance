import pytest
from multi_imbalance.resampling.spider import SPIDER3
import numpy as np

cost = np.ones((3,3))
np.fill_diagonal(cost, 0)
spider = SPIDER3(1, cost, majority_classes=["MAJ"], intermediate_classes=["INT"], minority_classes=["MIN"])


def test_union():
    arr1 = np.array([[1, 2, 3]])
    arr2 = np.array([[4, 5, 6]])
    actual = spider._union(arr1, arr2)
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    assert (actual == expected).all()

    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[1, 2, 3]])

    actual = spider._union(arr1, arr2)
    expected = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])

    assert (actual == expected).all()


def test_intersect():
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[1, 2, 3]])

    actual = spider._intersect(arr1, arr2)
    expected = np.array([[1, 2, 3]])

    assert (actual == expected).all()


def test_setdiff():
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[1, 2, 3]])

    actual = spider._setdiff(arr1, arr2)
    expected = np.array([[4, 5, 6]])

    assert (actual == expected).all()


def test_knn():
    X = np.array([
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
        [0, 0]
    ]).astype(object)

    y = np.array(["MIN", "MIN", "MAJ", "MAJ", "MAJ"])

    DS = np.append(X, y.reshape(y.shape[0], 1), axis=1)
    assert (DS[4] == spider._knn(DS[0], DS, c="MAJ")).all()
    assert (DS[4] == spider._knn(DS[2], DS, c="MAJ")).all()
    assert (DS[0] == spider._knn(DS[4], DS, c="MIN")).all()


def test_nearest():
    X = np.array([
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
        [0, 0]
    ]).astype(object)

    y = np.array(["MIN", "MIN", "MAJ", "MAJ", "MAJ"])

    DS = np.append(X, y.reshape(y.shape[0], 1), axis=1)
    assert (DS[4] == spider._nearest(DS[0], DS)).all()
    assert (DS[4] == spider._nearest(DS[3], DS)).all()
    assert (DS[4] == spider._nearest(DS[1], DS)).all()


def test_min_cost_classes():
    X = np.array([
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
        [0, 0]
    ]).astype(object)

    y = np.array(["MIN", "MIN", "MAJ", "MAJ", "MAJ"])

    DS = np.append(X, y.reshape(y.shape[0], 1), axis=1)
    spider._min_cost_classes(DS[0], DS)
    assert (spider._min_cost_classes(DS[0], DS) == ["MAJ"]).all()
    assert (spider._min_cost_classes(DS[4], DS) == ["MIN"]).all()
