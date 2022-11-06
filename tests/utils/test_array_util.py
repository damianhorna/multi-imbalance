import pytest
import numpy as np


from multi_imbalance.utils.array_util import setdiff, union, intersect, index_of


@pytest.mark.parametrize(
    "arr1, arr2, expected",
    [
        (
            [[1, 2, 3]],
            [[4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3]],
            [[1, 2, 3], [4, 5, 6], [1, 2, 3]],
        ),
        (
            [[1, 2, 3]],
            [],
            [[1, 2, 3]],
        ),
    ],
)
def test_union(arr1, arr2, expected):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    actual = union(arr1, arr2)
    expected = np.array(expected)
    assert (actual == expected).all()


@pytest.mark.parametrize(
    "arr1, arr2, expected",
    [
        (
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3]],
            [[1, 2, 3]],
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [],
            [],
        ),
    ],
)
def test_intersect(arr1, arr2, expected):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    actual = intersect(arr1, arr2)
    expected = np.array(expected)

    assert (actual == expected).all()


@pytest.mark.parametrize(
    "arr1, arr2, expected",
    [
        (
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3]],
            [[4, 5, 6]],
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [],
            [[1, 2, 3], [4, 5, 6]],
        ),
    ],
)
def test_setdiff(arr1, arr2, expected):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    actual = setdiff(arr1, arr2)
    expected = np.array(expected)

    assert (actual == expected).all()


@pytest.mark.parametrize(
    "arr1, arr2, expected",
    [
        (
            [[1, 2, 3], [4, 5, 6]],
            [1, 2, 3],
            0,
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [4, 5, 6],
            1,
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [7, 8, 9],
            -1,
        ),
    ],
)
def test_index_of(arr1, arr2, expected):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    actual = index_of(arr1, arr2)
    expected = np.array(expected)

    assert (actual == expected).all()
