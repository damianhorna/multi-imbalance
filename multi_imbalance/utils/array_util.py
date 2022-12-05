import numpy as np
from typing import Optional, Tuple
import sklearn


def setdiff(arr1, arr2):
    """
    Performs the difference over two numpy arrays.

    :param arr1:
        Numpy array number 1.
    :param arr2:
        Numpy array number 2.
    :return:
        Result of the difference of arr1 and arr2.
    """

    for element in arr2:
        if contains(arr1, element):
            arr1 = np.delete(arr1, index_of(arr1, element), 0)
    return arr1


def union(arr1, arr2):
    """
    Performs the union over two numpy arrays
    (not removing duplicates, as it's how the algorithm SPIDER3 actually works).

    :param arr1:
        Numpy array number 1.
    :param arr2:
        Numpy array number 2.
    :return:
        The union of arr1 and arr2.
    """

    if arr1.size == 0:
        return arr2
    elif arr2.size == 0:
        return arr1
    else:
        return np.append(arr1, arr2, axis=0)


def contains(dataset, example):
    """
    Returns if dataset contains the example.
    :param dataset:
    :param example:
    :return: True or False depending on whether dataset contains the example.
    """
    for x in dataset:
        if all(x == example):
            return True
    return False


def index_of(arr, example):
    """
    :return: Index of learning exmaple in arr.
    """
    for i, x in enumerate(arr):
        if all(x == example):
            return i
    return -1


def intersect(arr1, arr2):
    """
    Performs the intersection operation over two numpy arrays (not removing duplicates).

    :param arr1:
        Numpy array number 1.
    :param arr2:
        Numpy array number 2.
    :return:
        The intersection of arr1 and arr2.
    """

    if arr1.size == 0 or arr2.size == 0:
        return np.array([])

    result = np.array([])
    for x in arr1:
        if contains(arr2, x):
            result = union(result, np.array([x]))
    return result


def shuffle(*arrs: np.ndarray,
            state: Optional[np.random.RandomState] = None) -> Tuple[
    np.ndarray, ...]:
    """
    Shuffles rows of many arrays at once.

    Shuffles given arrays using a shuffled matrix of row indices.
    The number of rows in the given arrays should be the same.

    :param arrs:
        Numpy arrays to shuffle.
    :param state:
        Optional RandomState used to shuffle.
    :return:
        A tuple of shuffled copies of given arrays.
    """
    indices = np.arange(arrs[0].shape[0])
    sklearn.utils.check_random_state(state).shuffle(indices)
    return tuple(arr[indices] for arr in arrs)
