import os
from collections import OrderedDict

import numpy as np
import pytest

from multi_imbalance.utils.data import (
    construct_flat_2pc_df,
    load_arff_dataset,
    load_datasets_arff,
    construct_maj_int_min,
)


def test_2pc():
    x = np.array([[0, 1], [1, 2]])
    y = np.array([9, 5])
    result = construct_flat_2pc_df(x, y).to_numpy()
    expected = np.array([[0, 1, 9], [1, 2, 5]])
    assert all((expected == result).flatten())


def test_preprocess():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds_path = os.path.join(dir_path, "ds_example.arrf")
    x, y, non_cat = load_arff_dataset(ds_path, return_non_cat_length=True)
    assert all(y == np.array([0, 0, 0, 0, 0, 0, 0]))
    assert non_cat == 2
    assert x.shape == (7, 2)


def test_preprocess_without_one_hot_encode():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds_path = os.path.join(dir_path, "ds_example.arrf")
    x, y, non_cat = load_arff_dataset(
        ds_path, return_non_cat_length=True, one_hot_encode=False
    )
    assert all(y == np.array([0, 0, 0, 0, 0, 0, 0]))
    assert non_cat == 2
    assert x.shape == (7, 2)


def test_load_arff_datasets():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds_paths = [os.path.join(dir_path, "ds_example.arrf")]
    datasets = load_datasets_arff(return_non_cat_length=False, dataset_paths=ds_paths)

    keys = list(datasets.keys())
    assert type(datasets) == OrderedDict
    assert "ds_example" in keys
    assert len(keys) == 1

    for k in ["data", "target", "DESCR"]:
        assert k in list(datasets["ds_example"].keys())


def test_load_arff_datasets_wth_non_cats():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds_paths = [os.path.join(dir_path, "ds_example.arrf")]
    datasets = load_datasets_arff(return_non_cat_length=True, dataset_paths=ds_paths)
    keys = list(datasets.keys())
    assert type(datasets) == OrderedDict
    assert "ds_example" in keys
    assert len(keys) == 1

    for k in ["data", "target", "DESCR", "non_cat_length"]:
        assert k in list(datasets["ds_example"].keys())


def test_construct_maj_int_min_when_correct_and_median_strategy():
    class_sizes = {
        0: 5,  # class 0 occurs 5 times in dataset
        1: 6,
        3: 7,  # median
        5: 10,
        8: 12,
    }
    y = np.array(
        [
            class_label
            for class_label, class_size in class_sizes.items()
            for _ in range(class_size)
        ]
    )
    np.random.shuffle(y)

    maj_int_dict = construct_maj_int_min(y, strategy="median")

    assert len(maj_int_dict["int"]) == 1
    assert maj_int_dict["int"][0] == 3

    assert len(maj_int_dict["min"]) == 2
    assert all(i in maj_int_dict["min"] for i in [0, 1])

    assert len(maj_int_dict["maj"]) == 2
    assert all(i in maj_int_dict["maj"] for i in [5, 8])


def test_construct_maj_int_min_when_correct_and_average_strategy():
    class_sizes = {
        0: 5,  # class 0 occurs 5 times in dataset
        1: 6,
        3: 7,
        5: 10,
        8: 2000,
    }
    y = np.array(
        [
            class_label
            for class_label, class_size in class_sizes.items()
            for _ in range(class_size)
        ]
    )
    np.random.shuffle(y)

    maj_int_dict = construct_maj_int_min(y, strategy="average")

    assert len(maj_int_dict["int"]) == 0

    assert len(maj_int_dict["min"]) == 4
    assert all(i in maj_int_dict["min"] for i in [0, 1, 3, 5])

    assert len(maj_int_dict["maj"]) == 1
    assert maj_int_dict["maj"][0] == 8


def test_construct_maj_int_min_when_wrong_strategy():
    class_sizes = {
        0: 5,  # class 0 occurs 5 times in dataset
        1: 6,
        3: 7,
        5: 10,
        8: 2000,
    }
    y = np.array(
        [
            class_label
            for class_label, class_size in class_sizes.items()
            for _ in range(class_size)
        ]
    )
    np.random.shuffle(y)

    with pytest.raises(ValueError):
        construct_maj_int_min(y, strategy="WRONG_STRATEGY")
