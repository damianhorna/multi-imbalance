import os
from collections import OrderedDict

import numpy as np

from multi_imbalance.utils.data import construct_flat_2pc_df, preprocess_dataset, load_arff_datasets


def test_2pc():
    x = np.array([[0, 1], [1, 2]])
    y = np.array([9, 5])
    result = construct_flat_2pc_df(x, y).to_numpy()
    expected = np.array([[0, 1, 9], [1, 2, 5]])
    assert all((expected == result).flatten())


def test_preprocess():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds_path = os.path.join(dir_path,'ds_example.arrf')
    x, y, non_cat = preprocess_dataset(ds_path, return_non_cat_length=True)
    assert all(y == np.array([0, 0, 0, 0, 0, 0, 0]))
    assert non_cat == 2
    assert x.shape == (7, 2)


def test_load_arff_datasets():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds_paths = [os.path.join(dir_path,'ds_example.arrf')]
    datasets = load_arff_datasets(return_non_cat_length=False, dataset_paths=ds_paths)

    keys = list(datasets.keys())
    assert type(datasets) == OrderedDict
    assert 'ds_example' in keys
    assert len(keys) == 1

    for k in ['data', 'target', 'DESCR']:
        assert k in list(datasets['ds_example'].keys())


def test_load_arff_datasets_wth_non_cats():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds_paths = [os.path.join(dir_path,'ds_example.arrf')]
    datasets = load_arff_datasets(return_non_cat_length=True, dataset_paths=ds_paths)
    keys = list(datasets.keys())
    assert type(datasets) == OrderedDict
    assert 'ds_example' in keys
    assert len(keys) == 1

    for k in ['data', 'target', 'DESCR', 'cat_length']:
        assert k in list(datasets['ds_example'].keys())
