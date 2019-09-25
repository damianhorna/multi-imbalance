"""Test the datasets loader.
"""

from multi_imbalance.datasets import load_datasets

DATASET_SHAPE = {
    'balance_scale': (625, 4),
    'cleveland': (297, 13),
    'cmc': (1473, 9),
    'dermatology': (358, 34),
    'ecoli': (336, 7),
    'glass': (214, 9),
    'hayes_roth': (132, 5),
    'new_thyroid': (215, 5),
    'winequailty_red': (1599, 11)
}


def test_load_datasets():
    print("Testing loading datasets")
    datasets = load_datasets(data_home="./data/")
    for k in DATASET_SHAPE.keys():
        X = datasets[k].data
        assert DATASET_SHAPE[k] == X.shape
