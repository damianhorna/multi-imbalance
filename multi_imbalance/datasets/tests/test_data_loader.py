"""Test the datasets loader.
"""

from multi_imbalance.datasets import load_datasets

DATASET_SHAPE = {
    '1czysty-cut': (1200, 2),
    '2delikatne-cut': (1200, 2),
    '3mocniej-cut': (1200, 2),
    '4delikatne-bezover-cut': (1200, 2),
    'balance-scale': (625, 4),
    'cleveland': (303, 13),
    'cleveland_v2': (303, 13),
    'cmc': (1473, 9),
    'dermatology': (366, 34),
    'glass': (214, 9),
    'hayes-roth': (160, 4),
    'new_ecoli': (336, 7),
    'new_led7digit': (500, 7),
    'new_vehicle': (846, 18),
    'new_winequality-red': (1599, 11),
    'new_yeast': (1484, 8),
    'thyroid-newthyroid': (215, 5)
}


def test_load_datasets():
    print("Testing loading datasets")
    datasets = load_datasets(data_home="./data/")
    for k in DATASET_SHAPE.keys():
        X = datasets[k].data
        assert DATASET_SHAPE[k] == X.shape
