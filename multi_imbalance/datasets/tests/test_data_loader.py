"""Test the datasets loader.
"""

from multi_imbalance.datasets import load_datasets

DATASET_SHAPE = {
    'ecoli': (336, 7),
    'optical_digits': (5620, 64),
    'satimage': (6435, 36),
    'pen_digits': (10992, 16),
    'abalone': (4177, 10),
    'sick_euthyroid': (3163, 42),
    'spectrometer': (531, 93),
    'car_eval_34': (1728, 21),
    'isolet': (7797, 617),
    'us_crime': (1994, 100),
    'yeast_ml8': (2417, 103),
    'scene': (2407, 294),
    'libras_move': (360, 90),
    'thyroid_sick': (3772, 52),
    'coil_2000': (9822, 85),
    'arrhythmia': (452, 278),
    'solar_flare_m0': (1389, 32),
    'oil': (937, 49),
    'car_eval_4': (1728, 21),
    'wine_quality': (4898, 11),
    'letter_img': (20000, 16),
    'yeast_me2': (1484, 8),
    'webpage': (34780, 300),
    'ozone_level': (2536, 72),
    'mammography': (11183, 6),
    'protein_homo': (145751, 74),
    'abalone_19': (4177, 10)
}


def test_load_datasets():
    print("Testing loading datasets")
    datasets = load_datasets(data_home="./data/")
    for k in DATASET_SHAPE.keys():
        X = datasets[k].data
        assert DATASET_SHAPE[k] == X.shape
