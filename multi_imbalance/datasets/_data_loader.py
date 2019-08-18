from collections import OrderedDict
import tarfile
from io import BytesIO
from os import makedirs
from os.path import join, isfile

import numpy as np

from sklearn.datasets.base import Bunch

PRE_FILENAME = 'x'
POST_FILENAME = 'data.npz'

MAP_NAME_ID_KEYS = [
    'ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone',
    'sick_euthyroid', 'spectrometer', 'car_eval_34', 'isolet', 'us_crime',
    'yeast_ml8', 'scene', 'libras_move', 'thyroid_sick', 'coil_2000',
    'arrhythmia', 'solar_flare_m0', 'oil', 'car_eval_4', 'wine_quality',
    'letter_img', 'yeast_me2', 'webpage', 'ozone_level', 'mammography',
    'protein_homo', 'abalone_19'
]

MAP_NAME_ID = OrderedDict()
MAP_ID_NAME = OrderedDict()
for v, k in enumerate(MAP_NAME_ID_KEYS):
    MAP_NAME_ID[k] = v + 1
    MAP_ID_NAME[v + 1] = k


def fetch_datasets():

    data_home = "./../../data/"
    extracted_dir = join(data_home, "extracted")
    datasets = OrderedDict()

    filter_data_ = MAP_NAME_ID.keys()

    for it in filter_data_:
        filename = PRE_FILENAME + str(MAP_NAME_ID[it]) + POST_FILENAME
        filename = join(extracted_dir, filename)
        available = isfile(filename)

        if not available:
            makedirs(extracted_dir, exist_ok=True)
            with open(f'{data_home}data.tar.gz', 'rb') as fin:
                f = BytesIO(fin.read())
            tar = tarfile.open(fileobj=f)
            tar.extractall(path=extracted_dir)

        data = np.load(filename)
        X, y = data['data'], data['label']

        datasets[it] = Bunch(data=X, target=y, DESCR=it)

    return datasets
