"""Collection of imbalanced datasets.

The characteristics of the available datasets are presented in the table below.

 ID    Name             Repository     Class distribution                                            #S       #F


"""

from collections import OrderedDict
import tarfile
from io import BytesIO
from os import makedirs
from os.path import join, isfile

import numpy as np

from sklearn.datasets.base import Bunch

PRE_FILENAME = 'x'
POST_FILENAME = 'data.npz'
DATA_HOME_BASIC = "./../../data/"

MAP_NAME_ID_KEYS = ['1czysty-cut', '2delikatne-cut', '3mocniej-cut', '4delikatne-bezover-cut',
                    'balance-scale', 'cleveland', 'cleveland_v2', 'cmc', 'dermatology',
                    'glass', 'hayes-roth', 'new_ecoli', 'new_led7digit', 'new_vehicle',
                    'new_winequality-red', 'new_yeast', 'thyroid-newthyroid']

MAP_NAME_ID = OrderedDict()
MAP_ID_NAME = OrderedDict()
for v, k in enumerate(MAP_NAME_ID_KEYS):
    MAP_NAME_ID[k] = v + 1
    MAP_ID_NAME[v + 1] = k


def load_datasets(data_home=DATA_HOME_BASIC):
    """Load the benchmark datasets.
        Parameters
        -------
        data_home : Default catalogue in which the data is stored in .tar.gz format.

        Returns
        -------
        datasets : OrderedDict of Bunch object,
            The ordered is defined by ``filter_data``. Each Bunch object ---
            refered as dataset --- have the following attributes:

        dataset.data : ndarray, shape (n_samples, n_features)

        dataset.target : ndarray, shape (n_samples, )

        dataset.DESCR : string
            Description of the each dataset.

    """
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
