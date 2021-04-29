"""Collection of imbalanced datasets.

The characteristics of the available datasets are presented in the table below.

 ID    Name             Repository     Class distribution                                            #S       #F
 1     balance_scale    UCI            46.1 : 46.1 : 7.8                                             625      4
 2     cmc              UCI            42.7 : 34.7 : 22.6                                            1,473    9
 3     cleveland        UCI            53.9 : 18.2 : 11.8 : 11.8 : 4.4                               297      13
 4     dermatology      UCI            31   : 19.8 : 16.8 : 13.4 : 13.4 : 5.6                        358      34
 5     ecoli            UCI            42.6 : 22.9 : 15.5 : 10.4 : 6    : 1.5  : 0.6 : 0.6           336      7
 6     glass            UCI            35.5 : 32.7 : 13.6 : 8    : 6.1  : 4.2                        214      9
 7     hayes_roth       UCI            38.6 : 38.6 : 22.7                                            132      5
 8     new_thyroid      UCI            69.8 : 16.3 : 14                                              215      5
 9     winequailty_red  UCI            42.6 : 39.9 : 12.4 : 3.3  : 1.1 : 0.6                         1599     11
 10    yeast            UCI            31.2 : 28.9 : 16.4 : 11   : 3.4 : 3 : 2.5 : 2 : 1.4 : 0.3    1,484    9
"""
# TODO add missing characteristics

from collections import OrderedDict
import tarfile
from io import BytesIO
from os import makedirs
from os.path import join, isfile

import numpy as np

from sklearn.datasets._base import Bunch

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
    """
    Load the benchmark datasets.

    :param data_home: Default catalogue in which the data is stored in .tar.gz format.
    :returns:
        OrderedDict of Bunch object. Each Bunch object refered as dataset have the following attributes:

            * dataset.data :
                ndarray, shape (n_samples, n_features)
            * dataset.target :
                ndarray, shape (n_samples, )
            * dataset.DESCR :
                string Description of the each dataset.
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
