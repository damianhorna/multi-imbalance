"""Collection of imbalanced datasets.

This collection of datasets has been proposed in [1]_. The
characteristics of the available datasets are presented in the table
below.

 ID    Name           Repository & Target           Ratio  #S       #F
 1     ecoli          UCI, target: imU              8.6:1  336      7
 2     optical_digits UCI, target: 8                9.1:1  5,620    64
 3     satimage       UCI, target: 4                9.3:1  6,435    36
 4     pen_digits     UCI, target: 5                9.4:1  10,992   16
 5     abalone        UCI, target: 7                9.7:1  4,177    10
 6     sick_euthyroid UCI, target: sick euthyroid   9.8:1  3,163    42
 7     spectrometer   UCI, target: >=44             11:1   531      93
 8     car_eval_34    UCI, target: good, v good     12:1   1,728    21
 9     isolet         UCI, target: A, B             12:1   7,797    617
 10    us_crime       UCI, target: >0.65            12:1   1,994    100
 11    yeast_ml8      LIBSVM, target: 8             13:1   2,417    103
 12    scene          LIBSVM, target: >one label    13:1   2,407    294
 13    libras_move    UCI, target: 1                14:1   360      90
 14    thyroid_sick   UCI, target: sick             15:1   3,772    52
 15    coil_2000      KDD, CoIL, target: minority   16:1   9,822    85
 16    arrhythmia     UCI, target: 06               17:1   452      278
 17    solar_flare_m0 UCI, target: M->0             19:1   1,389    32
 18    oil            UCI, target: minority         22:1   937      49
 19    car_eval_4     UCI, target: vgood            26:1   1,728    21
 20    wine_quality   UCI, wine, target: <=4        26:1   4,898    11
 21    letter_img     UCI, target: Z                26:1   20,000   16
 22    yeast_me2      UCI, target: ME2              28:1   1,484    8
 23    webpage        LIBSVM, w7a, target: minority 33:1   34,780   300
 24    ozone_level    UCI, ozone, data              34:1   2,536    72
 25    mammography    UCI, target: minority         42:1   11,183   6
 26    protein_homo   KDD CUP 2004, minority        111:1  145,751  74
 27    abalone_19     UCI, target: 19               130:1  4,177    10

References
----------
.. [1] Ding, Zejin, "Diversified Ensemble Classifiers for Highly
   Imbalanced Data Learning and their Application in Bioinformatics."
   Dissertation, Georgia State University, (2011).

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

        Notes
        -----
        This collection of datasets have been proposed in [1]_. The
        characteristics of the available datasets are presented in the table
        below.

        +--+--------------+-------------------------------+-------+---------+-----+
        |ID|Name          | Repository & Target           | Ratio | #S      | #F  |
        +==+==============+===============================+=======+=========+=====+
        |1 |ecoli         | UCI, target: imU              | 8.6:1 | 336     | 7   |
        +--+--------------+-------------------------------+-------+---------+-----+
        |2 |optical_digits| UCI, target: 8                | 9.1:1 | 5,620   | 64  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |3 |satimage      | UCI, target: 4                | 9.3:1 | 6,435   | 36  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |4 |pen_digits    | UCI, target: 5                | 9.4:1 | 10,992  | 16  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |5 |abalone       | UCI, target: 7                | 9.7:1 | 4,177   | 10  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |6 |sick_euthyroid| UCI, target: sick euthyroid   | 9.8:1 | 3,163   | 42  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |7 |spectrometer  | UCI, target: >=44             | 11:1  | 531     | 93  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |8 |car_eval_34   | UCI, target: good, v good     | 12:1  | 1,728   | 21  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |9 |isolet        | UCI, target: A, B             | 12:1  | 7,797   | 617 |
        +--+--------------+-------------------------------+-------+---------+-----+
        |10|us_crime      | UCI, target: >0.65            | 12:1  | 1,994   | 100 |
        +--+--------------+-------------------------------+-------+---------+-----+
        |11|yeast_ml8     | LIBSVM, target: 8             | 13:1  | 2,417   | 103 |
        +--+--------------+-------------------------------+-------+---------+-----+
        |12|scene         | LIBSVM, target: >one label    | 13:1  | 2,407   | 294 |
        +--+--------------+-------------------------------+-------+---------+-----+
        |13|libras_move   | UCI, target: 1                | 14:1  | 360     | 90  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |14|thyroid_sick  | UCI, target: sick             | 15:1  | 3,772   | 52  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |15|coil_2000     | KDD, CoIL, target: minority   | 16:1  | 9,822   | 85  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |16|arrhythmia    | UCI, target: 06               | 17:1  | 452     | 278 |
        +--+--------------+-------------------------------+-------+---------+-----+
        |17|solar_flare_m0| UCI, target: M->0             | 19:1  | 1,389   | 32  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |18|oil           | UCI, target: minority         | 22:1  | 937     | 49  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |19|car_eval_4    | UCI, target: vgood            | 26:1  | 1,728   | 21  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |20|wine_quality  | UCI, wine, target: <=4        | 26:1  | 4,898   | 11  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |21|letter_img    | UCI, target: Z                | 26:1  | 20,000  | 16  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |22|yeast_me2     | UCI, target: ME2              | 28:1  | 1,484   | 8   |
        +--+--------------+-------------------------------+-------+---------+-----+
        |23|webpage       | LIBSVM, w7a, target: minority | 33:1  | 34,780  | 300 |
        +--+--------------+-------------------------------+-------+---------+-----+
        |24|ozone_level   | UCI, ozone, data              | 34:1  | 2,536   | 72  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |25|mammography   | UCI, target: minority         | 42:1  | 11,183  | 6   |
        +--+--------------+-------------------------------+-------+---------+-----+
        |26|protein_homo  | KDD CUP 2004, minority        | 111:1 | 145,751 | 74  |
        +--+--------------+-------------------------------+-------+---------+-----+
        |27|abalone_19    | UCI, target: 19               | 130:1 | 4,177   | 10  |
        +--+--------------+-------------------------------+-------+---------+-----+

        References
        ----------
        .. [1] Ding, Zejin, "Diversified Ensemble Classifiers for Highly
           Imbalanced Data Learning and their Application in Bioinformatics."
           Dissertation, Georgia State University, (2011).
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
