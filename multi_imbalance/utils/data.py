import glob
from collections import OrderedDict, Counter
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch


def construct_flat_2pc_df(X, y) -> pd.DataFrame:
    """
    This function takes two dimensional X and one dimensional y arrays, concatenates and returns them as data frame

    :param X:
        two dimensional numpy array
    :param y:
        one dimensional numpy array with labels
    :return:
        Data frame with 3 columns x1 x2 and y and with number of rows equal to number of rows in X
    """
    y = pd.DataFrame({'y': y})
    X_df = pd.DataFrame(data=X, columns=['x1', 'x2'])

    df = pd.concat([X_df, y], axis=1)

    return df


def get_project_root() -> Path:  # pragma no cover
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def load_arff_dataset(path: str, one_hot_encode: bool = True, return_non_cat_length: bool = False):
    """
    Load and return the dataset saved in arff type file

    :param str path:
        location of dataset file
    :param bool one_hot_encode:
        flag, if true encodes categorical variables using OneHotEncoder
    :param bool return_non_cat_length:
        flag, if true returns the number of non categorical variables
    :returns:
        - ndarray X - dimensional numpy array where non categorical variables are stored in first columns followed by categorical variables
        - ndarray y - one dimensional numpy array with the classification target
        - bool non_cat_length - number of non categorical variables (only if return_non_cat_length=True)
    """
    data, meta = arff.loadarff(path)

    df = pd.DataFrame(data)
    y_index = len(df.columns) - 1
    y = df.pop(df.columns[y_index])

    le = LabelEncoder()
    y = le.fit_transform(y)

    categorical_feature_mask = df.dtypes == object

    categorical_cols = df.columns[categorical_feature_mask].tolist()
    non_categorical_cols = df.columns[~categorical_feature_mask].tolist()

    df[categorical_cols] = df[categorical_cols].replace({b'?': np.NaN})
    mode = df.mode().iloc[0]
    mean = df.filter(non_categorical_cols).mean()

    df[categorical_cols] = df.filter(categorical_cols).fillna(mode)
    df[non_categorical_cols] = df.filter(non_categorical_cols).fillna(mean)

    if one_hot_encode:
        X = pd.get_dummies(df, columns=categorical_cols)
    else:
        col_list = non_categorical_cols + categorical_cols
        X = df[col_list]

    if return_non_cat_length:
        return X.to_numpy(), y, len(non_categorical_cols)
    else:
        return X.to_numpy(), y


def load_datasets_arff(return_non_cat_length=False, dataset_paths=None):
    if dataset_paths is None:
        dataset_paths = glob.glob(f'{get_project_root()}/data/arff/*')

    datasets = OrderedDict()
    for path in sorted(dataset_paths):
        dataset_file = path.split('/')[-1]
        dataset_name = dataset_file.split('.')[0]
        if return_non_cat_length:
            X, y, cat_length = load_arff_dataset(path, return_non_cat_length=return_non_cat_length)
            datasets[dataset_name] = Bunch(data=X, target=y, non_cat_length=cat_length, DESCR=dataset_name)
        else:
            X, y = load_arff_dataset(path, return_non_cat_length=return_non_cat_length)
            datasets[dataset_name] = Bunch(data=X, target=y, DESCR=dataset_name)

    return datasets


def construct_maj_int_min(y: np.ndarray, strategy='median') -> OrderedDict:
    """
    This function creates dictionary with information which classes are minority or majority

    :param y:
        One dimensional numpy array that contains class labels
    :param strategy:
        The principle according to which the division into minority and majority classes will be determined:

        * 'median':
            A class whose size is equal to the median of the class sizes will be considered "intermediate"
        * 'average':
            The average class size will be calculated, all classes that are smaller will be considered as minority and
            the rest will be considered majority
    :return:
        dictionary with keys 'maj', 'int', 'min. The value for each key is a list containing the class labels belonging
        to the given group
    """
    class_sizes = Counter(y)

    if strategy == 'median':
        middle_size = median(list(class_sizes.values()))
    elif strategy == 'average':
        middle_size = np.mean(list(class_sizes.values()))
    else:
        raise ValueError(f'Unrecognized {strategy}. Only "median" and "average" are allowed.')

    maj_int_min = OrderedDict({
        'maj': list(),
        'int': list(),
        'min': list()
    })
    for class_label, class_size in class_sizes.items():
        if class_size == middle_size:
            class_group = 'int'
        elif class_size < middle_size:
            class_group = 'min'
        else:
            class_group = 'maj'

        maj_int_min[class_group].append(class_label)

    return maj_int_min
