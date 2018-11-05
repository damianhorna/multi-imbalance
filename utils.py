import math
from collections import Counter
import os

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sklearn.datasets


MISSING_VAL = np.nan
NOMINAL = 0
NUMERIC = 1
TYPE_MAPPING = {""}
DATA_TYPES = []


def read_dataset(src, excluded=[], skip_rows=0, na_values=[], normalize=True):
    """
    Reads in a dataset in csv format and stores it in a dataFrame.

    Parameters
    ----------
    src: string - path to input dataset
    excluded: list of int - 0-based indices of columns/features to be ignored
    skip_rows: int - number of rows to skip at the beginning of <src>
    na_values: list of str - values encoding missing values - these are represented by NaN
    normalize: bool - True if numeric values should be in range between 0 and 1

    Returns
    -------
    pd.DataFrame, dict - dataset and counts matrix which contains for nominal classes how often the value of an
    co-occurs with each class label

    """
    df = pd.read_csv(src, skiprows=skip_rows, na_values=na_values)
    # TODO: normalize numeric features to the range (0-1)
    if normalize:
       df = normalize(df)
    return df


def normalize(df):
    """Normalize numeric features"""
    for col_idx, _ in enumerate(df):
        col = df.iloc[:, col_idx]
        if is_numeric_dtype(col):
            min_val = col.min()
            max_val = col.max()
            df.iloc[:, col_idx] = (col - min_val) / (max_val - min_val)
    return df


def hvdm(xi, yi, counts):
    """
    Computes the distance (Heterogenous Value Difference Metrics) between a rule/example and another example.

    Parameters
    ----------
    xi: pd.dataFrame - (m x n or a x b), where a<=m, b<=n rule or example
    yi: pd.dataFrame - (m x n) example
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label

    Returns
    -------
    float - distance.

    """
    # Select same columns in both inputs
    # https://stackoverflow.com/questions/46228574/pandas-select-dataframe-columns-based-on-another-dataframes-columns
    long = yi[yi.columns.intersection(xi.columns)]
    short = xi[xi.columns.intersection(yi.columns)]
    print(xi.shape + ": " + xi.columns, yi.shape + ": " + yi.columns)
    dists = []
    # Compute distance for j-th feature (=column)
    for j, _ in enumerate(short):
        # Extract column from both dataframes into numpy array
        col1 = short.iloc[:, j].values
        col2 = long.iloc[:, j].values
        # Compute nominal/numeric distance
        if pd.api.types.is_numeric_dtype(short.columns[j]):
            dist = di(col1, col2)
        else:
            dist = svdm(col1, col2, counts)
        dists.append(dist*dist)
    # Compute HVDM
    return math.sqrt(sum(dists))


def svdm(f1, f2, counts):
    """
    Computes the Value difference metric for nominal values. Assumes that the data is normalized.

    Parameters
    ----------
    f1: pd.Series - features of input 1.
    f2: pd.Series - features of input 2.
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label

    Returns
    -------
    float - distance

    """
    # If NaN is included anywhere
    if f1.hasnans or f2.hashnans:
        # if f1.isnull().values.any() or f2.isnull().values.any():
        print("NaN(s) in svdm()")
        return 1.
    dist = 0.

    return dist


def di(f1, f2):
    """
    Computes the Euclidean distance for numeric values. Assumes that the data is normalized.

    Parameters
    ----------
    f1: pd.Series - features of input 1.
    f2: pd.Series - features of input 2.

    Returns
    -------
    float - distance

    """
    # If NaN is included anywhere
    if f1.hasnans or f2.hashnans:
        # if f1.isnull().values.any() or f2.isnull().values.any():
        print("NaN(s) in di()")
        return 1.
    dist = 0.

    return dist


def sklearn_to_df(sklearn_dataset):
    """Converts sklearn dataset into a pd.dataFrame."""
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['class'] = pd.Series(sklearn_dataset.target)
    return df


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Iris dataset
    df = sklearn_to_df(sklearn.datasets.load_iris())
    print(df)

    src = os.path.join(base_dir, "datasets", "iris.csv")
    dataset = read_dataset(src)
    print("own function")
    print(dataset)
    print(dataset.columns)

