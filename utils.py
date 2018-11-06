import math
from collections import Counter, defaultdict
import os
import itertools

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sklearn.datasets


MISSING_VAL = np.nan
NOMINAL = 0
NUMERIC = 1
TYPE_MAPPING = {""}
DATA_TYPES = []
CONDITIONAL = "Conditional"


def read_dataset(src, excluded=[], skip_rows=0, na_values=[], normalize=False, class_index=-1):
    """
    Reads in a dataset in csv format and stores it in a dataFrame.

    Parameters
    ----------
    src: string - path to input dataset
    excluded: list of int - 0-based indices of columns/features to be ignored
    skip_rows: int - number of rows to skip at the beginning of <src>
    na_values: list of str - values encoding missing values - these are represented by NaN
    normalize: bool - True if numeric values should be in range between 0 and 1
    class_index: int - 0-based index where class label is stored in dataset. -1 = last column

    Returns
    -------
    pd.DataFrame, dict - dataset and counts matrix which contains for nominal classes how often the value of an
    co-occurs with each class label

    """
    df = pd.read_csv(src, skiprows=skip_rows, na_values=na_values)
    lookup = {}
    # Convert fancy index to regular index - otherwise the loop below won't skip the column with class labels
    if class_index == -1:
        class_index = len(df.columns) - 1
    # Create lookup matrix for nominal features for SVDM + normalize numerical features columnwise, but ignore labels
    for i, _ in enumerate(df):
        if i == class_index:
            continue
        col = df.iloc[:, i]
        if is_numeric_dtype(col):
            if normalize:
                df.iloc[:, i] = normalize_series(col)
        else:
            lookup[i] = create_svdm_lookup_column(df, col, class_index)
    return df, lookup


def normalize_dataframe(df):
    """Normalize numeric features (=columns) using min-max normalization"""
    for col_idx, _ in enumerate(df):
        col = df.iloc[:, col_idx]
        if is_numeric_dtype(col):
            min_val = col.min()
            max_val = col.max()
            df.iloc[:, col_idx] = (col - min_val) / (max_val - min_val)
    return df


def normalize_series(col):
    """Normalizes a given series assuming it's data type is numeric"""
    if is_numeric_dtype(col):
        min_val = col.min()
        max_val = col.max()
        col = (col - min_val) / (max_val - min_val)
    return col


def create_svdm_lookup_column(df, coli, class_idx):
    """
    Create sparse lookup table for the feature representing the current column i.

    N(xi), N(yi), N(xi, Kj), N(yi, Kj), is stored per nominal feature, where N(xi) and N(yi) are the numbers of
    examples for which the value on i-th feature (coli) is equal to xi and yi respectively, N(xi , Kj) and N(yi,
    Kj) are the numbers of examples from the decision class Kj , which belong to N(xi) and N(yi), respectively

    Parameters
    ----------
    df: pd.DataFrame - dataset.
    coli: pd.Series - i-th column (= feature) of the dataset
    class_idx: int - index of class label in <df>.

    Returns
    -------
    dict - sparse dictionary holding the non-zero counts of all values of <coli> with the class labels

    """
    # classes = df.iloc[:, class_idx].unique()
    c = {}
    # print("Kj:\n", classes)
    # print(coli)
    nxiyi = Counter(coli.values)
    c.update(nxiyi)
    c[CONDITIONAL] = {}
    # print("N(xi/yi)\n", nxiyi)
    unique_xiyi = nxiyi.keys()
    # Create all pairwise combinations of two values
    combinations = itertools.combinations(unique_xiyi, 2)
    # for i in range(0, len(unique_xiyi), 2):
    for combo in combinations:
        for val in combo:
            # print("current value:\n", val)
            rows_with_val = df.loc[coli == val]
            # print("rows with value:\n", rows_with_val)
            nxiyikj = Counter(rows_with_val.iloc[:, class_idx].values)
            # print("counts:\n", nxiyikj)
            c[CONDITIONAL][val] = nxiyikj
    return c


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
    for col_idx, _ in enumerate(df):
        col = df.iloc[:, col_idx]
        if is_numeric_dtype(col):
            min_val = col.min()
            max_val = col.max()
            df.iloc[:, col_idx] = (col - min_val) / (max_val - min_val)
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
    dataset, lookup = read_dataset(src)
    print("own function")
    print(dataset)
    print(dataset.columns)
    print(lookup)

