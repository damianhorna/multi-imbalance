import math
from collections import Counter
import os
import itertools
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import sklearn.datasets


MISSING_VAL = np.nan
NOMINAL = 0
NUMERIC = 1
TYPE_MAPPING = {""}
DATA_TYPES = []
CONDITIONAL = "Conditional"
CLASSES = []


def read_dataset(src, excluded=[], skip_rows=0, na_values=[], normalize=False, class_index=-1, header=True):
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
    header: bool - True if header row is included in the dataset else False

    Returns
    -------
    pd.DataFrame, pd.DataFrame, dict - dataset, initial rule set and counts matrix which contains for nominal classes
    how often the value of a feature co-occurs with each class label

    """
    # Add column names
    if header:
        df = pd.read_csv(src, skiprows=skip_rows, na_values=na_values)
    else:
        df = pd.read_csv(src, skiprows=skip_rows, na_values=na_values, header=None)
        df.columns = [i for i in range(len(df.columns))]
    lookup = {}
    # Convert fancy index to regular index - otherwise the loop below won't skip the column with class labels
    if class_index == -1:
        class_index = len(df.columns) - 1
    global CLASSES
    CLASSES = df.iloc[:, class_index].unique()
    class_col_name = df.columns[class_index]
    rules = extract_initial_rules(df, class_col_name)
    df = add_tags(df, class_col_name)
    # Create lookup matrix for nominal features for SVDM + normalize numerical features columnwise, but ignore labels
    for col_name in df:
        if col_name == class_col_name:
            continue
        col = df[col_name]
        if is_numeric_dtype(col):
            if normalize:
                df[col_name] = normalize_series(col)
        else:
            lookup[col_name] = create_svdm_lookup_column(df, col, class_col_name)
    return df, lookup


def extract_initial_rules(df, class_col_name):
    """
    Creates the initial rule set for a given dataset, which corresponds to the examples in the dataset, i.e.
    lower and upper bound (of numeric) features are the same. Note that for every feature in the dataset,
    two values are stored per rule, namely lower and upper bound. For example, if we have
    A   B ...                                                   A        B ...
    ---------- in the dataset, the corresponding rule stores:  ---------------
    1.0 x ...                                                  (1.0,1.0) x
    df: pd.DataFrame - dataset
    class_col_name: str - name of the column holding the class labels

    Returns
    -------
    pd.DataFrame.
    Rule set

    """
    rules = df.copy()
    for col_name in df:
        if col_name == class_col_name:
            continue
        col = df[col_name]
        if is_numeric_dtype(col):
            # Assuming the value is x, we now store tuples of (x, x) per row
            rules[col_name] = [tuple([row[col_name], row[col_name]]) for _, row in df.iterrows()]
    return rules


def add_tags(df, class_col_name):
    """
    Assigns each example in the dataset a tag, either "SAFE" or "UNSAFE", "NOISY", "BORDERLINE".
    SAFE: example is classified correctly when looking at its k neighbors
    UNSAFE: example is misclassified when looking at its k neighbors
    NOISY: example is UNSAFE and all its k neighbors belong to the opposite class
    BORDERLINE: example is UNSAFE and it's not NOISY

    Parameters
    ----------
    df: pd.DataFrame - dataset
    class_col_name: str - name of the column holding the class labels

    Returns
    -------
    pd.DataFrame.
    Dataset with an additional column containing the tag.

    """


def normalize_dataframe(df):
    """Normalize numeric features (=columns) using min-max normalization"""
    for col_name in df.columns:
        col = df[col_name]
        if is_numeric_dtype(col):
            min_val = col.min()
            max_val = col.max()
            df[col_name] = (col - min_val) / (max_val - min_val)
    return df


def normalize_series(col):
    """Normalizes a given series assuming it's data type is numeric"""
    if is_numeric_dtype(col):
        min_val = col.min()
        max_val = col.max()
        col = (col - min_val) / (max_val - min_val)
    return col


def create_svdm_lookup_column(df, coli, class_col_name):
    """
    Create sparse lookup table for the feature representing the current column i.

    N(xi), N(yi), N(xi, Kj), N(yi, Kj), is stored per nominal feature, where N(xi) and N(yi) are the numbers of
    examples for which the value on i-th feature (coli) is equal to xi and yi respectively, N(xi , Kj) and N(yi,
    Kj) are the numbers of examples from the decision class Kj , which belong to N(xi) and N(yi), respectively

    Parameters
    ----------
    df: pd.DataFrame - dataset.
    coli: pd.Series - i-th column (= feature) of the dataset
    class_col_name: str - name of class label in <df>.

    Returns
    -------
    dict - sparse dictionary holding the non-zero counts of all values of <coli> with the class labels

    """
    c = {}
    nxiyi = Counter(coli.values)
    c.update(nxiyi)
    c[CONDITIONAL] = {}
    # print("N(xi/yi)\n", nxiyi)
    unique_xiyi = nxiyi.keys()
    # Create all pairwise combinations of two values
    combinations = itertools.combinations(unique_xiyi, 2)
    for combo in combinations:
        for val in combo:
            # print("current value:\n", val)
            rows_with_val = df.loc[coli == val]
            # print("rows with value:\n", rows_with_val)
            # nxiyikj = Counter(rows_with_val.iloc[:, class_idx].values)
            nxiyikj = Counter(rows_with_val[class_col_name].values)
            print("counts:\n", nxiyikj)
            c[CONDITIONAL][val] = nxiyikj
    return c


def find_neighbors(df, k, rule, class_col_name):
    """
    Finds k nearest examples for a given rule with the same class label as the rule.

    Parameters
    ----------
    df: pd.DataFrame - dataset
    k: int - number of neighbors to consider
    rule: pd.Series - rule
    class_col_name: str - name of class label

    Returns
    -------
    list.
    k nearest examples for the given rule.

    """
    class_label = rule[class_col_name]
    examples_with_same_label = df.loc[df[class_col_name] == class_label]


def most_specific_generalization(example, rule, class_col_name, i):
    """
    Implements MostSpecificGeneralization() from the paper, i.e. Algorithm 2.

    Parameters
    ----------
    example: pd.Series - row from the dataset.
    rule: pd.Series - rule that will be potentially generalized.
    class_col_name: str - name of the column hold the class labels.
    i: int - row index.

    Returns
    -------
    pd.Series.
    Generalized rule

    """
    for col_name in example:
        if col_name == class_col_name:
            continue
        example_dtype = example[col_name].dtype
        # print("example feature:", example[col_name], type(example[col_name]), col_name)
        example_val = example[col_name][i]
        # print("example val:", example_val)
        if col_name in rule:
            # Cast object to tuple datatype -> this is only automatically done if it's not a string
            rule_val = (rule[col_name])
            # print("rule_val", rule_val, "\nrule type:", type(rule_val))
            if is_string_dtype(example_dtype) and example_val != rule_val:
                rule = rule.drop(labels=[col_name])
            elif is_numeric_dtype(example_dtype):
                if example_val > rule_val[1]:
                    # print("new upper limit", (rule_val[0], example_val))
                    rule[col_name] = (rule_val[0], example_val)
                elif example_val < rule_val[0]:
                    # print("new lower limit", (example_val, rule_val[1]))
                    rule[col_name] = (example_val, rule_val[1])
                    # print("updated:", rule)
    return rule


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
            dist = svdm(col1, col2, j, counts)
        dists.append(dist*dist)
    # Compute HVDM
    return math.sqrt(sum(dists))


def svdm(f1, f2, i, counts):
    """
    Computes the Value difference metric for nominal values. Assumes that the data is normalized.

    Parameters
    ----------
    f1: pd.Series - features of input 1.
    f2: pd.Series - features of input 2.
    i: int - index of <f1> and <f2> respectively in the dataset, i.e. in which column they are stored
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
    singles = set()
    for k in counts[i]:
        if k != CONDITIONAL:
            singles.add(k)
    combos = itertools.combinations(singles, 2)
    print(combos)
    dist = 0.
    for k in CLASSES:
        kl = k
        for val in combos:
            nxi = counts[i][val]
            nyi = counts[i][val]
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

