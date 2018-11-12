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
SAFE = "safe"
NOISY = "noisy"
BORDERLINE = "borderline"
TAG = "tag"
DISTANCE_MATRIX = {}


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
    pd.DataFrame, pd.DataFrame, dict, pd.DataFrame - dataset, initial rule set, counts matrix which contains for
    nominal classes how often the value of a feature co-occurs with each class label, min/max values per numeric column

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
    minmax = {}
    # Create lookup matrix for nominal features for SVDM + normalize numerical features columnwise, but ignore labels
    for col_name in df:
        if col_name == class_col_name:
            continue
        col = df[col_name]
        if is_numeric_dtype(col):
            if normalize:
                df[col_name] = normalize_series(col)
            minmax[col_name] = {"min": col.min(), "max": col.max()}
        else:
            lookup[col_name] = create_svdm_lookup_column(df, col, class_col_name)
    min_max = pd.DataFrame(minmax)
    return df, lookup, rules, min_max


def extract_initial_rules(df, class_col_name):
    """
    Creates the initial rule set for a given dataset, which corresponds to the examples in the dataset, i.e.
    lower and upper bound (of numeric) features are the same. Note that for every feature in the dataset,
    two values are stored per rule, namely lower and upper bound. For example, if we have
    A   B ...                                                   A        B ...
    ---------- in the dataset, the corresponding rule stores:  ---------------
    1.0 x ...                                                  (1.0,1.0) x

    Parameters
    ----------
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


def add_tags(df, k, class_col_name, counts, min_max, classes):
    """
    Assigns each example in the dataset a tag, either "SAFE" or "UNSAFE", "NOISY", "BORDERLINE".
    SAFE: example is classified correctly when looking at its k neighbors
    UNSAFE: example is misclassified when looking at its k neighbors
    NOISY: example is UNSAFE and all its k neighbors belong to the opposite class
    BORDERLINE: example is UNSAFE and it's not NOISY.
    Assumes that <df> contains at least 2 rows.

    Parameters
    ----------
    df: pd.DataFrame - dataset
    k: int - number of neighbors to consider
    class_col_name: str - name of class label
    counts: dict - lookup table for SVDM
    min_max: pd:DataFrame - contains min/max values per numeric feature
    classes: list of str - class labels in the dataset.

    Returns
    -------
    pd.DataFrame.
    Dataset with an additional column containing the tag.

    """
    converted_examples = extract_initial_rules(df, class_col_name)
    tags = []
    for idx, converted_example in converted_examples.iterrows():
        # Ignore current row
        ignored_rows = converted_examples.index.isin([idx])
        examples_for_pairwise_distance = df[~ignored_rows]
        if examples_for_pairwise_distance.shape[0] > 0:
            # print("pairwise distances for:\n{}".format(converted_example))
            # print("compute distance to:\n{}".format(examples_for_pairwise_distance))
            neighbors = find_neighbors(examples_for_pairwise_distance, k, converted_example, class_col_name, counts,
                                       min_max, classes, use_same_label=False)
            labels = Counter(neighbors[class_col_name].values)
            tag = assign_tag(labels, converted_example[class_col_name])
            tags.append(tag)
    df[TAG] = pd.Series(tags)
    return df


def assign_tag(labels, label):
    """
    Assigns a tag to an example ("safe", "noisy" or "borderline").

    Parameters
    ----------
    labels: collections.Counter - frequency of labels
    label: str - label of the example

    Returns
    -------
    string.
    Tag, either "safe", "noisy" or "borderline".

    """
    total_labels = sum(labels.values())
    frequencies = labels.most_common(2)
    # print(frequencies)
    most_common = frequencies[0]
    tag = SAFE
    if most_common[1] == total_labels and most_common[0] != label:
        tag = NOISY
    elif most_common[1] < total_labels:
        second_most_common = frequencies[1]
        # print("most common: {} 2nd most common: {}".format(most_common, second_most_common))

        # Tie
        if most_common[1] == second_most_common[1] or most_common[0] != label:
            tag = BORDERLINE
    # print("neighbor labels: {} vs. {}".format(labels, label))
    # print("tag:", tag)
    return tag


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
            # print("counts:\n", nxiyikj)
            c[CONDITIONAL][val] = nxiyikj
    return c


def find_neighbors(df, k, rule, class_col_name, counts, min_max, classes, use_same_label=True):
    """
    Finds k nearest examples for a given rule with the same class label as the rule.
    If less than k examples exist, a warning is issued.

    Parameters
    ----------
    df: pd.DataFrame - dataset
    k: int - number of neighbors to consider
    rule: pd.Series - rule
    class_col_name: str - name of class label
    counts: dict - lookup table for SVDM
    min_max: pd:DataFrame - contains min/max values per numeric feature
    classes: list of str - class labels in the dataset.
    use_same_label: bool - True if only examples with the same label as the rule should be considered as neighbors.
    Otherwise all labels are used.

    Returns
    -------
    pd.DataFrame.
    k nearest examples for the given rule.

    """
    if use_same_label:
        class_label = rule[class_col_name]
        examples_with_same_label = df.loc[df[class_col_name] == class_label]
    else:
        examples_with_same_label = df.copy()
    # print("examples:\n{}".format(examples_with_same_label))
    neighbors = examples_with_same_label.shape[0]
    # print("neighbors:", neighbors)
    if neighbors < k:
        warnings.warn("Only {} neighbors for {}".format(examples_with_same_label.shape[0], examples_with_same_label),
                      UserWarning)
    if neighbors > 0:
        dists = hvdm(examples_with_same_label, rule, counts, classes, min_max, class_col_name)
        neighbor_ids = dists.index[: k]
        # print("{} nearest neighbors:\n{}\n{}".format(k, dists, neighbor_ids))
        return df.loc[neighbor_ids]
    return None


def most_specific_generalization(example, rule, class_col_name, i):
    """
    Implements MostSpecificGeneralization() from the paper, i.e. Algorithm 2.

    Parameters
    ----------
    example: pd.Series - row from the dataset.
    rule: pd.Series - rule that will be potentially generalized.
    class_col_name: str - name of the column hold the class labels.
    i: int - row index of <example>.

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


def hvdm(examples, rule, counts, classes, min_max, class_col_name):
    """
    Computes the distance (Heterogenous Value Difference Metrics) between a rule/example and another example.
    Assumes that there's at least 1 feature shared between <rule> and <examples>.

    Parameters
    ----------
    examples: pd.DataFrame - examples
    rule: pd.Series - (m x n) rule
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label
    classes: list of str - class labels in the dataset.
    min_max: pd: pd.DataFrame - contains min/max values per numeric feature
    class_col_name: str - name of class label

    Returns
    -------
    pd.DataFrame - distances.

    """
    # Select only those columns that exist in examples and rule
    # https://stackoverflow.com/questions/46228574/pandas-select-dataframe-columns-based-on-another-dataframes-columns
    examples = examples[rule.index.intersection(examples.columns)]
    dists = []
    # Compute distance for j-th feature (=column)
    for col_name in examples:
        if col_name == class_col_name:
            continue
        # Extract column from both dataframes into numpy array
        example_feature_col = examples[col_name]
        # Compute nominal/numeric distance
        if pd.api.types.is_numeric_dtype(example_feature_col):
                dist_squared = di(example_feature_col, rule, min_max)
        else:
            dist_squared = svdm(example_feature_col, rule, counts, classes)
        dists.append(dist_squared)
    # Note: this line assumes that there's at least 1 feature
    distances = pd.DataFrame(list(zip(*dists)), columns=[s.name for s in dists], index=dists[0].index)
    # Sum up rows to compute HVDM - no need to square the distances as the order won't change
    distances["dist"] = distances.select_dtypes(float).sum(1)
    distances = distances.sort_values("dist", ascending=True)
    return distances


def svdm(example_feat, rule_feat, counts, classes):
    """
    Computes the (squared) Value difference metric for nominal values.

    Parameters
    ----------
    example_feat: pd.Series - column (=feature) containing all examples.
    rule_feat: pd.Series - column (=feature) of the rule.
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label
    classes: list of str - class labels in the dataset.

    Returns
    -------
    pd.Series.
    (squared) distance of each example.

    """
    # If NaN is included anywhere
    # if example_feat.hasnans or rule_feat.hashnans:
    if example_feat.isna().sum() > 0 or rule_feat.isna().sum() > 0:
        print("NaN(s) in svdm()")
        return 1.
    col_name = example_feat.name
    # Use all single value counts that don't depend on the class label
    singles = set()
    for k in counts[col_name]:
        if k != CONDITIONAL:
            singles.add(k)
    rule_val = rule_feat[col_name]
    n_rule = counts[col_name][rule_val]
    # print("value in rule", rule_val)
    dists = []
    # For every row
    for idx, example_val in example_feat.iteritems():
        # print("compute example", idx)
        # print("------------------")
        # print(example_val)
        dist = 0.
        for k in classes:
            # print("processing class", k)
            n_example = counts[col_name][example_val]
            nk_example = counts[col_name][CONDITIONAL][example_val][k]
            nk_rule = counts[col_name][CONDITIONAL][rule_val][k]
            # print("n_example", n_example)
            # print("nk_example", nk_example)
            # print("n_rule", n_rule)
            # print("nk_rule", nk_rule)
            res = abs(nk_example/n_example - nk_rule/n_rule)
            dist += res
            # print("|{}/{}-{}/{}| = {}".format(nk_example, n_example, nk_rule, n_rule, res))
            # print("d={}".format(dist))
        dists.append((idx, dist*dist))
    # print("distances:", dists)
    # Split tuples into 2 separate lists, one containing the indices and the other one containing the values
    zlst = list(zip(*dists))
    out = pd.Series(zlst[1], index=zlst[0], name=col_name)
    return out


def di(example_feats, rule_feat, min_max):
    """
    Computes the (squared) partial distance for numeric values between an example and a rule.

    Parameters
    ----------
    example_feats: pd.Series - column (=feature) containing all examples.
    rule_feat: pd.Series - column (=feature) of the rule.
    min_max: pd.DataFrame - min and max value per numeric feature.

    Returns
    -------
    pd.Series
    (squared) distance of each example.

    """
    # If NaN is included anywhere
    if example_feats.isna().sum() > 0 or rule_feat.isna().sum() > 0:
        # if f1.isnull().values.any() or f2.isnull().values.any():
        print("NaN(s) in di()")
        return 1.
    col_name = example_feats.name
    dists = []
    lower_rule_val, upper_rule_val = rule_feat[col_name]
    # Per row
    for idx, example_val in example_feats.iteritems():
        # print("processing", example_val)
        min_rule_val = min_max.at["min", col_name]
        max_rule_val = min_max.at["max", col_name]
        # print("min({})={}".format(col_name, min_rule_val))
        # print("max({})={}".format(col_name, max_rule_val))
        if example_val > upper_rule_val:
            # print("example > upper")
            # print("({} - {}) / ({} - {})".format(example_val, upper_rule_val, max_rule_val, min_rule_val))
            dist = (example_val - upper_rule_val) / (max_rule_val - min_rule_val)
        elif example_val < lower_rule_val:
            # print("example < lower")
            # print("({} - {}) / ({} - {})".format(lower_rule_val, example_val, max_rule_val, min_rule_val))
            dist = (lower_rule_val - example_val) / (max_rule_val - min_rule_val)
        else:
            dist = 0
        dists.append((idx, dist*dist))
    zlst = list(zip(*dists))
    out = pd.Series(zlst[1], index=zlst[0], name=col_name)
    return out


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
    class_col_name = "Class"
    k = 3
    classes = ["apple", "banana"]
    dataset, lookup, rules, min_max = read_dataset(src)
    df = add_tags(df, k, class_col_name, lookup, min_max, classes)
    print("own function")
    print(dataset)
    print(dataset.columns)
    print(lookup)

