from collections import Counter, deque
import os
import itertools
import warnings
import copy
from operator import itemgetter

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import sklearn.datasets

import scripts.vars as my_vars


class MyException(Exception):
    pass


def read_dataset(src, positive_class, excluded=[], skip_rows=0, na_values=[], normalize=False, class_index=-1, header=True):
    """
    Reads in a dataset in csv format and stores it in a dataFrame.

    Parameters
    ----------
    src: str - path to input dataset
    positive_class: str - name of the minority class. The rest is treated as negative.
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
    my_vars.positive_class = positive_class
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
    my_vars.CLASSES = df.iloc[:, class_index].unique()
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


def add_tags_and_extract_rules(df, k, class_col_name, counts, min_max, classes):
    """
    Extracts initial rules and assigns each example in the dataset a tag, either "SAFE" or "UNSAFE", "NOISY",
    "BORDERLINE".

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
    pd.DataFrame, list of pd.Series.
    Dataset with an additional column containing the tag, initially extracted rules.

    """
    rules_df = extract_initial_rules(df, class_col_name)
    # 1 rule per example
    assert(rules_df.shape[0] == df.shape[0])
    my_vars.seed_mapping = dict((x, {x}) for x in range(rules_df.shape[0]))
    my_vars.examples_covered_by_rule = dict((x, {x}) for x in range(rules_df.shape[0]))
    rules = []
    for i, rule in rules_df.iterrows():
        rules.append(rule)
    tagged = add_tags(df, k, rules, class_col_name, counts, min_max, classes)
    rules = deque(rules)
    return tagged, rules


def add_tags(df, k, rules, class_col_name, counts, min_max, classes):
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
    rules: list of pd.Series - list of rules
    class_col_name: str - name of class label
    counts: dict - lookup table for SVDM
    min_max: pd:DataFrame - contains min/max values per numeric feature
    classes: list of str - class labels in the dataset.

    Returns
    -------
    pd.DataFrame.
    Dataset with an additional column containing the tag.

    """
    tags = []
    for rule in rules:
        rule_id = rule.name
        # Ignore current row
        examples_for_pairwise_distance = df.loc[df.index != rule_id]
        if examples_for_pairwise_distance.shape[0] > 0:
            # print("pairwise distances for:\n{}".format(rule))
            # print("compute distance to:\n{}".format(examples_for_pairwise_distance))
            neighbors, _ = find_nearest_examples(examples_for_pairwise_distance, k, rule, class_col_name, counts,
                                                 min_max, classes, label_type=my_vars.ALL_LABELS,
                                                 only_uncovered_neighbors=False)
            # print("neighbors:\n{}".format(neighbors))
            labels = Counter(neighbors[class_col_name].values)
            tag = assign_tag(labels, rule[class_col_name])
            tags.append(tag)
    df[my_vars.TAG] = pd.Series(tags)
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
    tag = my_vars.SAFE
    if most_common[1] == total_labels and most_common[0] != label:
        tag = my_vars.NOISY
    elif most_common[1] < total_labels:
        second_most_common = frequencies[1]
        # print("most common: {} 2nd most common: {}".format(most_common, second_most_common))

        # Tie
        if most_common[1] == second_most_common[1] or most_common[0] != label:
            tag = my_vars.BORDERLINE
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
    c[my_vars.CONDITIONAL] = {}
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
            c[my_vars.CONDITIONAL][val] = nxiyikj
    return c


def does_rule_cover_example(example, rule, dtypes):
    """
    Tests if a rule covers a given example.

    Parameters
    ----------
    example: pd.Series - example
    rule: pd.Series - rule
    dtypes: pd.Series - data types of the respective columns in the dataset

    Returns
    -------
    bool.
    True if the rule covers the example else False.

    """
    is_covered = True
    for (col_name, example_val), dtype in zip(example.iteritems(), dtypes):
        example_dtype = dtype
        if col_name in rule:
            # Cast object to tuple datatype -> this is only automatically done if it's not a string
            rule_val = (rule[col_name])
            # print("rule_val", rule_val, "\nrule type:", type(rule_val))
            if is_string_dtype(example_dtype) and example_val != rule_val:
                is_covered = False
                break
            elif is_numeric_dtype(example_dtype):
                if rule_val[0] > example_val or rule_val[1] < example_val:
                    is_covered = False
                    break
    return is_covered


def is_empty(df):
    """Tests if a pd.DataFrame is empty or not"""
    return len(df.index) == 0


def find_nearest_examples(df, k, rule, class_col_name, counts, min_max, classes, label_type=my_vars.ALL_LABELS,
                          only_uncovered_neighbors=True):
    """
    Finds k-nearest examples for a given rule with the same class label as the rule.
    If less than k examples exist, a warning is issued.

    Parameters
    ----------
    df: pd.DataFrame - dataset
    k: int - number of neighbors to consider
    rule: pd.Series - rule
    class_col_name: str - name of class label
    counts: dict - lookup table for SVDM
    min_max: pd:DataFrame - contains min/max values per numeric feature
    classes: list of str - class labels in the dataset. It's assumed to be binary.
    label_type: str - consider only examples of the specified type as neighbors. Valid values:
    scripts.vars.ALL_LABELS - ignore the label and choose the k-nearest examples across all class labels
    scripts.vars.SAME_LABEL_AS_RULE - consider only examples as k-nearest examples with they have the same label as
    <rule>
    scripts.vars.OPPOSITE_LABEL_TO_RULE - consider only examples as k-nearest examples with they have the opposite
    label of <rule>
    only_uncovered_neighbors: bool - True if only examples should be considered that aren't covered by <rule> yet.
    Otherwise, all neighbors are considered. An example is covered by a rule if the example satisfies all conditions
    imposed by <rule>.

    Raises
    ------

    Returns
    -------
    pd.DataFrame, pd.DataFrame OR None, None
    k-nearest examples for the given rule, distances of the k nearest examples. Returns None, None if there are no
    neighbors.

    """
    # print("find neighbors with same label as rule ({}) and which aren't covered by the rule yet ({})"
    #       .format(use_same_label, only_uncovered_neighbors))
    class_label = rule[class_col_name]
    if label_type == my_vars.ALL_LABELS:
        examples_with_same_label = df.copy()
    elif label_type == my_vars.OPPOSITE_LABEL_TO_RULE:
        opposite_label = classes[0] if classes[0] != class_label else classes[1]
        print("opposite class label:", opposite_label)
        examples_with_same_label = df.loc[df[class_col_name] == opposite_label]
    elif label_type == my_vars.SAME_LABEL_AS_RULE:
        examples_with_same_label = df.loc[df[class_col_name] == class_label]
    else:
        raise MyException("'{}' is an invalid option for the label_type!".format(label_type))
    # Only consider examples, that have the same label as the rule and aren't covered by the rule yet
    if only_uncovered_neighbors:
        covered_examples = my_vars.examples_covered_by_rule.get(rule.name, set())
        # print("examples that are already covered by the rule:", covered_examples)
        # Select only examples which aren't covered yet
        examples_with_same_label = examples_with_same_label.loc[~examples_with_same_label.index.isin(covered_examples)]
        # Check if any remaining examples are covered as well
        examples_with_same_label[my_vars.COVERED] = examples_with_same_label.loc[:, :]\
            .apply(does_rule_cover_example, axis=1, args=(rule, examples_with_same_label.dtypes))
        # Only keep the uncovered examples
        examples_with_same_label = examples_with_same_label.loc[examples_with_same_label["is_covered"] == False]
    # print("neighbors:\n{}".format(examples_with_same_label))

    if is_empty(examples_with_same_label):
        return None, None

    neighbors = examples_with_same_label.shape[0]
    # print("neighbors:", neighbors)
    if neighbors < k:
        warnings.warn("Only {} neighbors for\n{}".format(examples_with_same_label.shape[0], examples_with_same_label),
                      UserWarning)
    # if neighbors > 0:
    dists = hvdm(examples_with_same_label, rule, counts, classes, min_max, class_col_name)
    neighbor_ids = dists.index[: k]
    # print("{} nearest neighbors:\n{}\n{}".format(k, dists, neighbor_ids))
    return df.loc[neighbor_ids], dists.loc[neighbor_ids]
    # return None, None


def find_nearest_rule(rules, example, class_col_name, counts, min_max, classes, examples_covered_by_rule):
    """
    Finds the nearest rule for a given example. Certain rules are ignored, namely those for which the example is a
    seed if the rule doesn't cover multiple examples.

    Parameters
    ----------
    rules: list pd.Series - rules
    example: pd.Series - example
    class_col_name: str - name of class label
    counts: dict - lookup table for SVDM
    min_max: pd:DataFrame - contains min/max values per numeric feature
    classes: list of str - class labels in the dataset.
    examples_covered_by_rule: dict - which rule covers which examples, i.e. {rule ri: set(example ei, example ej)}

    Returns
    -------
    pd.Series, float OR None, None.
    Rule that is closest to an example, distance of that rule to the example.
    None, None if only 1 rule and 1 example are provided and that example is seed for the rule, where the latter doesn't
    cover multiple examples.

    """
    min_dist = None
    min_rule_id = None
    # hvdm() expects a dataFrame of examples, not a Series
    # Plus, data type is "object", but then numeric columns won't be detected in di(), so we need to infer them
    example_df = example.to_frame().T.infer_objects()
    # print("rules")
    # print(rules)
    try:
        for idx, rule in enumerate(rules):
            rule_id = rule.name
            # print("rule id", rule_id)
            # print("Now checking rule with ID {}:\n{}".format(rule_id, rule))
            examples = len(examples_covered_by_rule.get(rule.name, {}))
            # print("#covered examples by rule:", examples_covered_by_rule.get(rule.name, {}))
            covers_multiple_examples = True if examples > 1 else False
            # Ignore rule because current example was seed for it and the rule doesn't cover multiple examples
            if not covers_multiple_examples and rule_id == example.name:
                # Ignore rule as it's the seed for the example
                print("rule {} is seed for example {}".format(rule_id, example.name))
                continue
            neighbors, dists = find_nearest_examples(example_df, 1, rule, class_col_name, counts, min_max, classes,
                                                     label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
            # print("neighbors:", neighbors)
            if neighbors is not None:
                dist = dists.iloc[0][my_vars.DIST]
                # print("-------")
                # print("dist", dist)
                # print("-------")
                if min_dist is not None:
                    if dist < min_dist:
                        min_dist = dist
                        min_rule_id = idx
                else:
                    min_dist = dist
                    min_rule_id = idx
            else:
                raise MyException("No neighbors for rule:\n{}".format(rule))
        if min_rule_id is not None:
            return rules[min_rule_id], min_dist
        return None, None
    except MyException:
        return None, None


def most_specific_generalization(example, rule, class_col_name, dtypes):
    """
    Implements MostSpecificGeneralization() from the paper, i.e. Algorithm 2.

    Parameters
    ----------
    example: pd.Series - row from the dataset.
    rule: pd.Series - rule that will be potentially generalized.
    class_col_name: str - name of the column hold the class labels.
    dtypes: pd.Series - data types of the respective columns in the dataset.

    Returns
    -------
    pd.Series.
    Generalized rule

    """
    for (col_name, example_val), dtype in zip(example.iteritems(), dtypes):
        if col_name == class_col_name:
            continue
        example_dtype = dtype
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
    distances[my_vars.DIST] = distances.select_dtypes(float).sum(1)
    distances = distances.sort_values(my_vars.DIST, ascending=True)
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
    col_name = example_feat.name
    rule_val = rule_feat[col_name]
    dists = []
    # Feature is NaN in rule -> all distances will become 1 automatically by definition
    if pd.isnull(rule_val):
        print("column {} is NaN in rule:\n{}".format(col_name, rule_feat))
        dists = [(idx, 1.0) for idx, _ in example_feat.iteritems()]
        zlst = list(zip(*dists))
        out = pd.Series(zlst[1], index=zlst[0], name=col_name)
        return out
    n_rule = counts[col_name][rule_val]
    # For every row/example
    for idx, example_val in example_feat.iteritems():
        if pd.isnull(example_val):
            print("NaN(s) in svdm() in column '{}' in row {}".format(col_name, idx))
            dist = 1.0
        else:
            # print("compute example", idx)
            # print("------------------")
            # print(example_val)
            dist = 0.
            if example_val != rule_val:
                for k in classes:
                    # print("processing class", k)
                    n_example = counts[col_name][example_val]
                    nk_example = counts[col_name][my_vars.CONDITIONAL][example_val][k]
                    nk_rule = counts[col_name][my_vars.CONDITIONAL][rule_val][k]
                    # print("n_example", n_example)
                    # print("nk_example", nk_example)
                    # print("n_rule", n_rule)
                    # print("nk_rule", nk_rule)
                    res = abs(nk_example/n_example - nk_rule/n_rule)
                    dist += res
                    # print("|{}/{}-{}/{}| = {}".format(nk_example, n_example, nk_rule, n_rule, res))
                    # print("d={}".format(dist))
            # else:
            #     print("same val ({}) in row {}".format(example_val, idx))
        dists.append((idx, dist*dist))
    # Split tuples into 2 separate lists, one containing the indices and the other one containing the values
    zlst = list(zip(*dists))
    out = pd.Series(zlst[1], index=zlst[0], name=col_name)
    return out


def di(example_feat, rule_feat, min_max):
    """
    Computes the (squared) partial distance for numeric values between an example and a rule.

    Parameters
    ----------
    example_feat: pd.Series - column (=feature) containing all examples.
    rule_feat: pd.Series - column (=feature) of the rule.
    min_max: pd.DataFrame - min and max value per numeric feature.

    Returns
    -------
    pd.Series
    (squared) distance of each example.

    """
    col_name = example_feat.name
    lower_rule_val, upper_rule_val = rule_feat[col_name]
    dists = []
    # Feature is NaN in rule -> all distances will become 1 automatically by definition
    if pd.isnull(lower_rule_val) or pd.isnull(upper_rule_val):
        print("column {} is NaN in rule:\n{}".format(col_name, rule_feat))
        dists = [(idx, 1.0) for idx, _ in example_feat.iteritems()]
        zlst = list(zip(*dists))
        out = pd.Series(zlst[1], index=zlst[0], name=col_name)
        return out
    # For every row/example
    for idx, example_val in example_feat.iteritems():
        # print("processing", example_val)
        if pd.isnull(example_val):
            print("NaN(s) in svdm() in column '{}' in row {}".format(col_name, idx))
            dist = 1.0
        else:
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


def evaluate_f1_initialize_confusion_matrix(df, rules, class_col_name, counts, min_max, classes):
    """
    Computes the F1 score of the dataset for a given set of rules using leave-one-out cross-evaluation.
    Builds the initial confusion matrix.

    Parameters
    ----------
    df: pd.DataFrame - examples
    rules: list of pd.Series - list of rules
    class_col_name: str - name of the column in the series holding the class label
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label
    min_max: pd.DataFrame - min and max value per numeric feature.
    classes: list of str - class labels in the dataset.

    Raises
    ------
    Exception: if no nearest examples at all exist for a certain rule - this can only happen if <df> is empty, it
    contains only the seed of the rule for which nearest neighbors are computed and that rule doesn't cover any other
    examples. In short, this exception might be thrown if <= 1 example are contained in <df>.

    Returns
    -------
    float - F1 score.

    """
    # Let closest rule classify an example, but this example mustn't be the seed for the closest rule unless that
    # rule covers more examples
    # Problem: rules can't be stored in dataFrame because they might contain different features
    for row_id, example in df.iterrows():
        # print("Searching nearest rule for example:\n{}\n{}".format("------------------------------------", example))
        rule, rule_dist = find_nearest_rule(rules, example, class_col_name, counts, min_max, classes,
                                            my_vars.examples_covered_by_rule)

        # Update which rule predicts the label of the example
        # print("minimum distance ({}) by rule: {}".format(rule_dist, rule.name))
        my_vars.closest_rule_per_example[example.name] = (rule.name, rule_dist)
        my_vars.conf_matrix = update_confusion_matrix(example, rule, my_vars.positive_class, class_col_name,
                                                      my_vars.conf_matrix)
    return f1(my_vars.conf_matrix)
    # Let closest rule classify an example, but this example mustn't be the seed for the closest rule unless that
    # rule covers more examples
    # for rule in rules:
    #     print("Searching nearest examples for rule:\n{}\n{}".format("------------------------------------", rule))
    #     examples_covered_by_rule = len(my_vars.examples_covered_by_rule.get(rule.name, {}))
    #     print("#covered examples by rule:", examples_covered_by_rule)
    #     covers_multiple_examples = True if examples_covered_by_rule > 1 else False
    #     if not covers_multiple_examples:
    #         # Remove seed for rule
    #         examples = df.loc[df.index != rule.name]
    #     else:
    #         examples = df
    #     # Find the nearest example, regardless of its class label and whether the rule already covers it
    #     neighbor_example, dists = find_nearest_examples(examples, 1, rule, class_col_name, counts, min_max, classes,
    #                                                     use_same_label=False, only_uncovered_neighbors=False)
    #     if neighbor_example is not None:
    #         neighbor = neighbor_example.iloc[0]
    #         # Update which rule predicts the label of the example
    #         my_vars.example_predicted_by_rule[rule.name] = neighbor.name
    #         # my_vars.example_predicted_by_rule[neighbor.name] = rule.name
    #         update_confusion_matrix(neighbor, rule, my_vars.positive_class, class_col_name)
    #     else:
    #         raise Exception("No neighbors for rule:\n{}".format(rule))


def evaluate_f1_update_confusion_matrix(df, new_rule, class_col_name, counts, min_max, classes):
    """
    Computes the F1 score of the dataset for a given set of rules using leave-one-out cross-evaluation.
    Assumes that the initial confusion matrix already exists, hence evaluate_f1_initialize_confusion_matrix() should
    be called prior to it.

    Parameters
    ----------
    df: pd.DataFrame - examples
    new_rule: pd.Series - new rule whose effect on the F1 score should be evaluated
    class_col_name: str - name of the column in the series holding the class label
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label
    min_max: pd.DataFrame - min and max value per numeric feature.
    classes: list of str - class labels in the dataset.

    Raises
    ------
    Exception: if no nearest examples at all exist for a certain rule - this can only happen if <df> is empty, it
    contains only the seed of the rule for which nearest neighbors are computed and that rule doesn't cover any other
    examples. In short, this exception might be thrown if <= 1 example are contained in <df>.

    Returns
    -------
    float - F1 score.

    """
    print("checking new rule:")
    print(new_rule)
    # Go through all examples and check if the new rule's distance to any example is smaller than the current minimum
    # distance, i.e. if the new rule is closer to an example than any other rules
    for example_id, example in df.iterrows():
        print("Potentially update nearest rule for example:\n{}\n{}".format("------------------------------------",
                                                                            example))
        # find_nearest_examples() expects a dataFrame of examples, not a Series
        # Plus, data type is "object", but then numeric columns won't be detected in di(), so we need to infer them
        # example_df = example.to_frame().T.infer_objects()
        _, new_dist = find_nearest_rule([new_rule], example, class_col_name, counts, min_max, classes,
                                        my_vars.examples_covered_by_rule)
        if new_dist is not None:
            print("current min value", my_vars.closest_rule_per_example[example_id][1])
            print("new dist", new_dist)
            cur_min_dist = my_vars.closest_rule_per_example[example_id][1]
            if new_dist < cur_min_dist:
                print("****************************")
                print("update mapping for example", example.name)
                print("****************************")
                print("old mapping:", my_vars.closest_rule_per_example[example_id])
                my_vars.closest_rule_per_example[example_id] = (new_rule.name, new_dist)
                print("new mapping", my_vars.closest_rule_per_example[example_id])
                print("old confusion matrix:", my_vars.conf_matrix)
                my_vars.conf_matrix = update_confusion_matrix(example, new_rule, my_vars.positive_class, class_col_name,
                                                              my_vars.conf_matrix)
                print("new confusion matrix:", my_vars.conf_matrix)
    return f1(my_vars.conf_matrix)


def evaluate_f1_temporarily(df, new_rule, class_col_name, counts, min_max, classes):
    """
    Same as evaluate_f1_update_confusion_matrix(), with the difference that the actual confusion matrix isn't updated,
    but a copy of it.

    Parameters
    ----------
    df: pd.DataFrame - examples
    new_rule: pd.Series - new rule whose effect on the F1 score should be evaluated
    class_col_name: str - name of the column in the series holding the class label
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label
    min_max: pd.DataFrame - min and max value per numeric feature.
    classes: list of str - class labels in the dataset.

    Raises
    ------
    Exception: if no nearest examples at all exist for a certain rule - this can only happen if <df> is empty, it
    contains only the seed of the rule for which nearest neighbors are computed and that rule doesn't cover any other
    examples. In short, this exception might be thrown if <= 1 example are contained in <df>.

    Returns
    -------
    float, dict, dict.
    F1 score, confusion matrix, closest rule per example.

    """
    # print("checking new rule:")
    # print(new_rule)
    # initial_closest_rule_per_example = copy.deepcopy(my_vars.closest_rule_per_example)
    closest_rule_per_example = copy.deepcopy(my_vars.closest_rule_per_example)
    # initial_conf_matrix = copy.deepcopy(my_vars.conf_matrix)
    conf_matrix = copy.deepcopy(my_vars.conf_matrix)
    # Go through all examples and check if the new rule's distance to any example is smaller than the current minimum
    # distance, i.e. if the new rule is closer to an example than any other rules
    for example_id, example in df.iterrows():
        # print("Potentially update nearest rule for example:\n{}\n{}".format("------------------------------------",
        #                                                                     example))
        _, new_dist = find_nearest_rule([new_rule], example, class_col_name, counts, min_max, classes,
                                        my_vars.examples_covered_by_rule)
        if new_dist is not None:
            # print("current min value", closest_rule_per_example[example_id][1])
            # print("new dist", new_dist)
            cur_min_dist = closest_rule_per_example[example_id][1]
            if new_dist < cur_min_dist:
                print("*****************************")
                print("update mapping for example", example.name)
                print("*****************************")
                print("old mapping:", closest_rule_per_example[example_id])
                closest_rule_per_example[example_id] = (new_rule.name, new_dist)
                print("new mapping", closest_rule_per_example[example_id])
                print("old confusion matrix:", conf_matrix)
                conf_matrix = update_confusion_matrix(example, new_rule, my_vars.positive_class, class_col_name,
                                                      conf_matrix)
                print("new confusion matrix:", conf_matrix)
    return f1(conf_matrix), conf_matrix, closest_rule_per_example


def update_confusion_matrix(example, rule, positive_class, class_col_name, conf_matrix):
    """
    Updates the confusion matrix.

    Parameters
    ----------
    example: pd.Series - nearest example to a rule.
    rule: pd.Series - actual label of the rule.
    positive_class: str - name of the class label considered as true positive
    class_col_name: str - name of the column in the series holding the class label
    conf_matrix: dict - confusion matrix holding a set of example IDs for my_vars.TP/TN/FP/FN

    Returns
    -------
    dict.
    Updated confusion matrix.

    """
    # print("neighbors:\n{}".format(neighbor))
    predicted = rule[class_col_name]
    true = example[class_col_name]
    # print("example label: {} vs. rule label: {}".format(predicted, true))
    predicted_id = example.name
    # Potentially remove example from confusion matrix
    conf_matrix[my_vars.TP].discard(predicted_id)
    conf_matrix[my_vars.TN].discard(predicted_id)
    conf_matrix[my_vars.FP].discard(predicted_id)
    conf_matrix[my_vars.FN].discard(predicted_id)
    # Add updated value
    if true == positive_class:
        if predicted == true:
            conf_matrix[my_vars.TP].add(predicted_id)
            print("pred: {} <-> true: {} -> tp".format(predicted, true))
        else:
            conf_matrix[my_vars.FN].add(predicted_id)
            print("pred: {} <-> true: {} -> fn".format(predicted, true))
    else:
        if predicted == true:
            conf_matrix[my_vars.TN].add(predicted_id)
            print("pred: {} <-> true: {} -> tn".format(predicted, true))
        else:
            conf_matrix[my_vars.FP].add(predicted_id)
            print("pred: {} <-> true: {} -> fp".format(predicted, true))
    return conf_matrix


def f1(conf_matrix):
    """
    Computes the F1 score: F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    conf_matrix: dict - confusion matrix holding a set of example IDs for my_vars.TP/TN/FP/FN

    Returns
    -------
    float.
    F1-score

    """
    f1 = 0.0
    if conf_matrix is not None:
        tp = len(conf_matrix[my_vars.TP])
        fp = len(conf_matrix[my_vars.FP])
        fn = len(conf_matrix[my_vars.FN])
        # tn = len(my_vars.conf_matrix[my_vars.TN])
        precision = 0
        recall = 0
        prec_denom = tp + fp
        rec_denom = tp + fn
        if prec_denom > 0:
            precision = tp / prec_denom
        if rec_denom > 0:
            recall = tp / rec_denom
        # print("recall: {} precision: {}".format(recall, precision))
        f1_denom = precision + recall
        if f1_denom > 0:
            f1 = 2*precision*recall / f1_denom
    return f1


def add_one_best_rule(df, neighbors, rule, rules, f1,  class_col_name, counts, min_max, classes):
    """
    Implements AddOneBestRule() from the paper, i.e. Algorithm 3.

    Parameters
    ----------
    neighbors: pd.DataFrame - nearest examples for <rule>
    rule: pd.Series - rule whose effect on the F1 score should be evaluated
    rules: list of pd.Series - list of all rules in the rule set RS
    class_col_name: str - name of the column in the series holding the class label
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label
    min_max: pd.DataFrame - min and max value per numeric feature.
    classes: list of str - class labels in the dataset.

    Returns
    -------
    bool, list of pd.Series.
    True if a generalized version of the rule improves the F1 score, False otherwise.

    """
    best_f1 = f1
    best_generalization = rule
    improved = False
    best_example_id = None
    best_closest_rule_dist = None
    best_conf_matrix = None
    print("rule:\n{}".format(rule))
    print("best f1:", best_f1)
    dtypes = neighbors.dtypes
    for example_id, example in neighbors.iterrows():
        print("generalize rule for:\n{}".format(example))
        generalized_rule = most_specific_generalization(example, rule, class_col_name, dtypes)
        print("generalized rule:\n{}".format(generalized_rule))
        current_f1, current_conf_matrix, current_closest_rule = \
            evaluate_f1_temporarily(df, generalized_rule, class_col_name, counts, min_max, classes)
        print("current f1")
        print(current_f1)
        if current_f1 >= best_f1:
            print("{} >= {}".format(current_f1, f1))
            best_f1 = current_f1
            best_generalization = generalized_rule
            improved = True
            best_example_id = example_id
            best_conf_matrix = current_conf_matrix
            print("closest")
            print(current_closest_rule)
            best_closest_rule_dist = current_closest_rule
            print("closest")
            print(best_closest_rule_dist)
    if improved:
        print("improvement!")
        # Replace old rule with new one
        for idx, r in enumerate(rules):
            if rule.name == r.name:
                rules[idx] = best_generalization
                print("updated best rule per example for example {}:\n{}"
                      .format(best_example_id, (r.name, best_closest_rule_dist[best_example_id])))
                print(best_closest_rule_dist)
                # my_vars.closest_rule_per_example[best_example_id] = best_closest_rule_dist[best_example_id]
                my_vars.closest_rule_per_example = best_closest_rule_dist
                my_vars.conf_matrix = best_conf_matrix
                break
    return improved, rules


def add_all_good_rules(df, neighbors, rule, rules, f1,  class_col_name, counts, min_max, classes):
    """
    Implements AddAllGoodRules() from the paper, i.e. Algorithm 3.

    Parameters
    ----------
    neighbors: pd.DataFrame - nearest examples for <rule>
    rule: pd.Series - rule whose effect on the F1 score should be evaluated
    rules: list of pd.Series - list of all rules in the rule set RS
    class_col_name: str - name of the column in the series holding the class label
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label
    min_max: pd.DataFrame - min and max value per numeric feature.
    classes: list of str - class labels in the dataset.

    Returns
    -------
    bool, list of pd.Series.
    True if a generalized version of the rule improves the F1 score, False otherwise.

    """
    improved = False
    print("rule:\n{}".format(rule))
    print("best f1:", f1)
    dtypes = neighbors.dtypes
    best_f1 = f1
    while not is_empty(neighbors):
        for example_id, example in neighbors.iterrows():
            print("generalize rule for:\n{}".format(example))
            generalized_rule = most_specific_generalization(example, rule, class_col_name, dtypes)
            print("generalized rule:\n{}".format(generalized_rule))
            current_f1, current_conf_matrix, current_closest_rule = \
                evaluate_f1_temporarily(df, generalized_rule, class_col_name, counts, min_max, classes)
            print("neighbors before dropping:")
            print(neighbors)
            # Remove current example
            neighbors.drop(example_id, inplace=True)
            print("neighbors after dropping:")
            print(neighbors)
            print("current f1")
            print(current_f1)
            # Generalized rule is better
            if current_f1 >= best_f1:
                print("{} >= {}".format(current_f1, f1))
                best_f1 = current_f1
                improved = True
                print("improvement!")
                # Replace old rule with new one
                for idx, r in enumerate(rules):
                    if rule.name == r.name:
                        rules[idx] = generalized_rule
                        print("updated best rule per example for example {}:\n{}"
                              .format(example_id, (r.name, current_closest_rule[example_id])))
                        print(current_closest_rule)
                        my_vars.closest_rule_per_example = current_closest_rule
                        my_vars.conf_matrix = current_conf_matrix
                        break
                # Sort remaining neighbors ascendingly w.r.t. the distance to the generalized rule
                dists = []
                for neighbor_id, neighbor in neighbors.iterrows():
                    _, dist = find_nearest_rule([generalized_rule], neighbor, class_col_name, counts, min_max, classes,
                                                my_vars.examples_covered_by_rule)
                    dists.append((neighbor_id, dist))
                # At least 1 example still existed after dropping the previous one
                if len(dists) > 0:
                    dists.sort(key=itemgetter(1))
                    print("recomputed distances:", dists)
                    example_ids, dists = map(list, (zip(*dists)))
                    neighbors = neighbors.loc[example_ids]
                    # Stop current loop because neighbors' distance was recomputed
                    break
    return improved, rules


def extend_rule(df, k, rule, class_col_name, counts, min_max, classes):
    """
    Extends a rule in terms of numeric features according to algorithm 4 of the paper, i.e. extend().

    Parameters
    ----------
    df: pd.DataFrame - dataset
    k: int - number of neighbors with opposite label of <rule> to consider
    rule: pd.Series - rule
    class_col_name: str - name of class label
    counts: dict - lookup table for SVDM
    min_max: pd:DataFrame - contains min/max values per numeric feature
    classes: list of str - class labels in the dataset. It's assumed to be binary.

    Returns
    -------
    pd.Series.
    Extended rule.

    """
    neighbors, _ = find_nearest_examples(df, k, rule, class_col_name, counts, min_max, classes,
                                         label_type=my_vars.OPPOSITE_LABEL_TO_RULE, only_uncovered_neighbors=True)
    print("rule before extension:\n{}".format(rule))
    dtypes = rule.apply(type).tolist()
    print("data types", dtypes)
    for col_name, col_val in rule.iteritems():
        # Only numeric features - they're stored in a tuple
        if isinstance(col_val, tuple):
            lower_rule, upper_rule = col_val
            print("lower: {} upper: {}".format(lower_rule, upper_rule))
            remaining_lower = neighbors.loc[neighbors[col_name] < lower_rule]
            remaining_upper = neighbors.loc[neighbors[col_name] > upper_rule]
            print("neighbors meeting lower constraint:\n{}".format(remaining_lower))
            print("neighbors meeting upper constraint:\n{}".format(remaining_upper))
            new_lower = 0
            new_upper = 0
            # Extend left towards nearest neighbor
            if not is_empty(remaining_lower):
                lower_example = remaining_lower[col_name].max()
                print("lower val", lower_example)
                new_lower = 0.5 * (lower_rule - lower_example)
            # Extend right towards nearest neighbor
            if not is_empty(remaining_upper):
                upper_example = remaining_upper[col_name].min()
                print("upper val", upper_example)
                new_upper = 0.5 * (upper_example - upper_rule)
            rule[col_name] = (lower_rule - new_lower, upper_rule + new_upper)
    print("rule after extension:\n{}".format(rule))
    return rule


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

