from collections import Counter, deque, namedtuple
import os
import itertools
import warnings
import copy
from operator import itemgetter
import random
import math

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import sklearn.datasets

import scripts.vars as my_vars


class MyException(Exception):
    pass


# (ID of rule, distance of rule to the closest example, number of features in closest rule) is stored per example in a
# named tuple
Data = namedtuple("Data", ["rule_id", "dist"])
# Data = namedtuple("Data", ["rule_id", "dist", "features"])
Bounds = namedtuple("Bounds", ["lower", "upper"])

random.seed(189)


def read_dataset(src, positive_class, excluded=[], skip_rows=0, na_values=[], normalize=False, class_index=-1,
                 header=True):
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
    my_vars.minority_class = positive_class
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
    A   B ...                                                   A                           B ...
    ---------- in the dataset, the corresponding rule stores:  ----------------------------------
    1.0 x ...                                                  Bounds(lower=1.0, upper=1.0) x

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
            # Assuming the value is x, we now store named tuples of Bounds(lower=x, upper=x) per row
            # rules[col_name] = [tuple([row[col_name], row[col_name]]) for _, row in df.iterrows()]
            rules[col_name] = [Bounds(lower=row[col_name], upper=row[col_name]) for _, row in df.iterrows()]
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
    my_vars.latest_rule_id = rules_df.shape[0] - 1
    # 1 rule per example
    assert(rules_df.shape[0] == df.shape[0])
    my_vars.seed_example_rule = dict((x, {x}) for x in range(rules_df.shape[0]))
    my_vars.seed_rule_example = dict((x, x) for x in range(rules_df.shape[0]))
    # Don't store that seeds are covered by initial rules - that's given implicitly
    # my_vars.examples_covered_by_rule = dict((x, {x}) for x in range(rules_df.shape[0]))
    rules = []
    for rule_id, rule in rules_df.iterrows():
        # TODO: convert tuples into Bounds
        # converted_rule = pd.Series(name=rule_id)
        # for feat_name, val in rule.iteritems():
        #     if isinstance(val, Bounds):
        #         print("convert {} to Bounds".format(val))
        #         lower, upper = val
        #         converted_rule[feat_name] = Bounds(lower=lower, upper=upper)
        #         print(converted_rule[feat_name])
        #         print(isinstance(converted_rule[feat_name], Bounds))
        #     else:
        #         converted_rule[feat_name] = val
        # print("converted rule")
        # print(converted_rule)
        rules.append(rule)
        my_vars.all_rules[rule_id] = rule
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
            # print("pairwise distances for rule {}:".format(rule.name))
            # print("compute distance to:\n{}".format(examples_for_pairwise_distance))
            neighbors, _, _ = find_nearest_examples(examples_for_pairwise_distance, k, rule, class_col_name, counts,
                                                    min_max, classes, label_type=my_vars.ALL_LABELS,
                                                    only_uncovered_neighbors=False)
            # print("neighbors:\n{}".format(neighbors))
            labels = Counter(neighbors[class_col_name].values)
            tag = assign_tag(labels, rule[class_col_name])
            # print("=>", tag)

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
    MyException: if invalid option for <label_type> is supplied

    Returns
    -------
    pd.DataFrame, pd.DataFrame, bool OR None, None, None, None
    k-nearest examples for the given rule, distances of the k nearest examples, true if some rule became the closest
    rule for an example. Returns None, None if there are no neighbors.

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
        if is_empty(examples_with_same_label):
            return None, None, None
        # Check if any remaining examples are covered as well
        examples_with_same_label[my_vars.COVERED] = examples_with_same_label.loc[:, :]\
            .apply(does_rule_cover_example, axis=1, args=(rule, examples_with_same_label.dtypes))
        # Only keep the uncovered examples
        examples_with_same_label = examples_with_same_label.loc[examples_with_same_label["is_covered"] == False]
    # print("neighbors:\n{}".format(examples_with_same_label))

    if is_empty(examples_with_same_label):
        return None, None, None

    neighbors = examples_with_same_label.shape[0]
    # print("neighbors:", neighbors)
    if neighbors < k:
        warnings.warn("Only {} neighbors for\n{}".format(examples_with_same_label.shape[0], examples_with_same_label),
                      UserWarning)
    dists = hvdm(examples_with_same_label, rule, counts, classes, min_max, class_col_name)
    neighbor_ids = dists.index[: k]
    is_closer = _update_data_about_closest_rule(rule, dists)

    # print("{} nearest neighbors:\n{}\n{}".format(k, dists, neighbor_ids))
    return df.loc[neighbor_ids], dists.loc[neighbor_ids], is_closer


def _update_data_about_closest_rule(rule, dists):
    """
    Potentially update closest rule per example, covered examples by rule, and closest examples per rule.

    Parameters
    ----------
    rule: pd.Series - rule
    dists: pd.dataFrame - distances of neighbors

    Returns
    -------
    bool.
    True if data was updated, else False.

    Raises Exception:
    if rule doesn't contain a name/index, i.e. rule.name = None

    """
    was_updated = False
    for example_id, row in dists.iterrows():
        # print("example id:{}\ndata:{}".format(example_id, row[my_vars.DIST]))
        dist = row[my_vars.DIST]
        old_rule_id = None
        has_changed = False
        # No closest rule exists for the example yet
        if example_id not in my_vars.closest_rule_per_example:
            print("old closest rule per example:", my_vars.closest_rule_per_example)
            print("add new entry for example {}: {}".format(example_id, Data(rule_id=rule.name, dist=dist)))
            my_vars.closest_rule_per_example[example_id] = Data(rule_id=rule.name, dist=dist)
            has_changed = True
        else:
            old_rule_id, old_dist = my_vars.closest_rule_per_example[example_id]
            # if old_rule_id is None:
            #     error = "name is None in the closest rule for example {}, i.e. name=... wasn't set in the rule!"\
            #         .format(example_id)
            #     raise Exception(error)
            old_features = my_vars.all_rules[old_rule_id].size
            features = rule.size
            # 1. New rule is closer
            if dist < old_dist:
                print("new rule is closer ({}) vs. old ({})".format(dist, old_dist))
                my_vars.closest_rule_per_example[example_id] = Data(rule_id=rule.name, dist=dist)
                has_changed = True
            # 2. Occam's razor, i.e. 2 rules are equally close, then prefer the simpler (= with fewer features) one.
            # If both are equally simple, keep the current one
            elif abs(dist - old_dist) < my_vars.PRECISION and features < old_features:
                print(
                    "occam's razor: dist: {} and #old {} vs. #new features in rule {}".format(abs(dist - old_dist),
                                                                                              old_features,
                                                                                              features, rule.name))
                my_vars.closest_rule_per_example[example_id] = Data(rule_id=rule.name, dist=dist)
                has_changed = True
        if has_changed:
            was_updated = True
            # print("nearest rule was updated for example ({})".format(example_id))
            my_vars.closest_examples_per_rule.setdefault(rule.name, set()).add(example_id)
            # Delete old entry and possibly the whole entry (if the old rule isn't closest to any example anymore),
            # but only if the new closest rule is a different one (it could still be the old rule which came closer
            # to the example after generalization)
            if old_rule_id is not None and rule.name != old_rule_id:
                print(my_vars.closest_examples_per_rule)
                my_vars.closest_examples_per_rule[old_rule_id].discard(example_id)
                if len(my_vars.closest_examples_per_rule[old_rule_id]) == 0:
                    del my_vars.closest_examples_per_rule[old_rule_id]
        # Special case: rule covers example - an example could be covered by multiple rules theoretically
        if row[my_vars.DIST] == 0:
            my_vars.examples_covered_by_rule.setdefault(rule.name, set()).add(example_id)
    return was_updated


def find_nearest_rule(rules, example, class_col_name, counts, min_max, classes, examples_covered_by_rule):
    """
    Finds the nearest rule for a given example. Certain rules are ignored, namely those for which the example is a
    seed if the rule doesn't cover multiple examples. Deals with ties if multiple rules cover an example. In this case
    Occam's razor (simplest explanation is used) applies, i.e. the rule with fewest features is selected. In case of
    a tie, then the rule with the most common class label is selected. If the tie still prevails, a random rule is
    selected from the remaining set of candidate rules.

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
    pd.Series, float, bool OR None, None, None.
    Rule that is closest to an example, distance of that rule to the example, True if a rule became the closest rule.
    None, None, None if only 1 rule and 1 example are provided and that example is seed for the rule, where the latter
    doesn't cover multiple examples.

    """
    k = 5
    min_dist = math.inf
    min_rule_id = None
    if example.name in my_vars.closest_rule_per_example:
        min_rule_id, min_dist = my_vars.closest_rule_per_example[example.name]
        # print("entry exists for example {}: {}".format(example.name, my_vars.closest_rule_per_example[example.name]))
    # has_zero_dist = False
    # hvdm() expects a dataFrame of examples, not a Series
    # Plus, data type is "object", but then numeric columns won't be detected in di(), so we need to infer them
    example_df = example.to_frame().T.infer_objects()
    # print("\nexample", example.name)
    try:
        was_updated = False
        # covering_rule_ids = set()
        for rule in rules:
            rule_id = rule.name
            # print("rule id", rule_id)
            # print("Now checking rule with ID {}:\n{}".format(rule_id, rule))
            examples = len(examples_covered_by_rule.get(rule.name, set()))
            # > 0 (instead of 1) because seeds aren't stored in this dict, so we implicitly add 1
            covers_multiple_examples = True if examples > 0 else False
            # print("is rule {} seed for example {}? {}".format(rule_id, example.name, rule_id in my_vars.seed_example_rule.get(example.name, set())))
            # print("does rule {} cover multiple examples? {}".format(rule_id, covers_multiple_examples))
            if covers_multiple_examples:
                # print("covered:", examples_covered_by_rule.get(rule.name, set()))
                pass

            # Ignore rule because current example was seed for it and the rule doesn't cover multiple examples
            # if not covers_multiple_examples and my_vars.seed_example_rule[example.name] == rule_id:
            if not covers_multiple_examples and rule_id in my_vars.seed_example_rule.get(example.name, set()):
                # Ignore rule as it's the seed for the example
                # print("rule {} is seed for example {}, so ignore it".format(rule_id, example.name))
                continue
            neighbors, dists, is_closest = \
                find_nearest_examples(example_df, k, rule, class_col_name, counts, min_max, classes,
                                      label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
            # print("is closest", is_closest)
            if neighbors is not None:
                dist = dists.iloc[0][my_vars.DIST]
                # if dist == 0:
                #     covering_rule_ids.add(rule_id)
                if min_dist is not None:
                    # print("to update:", dist, min_dist, abs(dist - min_dist) < my_vars.PRECISION)
                    if is_closest:
                        # print("updated again")
                        was_updated = True
                        min_dist = dist
                        min_rule_id = rule_id
                else:
                    # print("init update")
                    min_dist = dist
                    min_rule_id = rule_id
                    was_updated = True
            else:
                raise MyException("No neighbors for rule:\n{}".format(rule))
        # print(min_dist, min_rule_id, was_updated, covering_rule_ids)
        if min_rule_id is not None:
            # print("rule with min dist ({}): {}".format(min_dist, min_rule_id))
            return my_vars.all_rules[min_rule_id], min_dist, was_updated
        return None, None, None
    except MyException:
        return None, None, None


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
    # Without a deep copy, any changes to the returned rule, will also affect <rule>, i.e. the original rule
    rule = copy.deepcopy(rule)
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
                    rule[col_name] = Bounds(lower=rule_val[0], upper=example_val)
                elif example_val < rule_val[0]:
                    # print("new lower limit", (example_val, rule_val[1]))
                    rule[col_name] = Bounds(lower=example_val, upper=rule_val[1])
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
        if col_name == class_col_name or col_name == my_vars.TAG or col_name == my_vars.COVERED:
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
        rule, rule_dist, _ = find_nearest_rule(rules, example, class_col_name, counts, min_max, classes,
                                               my_vars.examples_covered_by_rule)

        # Update which rule predicts the label of the example
        print("minimum distance ({}) to example {} by rule: {}".format(rule_dist, row_id, rule.name))
        my_vars.closest_rule_per_example[example.name] = Data(rule_id=rule.name, dist=rule_dist)
        my_vars.conf_matrix = update_confusion_matrix(example, rule, my_vars.minority_class, class_col_name,
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
    # print("checking new rule {}:".format(new_rule.name))
    print(new_rule)
    # Go through all examples and check if the new rule's distance to any example is smaller than the current minimum
    # distance, i.e. if the new rule is closer to an example than any other rules
    for example_id, example in df.iterrows():
        # print("Potentially update nearest rule for example {}:\n{}"
        #       .format(example.name, "------------------------------------"))

        _, new_dist, is_closest = find_nearest_rule([new_rule], example, class_col_name, counts, min_max, classes,
                                                    my_vars.examples_covered_by_rule)
        if new_dist is not None:
            # print("current min value", my_vars.closest_rule_per_example[example_id].dist)
            # print("new dist", new_dist)
            # print("updated?", is_closest)
            # cur_min_dist = my_vars.closest_rule_per_example[example_id][1]
            # Note that find_nearest_examples() has already updated my_vars.closest_rule_per_example in
            # _update_data_about_closest_rule(), so we only need to check for equality of floats
            if is_closest:
                # print("****************************")
                # print("update mapping for example", example.name)
                # print("****************************")
                # print("old mapping:", my_vars.closest_rule_per_example[example_id])
                # print("old examples per rule", my_vars.closest_examples_per_rule)
                old_rule_id = my_vars.closest_rule_per_example[example_id].rule_id
                my_vars.closest_examples_per_rule.setdefault(new_rule.name, set()).add(example_id)
                if old_rule_id in my_vars.closest_examples_per_rule and new_rule.name != old_rule_id:
                    my_vars.closest_examples_per_rule[old_rule_id].discard(example_id)
                    if len(my_vars.closest_examples_per_rule[old_rule_id]) == 0:
                        del my_vars.closest_examples_per_rule[old_rule_id]
                my_vars.closest_rule_per_example[example_id] = Data(rule_id=new_rule.name, dist=new_dist)
                # print("new mapping", my_vars.closest_rule_per_example[example_id])
                # print("new examples per rule", my_vars.closest_examples_per_rule)
                # print("old confusion matrix:", my_vars.conf_matrix)
                my_vars.conf_matrix = update_confusion_matrix(example, new_rule, my_vars.minority_class, class_col_name,
                                                              my_vars.conf_matrix)
                # print("new confusion matrix:", my_vars.conf_matrix)
    return f1(my_vars.conf_matrix)


def evaluate_f1_temporarily(df, new_rule, new_rule_id, class_col_name, counts, min_max, classes):
    """
    Same as evaluate_f1_update_confusion_matrix(), with the difference that the actual confusion matrix isn't updated,
    but a copy of it.

    Parameters
    ----------
    df: pd.DataFrame - examples
    new_rule: pd.Series - new rule whose effect on the F1 score should be evaluated
    new_rule_id: int - potentially new ID of <new_rule> because it might be added in add_all_good_rules() as a new rule
    or it could replace an existing rule or nothing happens. Note that if <new_rule>.name is passed as the value for
    <new_rule_id>, it has no effect on the results at all.
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
    float, dict, dict, dict, dict, list.
    F1 score, confusion matrix, closest rule per example, closest examples per rule, covered examples per rule,
    IDs of examples whose closest rule were updated

    """
    # print("\nevaluate f1 temporarily:")
    # print("+++++++++++++++++++++++++")
    # print("checking new rule", new_rule.name)
    # print(new_rule)
    # initial_closest_rule_per_example = copy.deepcopy(my_vars.closest_rule_per_example)
    closest_rule_per_example = copy.deepcopy(my_vars.closest_rule_per_example)
    closest_examples_per_rule = copy.deepcopy(my_vars.closest_examples_per_rule)
    covered_examples = copy.deepcopy(my_vars.examples_covered_by_rule)
    backup_closest_examples_per_rule = copy.deepcopy(my_vars.closest_examples_per_rule)
    backup_closest_rule = copy.deepcopy(my_vars.closest_rule_per_example)
    backup_covered = copy.deepcopy(my_vars.examples_covered_by_rule)
    conf_matrix = copy.deepcopy(my_vars.conf_matrix)
    has_changed = False
    updated_example_ids = []
    # Go through all examples and check if the new rule's distance to any example is smaller than the current minimum
    # distance, i.e. if the new rule is closer to an example than any other rules
    for example_id, example in df.iterrows():
        # print("Potentially update nearest rule for example {}:\n{}".format(example_id,
        #                                                                    "------------------------------------"))
        _, new_dist, was_updated = find_nearest_rule([new_rule], example, class_col_name, counts, min_max, classes,
                                                     my_vars.examples_covered_by_rule)
        if was_updated:
            has_changed = True

        if new_dist is not None:
            # print("current min value", closest_rule_per_example[example_id][1])
            # print("new dist", new_dist)
            # cur_min_dist = closest_rule_per_example[example_id][1]
            # if new_dist < cur_min_dist:
            if was_updated:
                print("*****************************")
                print("update mapping for example", example.name)
                print("*****************************")
                print("old mapping:", closest_rule_per_example[example_id])
                old_rule_id = closest_rule_per_example[example_id].rule_id
                # TODO: replaced
                # closest_rule_per_example[example_id] = Data(rule_id=new_rule.name, dist=new_dist)
                # TODO: problem: if this has to be replaced later on, we need to return all updated example IDs
                closest_rule_per_example[example_id] = Data(rule_id=new_rule_id, dist=new_dist)
                updated_example_ids.append(example_id)
                print("new mapping", closest_rule_per_example[example_id])
                print(closest_rule_per_example)
                print("old closest examples per rule", closest_examples_per_rule)
                # TODO: replaced
                # closest_examples_per_rule.setdefault(new_rule.name, set()).add(example_id)
                closest_examples_per_rule.setdefault(new_rule_id, set()).add(example_id)
                print("intermediate closest examples per rule", closest_examples_per_rule)
                # TODO: replaced
                if old_rule_id in closest_examples_per_rule and new_rule_id != old_rule_id:
                # if old_rule_id in closest_examples_per_rule and new_rule.name != old_rule_id:
                    print("delete")
                    closest_examples_per_rule[old_rule_id].discard(example_id)
                    if len(closest_examples_per_rule[old_rule_id]) == 0:
                        del closest_examples_per_rule[old_rule_id]
                print("new closest examples per rule", closest_examples_per_rule)
                # print("old confusion matrix:", conf_matrix)
                conf_matrix = update_confusion_matrix(example, new_rule, my_vars.minority_class, class_col_name,
                                                      conf_matrix)
                # print("new confusion matrix:", conf_matrix)
                # print("new distance", new_dist)
                if new_dist == 0:
                    print("new rule id", new_rule_id)
                    print("before covered examples by rule:", covered_examples)
                    # TODO: replaced
                    # covered_examples.setdefault(new_rule.name, set()).add(example_id)
                    covered_examples.setdefault(new_rule_id, set()).add(example_id)
                    # Delete entry for old rule only if the new rule is different from the old one
                    if old_rule_id in covered_examples and old_rule_id != new_rule_id:
                        covered_examples[old_rule_id].discard(example_id)
                        if len(covered_examples[old_rule_id]) == 0:
                            del covered_examples[old_rule_id]
                    print("after covered examples by rule:", covered_examples)

    # Reset data because it was updated in find_nearest_examples() in find_nearest_rule(), namely in
    # _update_data_about_closest_rule()
    if has_changed:
        my_vars.closest_rule_per_example = backup_closest_rule
        my_vars.closest_examples_per_rule = backup_closest_examples_per_rule
        my_vars.examples_covered_by_rule = backup_covered
    return f1(conf_matrix), conf_matrix, closest_rule_per_example, closest_examples_per_rule, covered_examples, \
        updated_example_ids


def update_confusion_matrix(example, rule, positive_class, class_col_name, conf_matrix):
    """
    Updates the confusion matrix.

    Parameters
    ----------
    example: pd.Series - nearest example to a rule
    rule: pd.Series - respective rule
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
            # print("pred: {} <-> true: {} -> tp".format(predicted, true))
        else:
            conf_matrix[my_vars.FN].add(predicted_id)
            # print("pred: {} <-> true: {} -> fn".format(predicted, true))
    else:
        if predicted == true:
            conf_matrix[my_vars.TN].add(predicted_id)
            # print("pred: {} <-> true: {} -> tn".format(predicted, true))
        else:
            conf_matrix[my_vars.FP].add(predicted_id)
            # print("pred: {} <-> true: {} -> fp".format(predicted, true))
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


def is_duplicate(new_rule, existing_rule_ids):
    """
    Checks if a rule is a duplicate of existing rules.

    Parameters
    ----------
    new_rule: pd.Series - new rule which is a potential duplicate rule
    existing_rule_ids: list of int - rule IDs that have the same hash as <new_rule> and are thus potential duplicates

    Returns
    -------
    int.
    ID of the duplicate rule or my_vars.UNIQUE_RULE otherwise.

    """
    # Check potential rules value by value if they are identical
    duplicate_rule_id = my_vars.UNIQUE_RULE
    # Since rule ID doesn't exist for the hashed rule, there might be a rule with a different ID but same values
    if new_rule.name not in existing_rule_ids:
        for rule_id in existing_rule_ids:
            possible_duplicate = my_vars.all_rules[rule_id]
            if _are_duplicates(new_rule, possible_duplicate):
                duplicate_rule_id = possible_duplicate.name
                break
    return duplicate_rule_id


def _are_duplicates(rule_i, rule_j):
    """Returns True if two rules are duplicates (= all values of the rules are identical) of each other and
    False otherwise"""
    are_identical = True
    # Same number of features in both rules
    if len(rule_i) == len(rule_j):
        for (idx_i, val_i), (idx_j, val_j) in zip(rule_i.iteritems(), rule_j.iteritems()):
            # Same feature
            if idx_i == idx_j:
                # Strings
                if isinstance(val_i, str):
                    if val_i != val_j:
                        are_identical = False
                        break
                # Tuples
                elif isinstance(val_i, Bounds):
                    lower_i, upper_i = val_i
                    lower_j, upper_j = val_j
                    if abs(lower_i - lower_j) > my_vars.PRECISION or abs(upper_i - upper_j > my_vars.PRECISION):
                        are_identical = False
                        break
                # Numbers
                else:
                    if abs(val_i - val_j) > my_vars.PRECISION:
                        are_identical = False
                        break
            else:
                are_identical = False
                break
    else:
        are_identical = False
    return are_identical


def add_one_best_rule(df, neighbors, rule, rules, f1,  class_col_name, counts, min_max, classes):
    """
    Implements AddOneBestRule() from the paper, i.e. Algorithm 3.

    Parameters
    ----------
    neighbors: pd.DataFrame - nearest examples for <rule>
    rule: pd.Series - rule whose effect on the F1 score should be evaluated
    rules: list of pd.Series - list of all rules in the rule set RS and <rule> is at the end in that list
    class_col_name: str - name of the column in the series holding the class label
    counts: dict of Counters - contains for nominal classes how often the value of an co-occurs with each class label
    min_max: pd.DataFrame - min and max value per numeric feature.
    classes: list of str - class labels in the dataset.

    Returns
    -------
    bool, list of pd.Series.
    True if a generalized version of the rule improves the F1 score, False otherwise. Returns the updated list of
    rules. All rules in that list are unique, i.e. if the best found rule becomes identical with any existing one
    (that isn't updated), it'll be ignored.

    """
    # Without deep copy, a shallow copy of <rules> is used, hence changing the returned rules would change the original
    # rules
    rules = copy.deepcopy(rules)
    best_f1 = f1
    best_generalization = rule
    improved = False
    # best_example_id = None
    best_closest_rule_dist = None
    best_conf_matrix = None
    best_closest_examples_per_rule = None
    best_covered = None
    print("rule:\n{}".format(rule))
    print("best f1:", best_f1)
    dtypes = neighbors.dtypes
    for example_id, example in neighbors.iterrows():
        print("add_1 generalize rule for example {}".format(example.name))
        generalized_rule = most_specific_generalization(example, rule, class_col_name, dtypes)
        # print("generalized rule:\n{}".format(generalized_rule))
        current_f1, current_conf_matrix, current_closest_rule, current_closest_examples_per_rule, current_covered, _\
            = evaluate_f1_temporarily(df, generalized_rule, generalized_rule.name, class_col_name, counts, min_max,
                                        classes)
        # print("current f1")
        # print(current_f1)
        if current_f1 >= best_f1:
            print("{} >= {}".format(current_f1, f1))
            best_f1 = current_f1
            best_generalization = generalized_rule
            best_closest_examples_per_rule = current_closest_examples_per_rule
            best_covered = current_covered
            improved = True
            # best_example_id = example_id
            best_conf_matrix = current_conf_matrix
            my_vars.all_rules[best_generalization.name] = best_generalization
            # print("closest")
            # print(current_closest_rule)
            best_closest_rule_dist = current_closest_rule
            # print("closest")
            # print(best_closest_rule_dist)
    if improved:
        print("improvement!")
        # Replace old rule with new one
        # for idx, r in enumerate(rules):
        # <rule> is the last rule in <rules>
        idx = -1
        # if rule.name == rules[idx].name:
        rules[idx] = best_generalization
        # print("updated best rule per example for example {}:\n{}"
        #       .format(best_example_id, (rule.name, best_closest_rule_dist[best_example_id])))
        # print(best_closest_rule_dist)
        my_vars.closest_rule_per_example = best_closest_rule_dist
        # print("closest rule per example updated", my_vars.closest_rule_per_example)
        my_vars.closest_examples_per_rule = best_closest_examples_per_rule
        # print("closest examples per rule updated", my_vars.closest_examples_per_rule)
        my_vars.examples_covered_by_rule = best_covered
        # print("covered examples by rule updated", my_vars.examples_covered_by_rule)
        my_vars.conf_matrix = best_conf_matrix
        rule_hash = compute_hashable_key(best_generalization)
        # Remove duplicate rules
        if rule_hash in my_vars.unique_rules:
            print("possible new rule")
            print(best_generalization)
            # Hash collisions might occur, so there could be multiple rules with the same hash value
            existing_rule_ids = my_vars.unique_rules[rule_hash]
            print("existing rule ids", existing_rule_ids)
            for rid in existing_rule_ids:
                print("possible duplicate:", my_vars.all_rules[rid])
            existing_rule_id = is_duplicate(best_generalization, existing_rule_ids)
            # Duplicate exists
            if existing_rule_id != my_vars.UNIQUE_RULE:
                print("Duplicate rules!!!!")
                print("rules before deletion:")
                print(rules)
                print("Duplicate rule: \n{}".format(existing_rule_id))
                print("best rule according to add_best_rule():\n{}".format(best_generalization))
                del rules[idx]
                print("rules after deletion")
                print(rules)
                # my_vars.seed_rule_example[existing_rule_id] = best_example_id
                print("delete seed rule_example entry: {}:{} "
                      .format(rule.name, my_vars.seed_rule_example[rule.name]))
                old_seed_example_id = my_vars.seed_rule_example[rule.name]
                del my_vars.seed_rule_example[rule.name]
                print("remaining entries:", my_vars.seed_rule_example)

                print("delete seed example_rule entry:", my_vars.seed_example_rule[old_seed_example_id])
                del my_vars.seed_example_rule[old_seed_example_id]
                print("remaining entries:", my_vars.seed_example_rule)

                # print("before updating examples covered by rule:", my_vars.examples_covered_by_rule)
                # print("after update:", my_vars.examples_covered_by_rule)
                print("updating which rule covers which examples:", my_vars.examples_covered_by_rule)
                covered = my_vars.examples_covered_by_rule[rule.name]
                del my_vars.examples_covered_by_rule[rule.name]
                print("currently covered by rule {}: {}".format(rule.name, covered))
                print("rule that covers it now:", existing_rule_id)
                my_vars.examples_covered_by_rule[existing_rule_id] = \
                    my_vars.examples_covered_by_rule[existing_rule_id].union(covered)

                print("updated:", my_vars.examples_covered_by_rule)
                print("closest examples per rule before update:", my_vars.closest_examples_per_rule)
                affected_examples = my_vars.closest_examples_per_rule.get(rule.name, set())
                print("affected examples that were closest to the deleted rule:", affected_examples)
                # if existing_rule_id in my_vars.closest_examples_per_rule:
                my_vars.closest_examples_per_rule[existing_rule_id] = \
                    my_vars.closest_examples_per_rule.setdefault(existing_rule_id, set())\
                    .union(affected_examples)
                # else:
                #     my_vars.closest_examples_per_rule[existing_rule_id] = affected_examples
                del my_vars.closest_examples_per_rule[rule.name]
                print("closest examples per rule after update:", my_vars.closest_examples_per_rule)
                print("closest rule before update:", my_vars.closest_rule_per_example)
                # Distance doesn't change, but the rule
                for example_id in affected_examples:
                    old_id, dist = my_vars.closest_rule_per_example[example_id]
                    my_vars.closest_rule_per_example[example_id] = Data(rule_id=existing_rule_id, dist=dist)
                print("closest rule after update:", my_vars.closest_rule_per_example)
                # break
    return improved, rules


def add_all_good_rules(df, neighbors, rule, rules, f1, class_col_name, counts, min_max, classes):
    """
    Implements AddAllGoodRules() from the paper, i.e. Algorithm 3.

    Parameters
    ----------
    df: pd.DataFrame - examples
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
    True if a generalized version of the rule improves the F1 score, False otherwise.  Returns the updated list of
    rules. All rules in that list are unique, i.e. if a new better rule becomes identical with any existing one
    (that isn't updated), it'll be ignored.

    """
    improved = False
    # print("rule: {}".format(rule.name))
    # print("best f1:", f1)
    dtypes = neighbors.dtypes
    best_f1 = f1
    iteration = 0
    while not is_empty(neighbors):
        for example_id, example in neighbors.iterrows():
            print("\nadd_all generalize rule {} for example {}".format(rule.name, example_id))
            print("-------------------------------------------")
            # Assume that the generalized rule will be added, so we generate a new ID for it in advance
            original_rule_id = rule.name
            my_vars.latest_rule_id += 1

            # print("old rule:\n{}".format(rule))
            generalized_rule = most_specific_generalization(example, rule, class_col_name, dtypes)

            # Ignore new rule if it's a duplicate
            already_exists = False
            rule_hash = compute_hashable_key(generalized_rule)
            if rule_hash in my_vars.unique_rules:
                print("possible new rule")
                print(generalized_rule)
                # Hash collisions might occur, so there could be multiple rules with the same hash value
                existing_rule_ids = my_vars.unique_rules[rule_hash]
                print("existing rule ids", existing_rule_ids)
                for rid in existing_rule_ids:
                    print("possible duplicate:", my_vars.all_rules[rid])
                existing_rule_id = is_duplicate(generalized_rule, existing_rule_ids)
                # Duplicate exists
                if existing_rule_id != my_vars.UNIQUE_RULE:
                    already_exists = True
                    print("generalized rule")
                    print(generalized_rule)
                    print("duplicate")
                    print(my_vars.unique_rules[existing_rule_id])
            if not already_exists:
                # print("generalized rule:\n{}".format(generalized_rule))
                current_f1, current_conf_matrix, current_closest_rule, current_closest_examples, current_covered, \
                    updated_example_ids = \
                    evaluate_f1_temporarily(df, generalized_rule, my_vars.latest_rule_id, class_col_name, counts,
                                            min_max, classes)
                # Remove current example
                neighbors.drop(example_id, inplace=True)

                # Generalized rule is better
                if current_f1 >= best_f1:
                    print("{} >= {}".format(current_f1, f1))
                    best_f1 = current_f1
                    improved = True
                    print("improvement!")
                    # Only update the mapping if the new rule has become the closest rule for >= 1 example, which might
                    # not be the case. For example, if an example is already covered by a different rule, and the newly
                    # generalized one would cover it too, but has the same number of features as the rule that already
                    # covers it, then there's no need to update anything in my opinion. The logic that decides if a new
                    # rule is closer is handled in find_nearest_rule() and that result is passed back as a Boolean
                    # variable
                    if iteration == 0:
                        # if was_updated:
                        print("replace rule!!!")
                        if my_vars.latest_rule_id not in current_closest_examples:
                            improved = False
                            print("new rule {} isn't closest to any example, so ignore the update".format(
                                my_vars.latest_rule_id))
                        else:
                            print("original rule ID", original_rule_id)
                            # Rule to be replaced is at the end
                            idx = -1
                            # Rule replaces existing one, so use the original one's ID and allow reuse of this new ID
                            wrong_rule_id = my_vars.latest_rule_id
                            print("wrong ID used to temporarily compute f1:", wrong_rule_id)
                            my_vars.latest_rule_id -= 1
                            # Replace old rule with new one - old rule is at the end of the list
                            rules[idx] = generalized_rule
                            print("replace rule {} which had as original id {}".format(generalized_rule.name,
                                                                                       original_rule_id))
                            my_vars.all_rules[original_rule_id] = generalized_rule
                            print("updated best rule per example for example {}:\n{}"
                                  .format(example_id, (rules[idx].name, current_closest_rule[example_id])))
                            print(current_closest_rule)
                            # We updated statistics assuming that the new rule ID would be used, but this isn't the
                            # case, so we need to rollback and merge the results of the original rule ID with those of
                            # the new rule ID
                            current_closest_rule[example_id] = Data(rule_id=original_rule_id,
                                                                    dist=current_closest_rule[example_id].dist)
                            # del current_closest_rule[wrong_rule_id]
                            my_vars.closest_rule_per_example = current_closest_rule
                            print("after update", current_closest_rule)

                            print("old closest examples per rule", current_closest_examples)
                            current_closest_examples[original_rule_id] = \
                                current_closest_examples.get(original_rule_id, set()).union(
                                    current_closest_examples.get(wrong_rule_id, set()))
                            print("intermediate closest examples per rule", current_closest_examples)
                            del current_closest_examples[wrong_rule_id]
                            my_vars.closest_examples_per_rule = current_closest_examples
                            print("new closest examples per rule", my_vars.closest_examples_per_rule)

                            print("old covered rules", current_covered)
                            current_covered[original_rule_id] = \
                                current_covered.get(original_rule_id, set()).union(current_covered.get(wrong_rule_id,
                                                                                                       set()))
                            print("intermediate covered rules", current_covered)
                            del current_covered[wrong_rule_id]
                            my_vars.examples_covered_by_rule = current_covered
                            print("new current covered rules", my_vars.examples_covered_by_rule)

                            for eid in updated_example_ids:
                                print("closest rule", my_vars.closest_rule_per_example[eid])
                                my_vars.closest_rule_per_example[eid] = \
                                    Data(rule_id=original_rule_id, dist=my_vars.closest_rule_per_example[eid].dist)
                                print("updated rule ID closest rule", my_vars.closest_rule_per_example[eid])

                            # Confusion matrix won't change, so no need to update it
                            my_vars.conf_matrix = current_conf_matrix

                            # break
                        # else:
                        #     # Delete rule
                        #     del rules[-1]
                    else:
                        # Add generalized rule instead of replacing the original one
                        print("add rule!!!")
                        print(generalized_rule)
                        print("closest new rule per example", current_closest_rule)
                        my_vars.closest_rule_per_example = current_closest_rule
                        my_vars.closest_examples_per_rule = current_closest_examples
                        my_vars.conf_matrix = current_conf_matrix
                        my_vars.examples_covered_by_rule = current_covered

                        # my_vars.latest_rule_id += 1
                        print("original rule id:", generalized_rule.name)
                        new_rule_id = my_vars.latest_rule_id
                        generalized_rule.name = new_rule_id
                        print("rule id", generalized_rule.name)

                        print("before adding unique hash:", my_vars.unique_rules)
                        new_hash = compute_hashable_key(generalized_rule)
                        is_added = True
                        if new_hash not in my_vars.unique_rules:
                            my_vars.unique_rules[new_hash] = {new_rule_id}
                        else:
                            generalized_rule.name = my_vars.latest_rule_id
                            print("hash collision when adding new rule!")
                            print("new rule:")
                            print(generalized_rule)
                            # Hash collisions might occur, so there could be multiple rules with the same hash value
                            existing_rule_ids = my_vars.unique_rules[new_hash]
                            existing_rule_id = is_duplicate(generalized_rule, existing_rule_ids)
                            # No duplicate exists
                            if existing_rule_id == -1:
                                print("hash collision, but they are different rules")
                                my_vars.unique_rules[new_hash].add(new_rule_id)
                            else:
                                print("duplicate exists!, so ignore the new rule")
                                print(my_vars.all_rules)
                                print("existing rule:")
                                print(my_vars.all_rules[existing_rule_id])
                                is_added = False
                        print("after adding unique hash:", my_vars.unique_rules)
                        # Only add if the generalized rule is no duplicate
                        if is_added:
                            print("before updating seed: example_rule:", my_vars.seed_example_rule)
                            my_vars.seed_example_rule.setdefault(example_id, set()).add(new_rule_id)
                            print("after updating seed: example_rule:", my_vars.seed_example_rule)
                            print("before updating seed: rule_example_rule:", my_vars.seed_rule_example)
                            my_vars.seed_rule_example[new_rule_id] = example_id
                            print("after updating seed: rule_example_rule:", my_vars.seed_rule_example)

                            print("new rule id:", generalized_rule.name)
                            print("added rule for example {}:\n{}"
                                  .format(example_id, (generalized_rule.name, current_closest_rule[example_id])))
                            print("covered:", my_vars.examples_covered_by_rule)
                            print("closest rule per example", my_vars.closest_rule_per_example)
                            print("closest examples per rule", my_vars.closest_examples_per_rule)
                            print("conf matrix", my_vars.conf_matrix)
                            my_vars.all_rules[generalized_rule.name] = generalized_rule
                            # Use the newly generated ID as name
                            generalized_rule.name = my_vars.latest_rule_id
                            rules.append(generalized_rule)
                        else:
                            # Rule was a duplicate, so reset the last ID
                            my_vars.latest_rule_id -= 1

                    iteration += 1

                    # Sort remaining neighbors ascendingly w.r.t. the distance to the generalized rule
                    dists = []
                    for neighbor_id, neighbor in neighbors.iterrows():
                        _, dist, _ = find_nearest_rule([generalized_rule], neighbor, class_col_name, counts, min_max,
                                                       classes, my_vars.examples_covered_by_rule)
                        dists.append((neighbor_id, dist))
                    print("recomputed distances:")
                    print(dists)
                    # At least 1 example still exists after dropping the previous one
                    if len(dists) > 0:
                        dists.sort(key=itemgetter(1))
                        example_ids, dists = map(list, (zip(*dists)))
                        neighbors = neighbors.loc[example_ids]
                    # Stop current loop because neighbors' distance was recomputed based on the generalized rule
                    break
                else:
                    # F1 score wasn't improved, so allow a reuse of this rule ID
                    my_vars.latest_rule_id -= 1
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
    neighbors, _, _ = find_nearest_examples(df, k, rule, class_col_name, counts, min_max, classes,
                                            label_type=my_vars.OPPOSITE_LABEL_TO_RULE, only_uncovered_neighbors=True)
    # print("neighbors")
    # print(neighbors)
    # print("rule before extension:\n{}".format(rule))
    # dtypes = rule.apply(type).tolist()
    # print("data types", dtypes)
    if neighbors is not None:
        for col_name, col_val in rule.iteritems():
            # Only numeric features - they're stored in a named tuple
            if isinstance(col_val, Bounds):
                lower_rule, upper_rule = col_val
                # print("lower: {} upper: {}".format(lower_rule, upper_rule))
                # print("neighbors")
                # print(neighbors)
                remaining_lower = neighbors.loc[neighbors[col_name] < lower_rule]
                remaining_upper = neighbors.loc[neighbors[col_name] > upper_rule]
                # print("neighbors meeting lower constraint:\n{}".format(remaining_lower))
                # print("neighbors meeting upper constraint:\n{}".format(remaining_upper))
                new_lower = 0
                new_upper = 0
                # Extend left towards nearest neighbor
                if not is_empty(remaining_lower):
                    lower_example = remaining_lower[col_name].max()
                    # print("lower val", lower_example)
                    new_lower = 0.5 * (lower_rule - lower_example)
                # Extend right towards nearest neighbor
                if not is_empty(remaining_upper):
                    upper_example = remaining_upper[col_name].min()
                    # print("upper val", upper_example)
                    new_upper = 0.5 * (upper_example - upper_rule)
                rule[col_name] = Bounds(lower=lower_rule - new_lower, upper=upper_rule + new_upper)
                # print("rule after extension of current column:\n{}".format(rule))
    my_vars.all_rules[rule.name] = rule
    return rule


def sklearn_to_df(sklearn_dataset):
    """Converts sklearn dataset into a pd.dataFrame."""
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['class'] = pd.Series(sklearn_dataset.target)
    return df


def delete_rule_statistics(df, rule, rules, class_col_name, counts, min_max, classes):
    """
    Deletes all statistics related to a specific rule.

    Parameters
    ----------
    df: pd.DataFrame - dataset
    rule: pd.Series - rule that was removed
    rules: list of pd.Series - list of rules
    class_col_name: str - name of class label
    counts: dict - lookup table for SVDM
    min_max: pd:DataFrame - contains min/max values per numeric feature
    classes: list of str - class labels in the dataset.

    """
    print("Rule that was deleted: \n{}".format(rule.name))
    print("delete seed rule_example entry: {}:{} "
          .format(rule.name, my_vars.seed_rule_example[rule.name]))
    old_seed_example_id = my_vars.seed_rule_example[rule.name]
    del my_vars.seed_rule_example[rule.name]
    print("remaining entries:", my_vars.seed_rule_example)

    print("delete seed example_rule entry:", my_vars.seed_example_rule[old_seed_example_id])
    del my_vars.seed_example_rule[old_seed_example_id]
    print("remaining entries:", my_vars.seed_example_rule)

    print("updating which rule covers which examples:", my_vars.examples_covered_by_rule)
    del my_vars.examples_covered_by_rule[rule.name]
    print("after update")

    affected_example_ids = my_vars.closest_examples_per_rule.get(rule.name, set())
    print("closest rule per example before update:", my_vars.closest_examples_per_rule)
    # TODO: should the distances of each example to each rule be stored in memory for fast look-up????
    # Idea: create a 2nd datastructure that stores the 2nd best rule for every example
    # If the first is removed (here), replace it by the 2nd best distance
    for example_id in affected_example_ids:
        # Delete existing entry because otherwise find_nearest_rule() won't update the distance properly as one rule,
        # the one that was just deleted, was closer
        del my_vars.closest_rule_per_example[example_id]
        example = df.loc[example_id]
        nearest, dist, is_updated = find_nearest_rule(rules, example, class_col_name, counts, min_max, classes,
                                                      my_vars.examples_covered_by_rule)
        print("new nearest rule: {} with dist {}".format(nearest.name, dist))
        print(my_vars.closest_rule_per_example)
    print("closest rule per example after update:", my_vars.closest_examples_per_rule)

    print("closest examples per rule before update:", my_vars.closest_examples_per_rule)
    if rule.name in my_vars.closest_examples_per_rule:
        del my_vars.closest_examples_per_rule[rule.name]
    print("closest examples per rule after update:", my_vars.closest_examples_per_rule)

    print("all rules before", my_vars.all_rules)
    del my_vars.all_rules[rule.name]
    print("all rules after", my_vars.all_rules)

    print("unique rules before", my_vars.unique_rules)
    rule_hash = compute_hashable_key(rule)
    rules_with_same_hash = my_vars.unique_rules[rule_hash]
    if len(rules_with_same_hash) > 1:
        my_vars.unique_rules[rule_hash].discard(rule.name)
    else:
        del my_vars.unique_rules[rule_hash]
    print("all rules after", my_vars.unique_rules)


def bracid(df, k, class_col_name, counts, min_max, classes, minority_label):
    """
    Implements the actual BRACID algorithm according to Algorithm 1 in the paper.

    Parameters
    ----------
    df: pd.DataFrame - dataset
    k: int - number of neighbors with opposite label of <rule> to consider
    class_col_name: str - name of class label
    counts: dict - lookup table for SVDM
    min_max: pd:DataFrame - contains min/max values per numeric feature
    classes: list of str - class labels in the dataset. It's assumed to be binary.
    minority_label: str - class label of the minority class. Note that all other labels are grouped into another class
    so that there's a binary classification task.

    Returns
    -------
    list of pd.Series.
    List of rules that classify the training data most accurately according to F1 score.

    """
    my_vars.minority_class = minority_label
    print("minority class label:", my_vars.minority_class)
    df, rules = add_tags_and_extract_rules(df, k, class_col_name, counts, min_max, classes)
    final_rules = []
    iteration = 0
    keep_running = True
    for rule in rules:
        rule_hash = compute_hashable_key(rule)
        print("rule/ hash:", rule_hash)
        if rule_hash not in my_vars.unique_rules:
            my_vars.unique_rules[rule_hash] = {rule.name}
        else:
            print("hashing collision ({}): {} <-> {}".format(rule_hash, my_vars.unique_rules[rule_hash], rule.name))
            my_vars.unique_rules[rule_hash].add(rule.name)
    f1 = evaluate_f1_initialize_confusion_matrix(df, rules, class_col_name, counts, min_max, classes)
    while keep_running:
        improved = False
        while len(rules) > 0:
            print("\nthere are {} rules left for evaluation:".format(len(rules)))
            print(rules)
            rule = rules.popleft()
            # for rule_idx, rule in enumerate(list(rules)):
            rule_id = rule.name
            print("rule {} is currently being processed:\n{}".format(rule_id, rule))
            # Add current rule at the end
            rules.append(rule)
            print("it was now added to the end of all rules:\n{}".format(rules))
            seed_id = my_vars.seed_rule_example[rule_id]
            print("seed id", seed_id)
            print(df)
            seed = df.loc[seed_id]
            print("seed\n{}".format(seed))
            seed_label = seed[class_col_name]
            seed_tag = seed[my_vars.TAG]
            print("seed label:", seed_label)
            print("closest rule per example", my_vars.closest_rule_per_example)
            print("closest examples per rule", my_vars.closest_examples_per_rule)
            print("covered examples", my_vars.examples_covered_by_rule)
            # Minority class label
            if seed_label == minority_label:
                neighbors, dists, _ = find_nearest_examples(df, k, rule, class_col_name, counts, min_max, classes)
                # Neighbors exist
                if neighbors is not None:
                    if seed_tag == my_vars.SAFE:
                        improved, generalized_rules = add_one_best_rule(df, neighbors, rule, rules, f1, class_col_name,
                                                                        counts, min_max, classes)
                    else:
                        improved, generalized_rules = add_all_good_rules(df, neighbors, rule, rules, f1, class_col_name,
                                                                         counts, min_max, classes)
                    if not improved:
                        # Don't extend for outlier
                        if iteration != 0:
                            extended_rule = extend_rule(df, k, rule, class_col_name, counts, min_max, classes)
                            final_rules.append(extended_rule)
                            # Delete rule
                            removed = rules.pop()
                            print("removed rule after extension:\n{}".format(removed))
                            delete_rule_statistics(df, removed, rules, class_col_name, counts, min_max, classes)
                    else:
                        # Use updated rules
                        rules = generalized_rules
                else:
                    # Delete rule
                    removed = rules.pop()
                    print("removed rule no neighbors minority:\n{}".format(removed))
                    delete_rule_statistics(df, removed, rules, class_col_name, counts, min_max, classes)
            # Majority label
            else:
                n = k
                if seed_tag == my_vars.SAFE:
                    n = 1
                neighbors, dists, _ = find_nearest_examples(df, n, rule, class_col_name, counts, min_max, classes)
                # Neighbors exist
                if neighbors is not None:
                    improved, generalized_rules = add_one_best_rule(df, neighbors, rule, rules, f1, class_col_name,
                                                                    counts, min_max, classes)
                    if not improved:
                        # Treat as noise
                        if iteration == 0:
                            # # Delete rule and corresponding seed (=noisy example)
                            example_id = my_vars.seed_rule_example[rule_id]
                            df, rules = treat_majority_example_as_noise(df, example_id, rules, rule_id)
                        else:
                            final_rules.append(rule)
                            # Delete rule
                            removed = rules.pop()
                            print("removed rule after adding majority final rule:\n{}".format(removed))
                            delete_rule_statistics(df, removed, rules, class_col_name, counts, min_max, classes)
                    else:
                        # Use updated rules
                        rules = generalized_rules
                else:
                    # Delete rule
                    removed = rules.pop()
                    print("removed rule no neighbors majority:\n{}".format(removed))
                    delete_rule_statistics(df, removed, rules, class_col_name, counts, min_max, classes)
            iteration += 1
        if not improved:
            keep_running = False
    return final_rules


def treat_majority_example_as_noise(df, example_id, rules, rule_id):
    """
    Deletes a noisy majority example and the rule for which it's a seed and updates corresponding statistics.

    Parameters
    ----------
    df: pd.dataFrame - dataset
    example_id: int - id of row in <dataset> to be removed
    rules: list of pd.Series - list of rules
    rule_id: int - id of rule for which <example_id> is the seed - it's at the last position in <rules>

    Returns
    -------
    pd.dataFrame, list of pd.Series.
    Updated dataset without the specified example, updated list of rules without the rule covering the removed example.

    """
    print("########noise!##############")
    # Delete rule and corresponding seed (=noisy example)
    # example_id = my_vars.seed_rule_example[rule_id]
    print("delete example id", example_id)
    del my_vars.seed_rule_example[rule_id]
    my_vars.seed_example_rule[example_id].discard(rule_id)
    print("before deleting the example")
    print(df)
    df.drop(df.index[example_id], inplace=True)
    print("after")
    print(df)
    # print("remaining entries for {}: {}".format(rule_id, my_vars.seed_example_rule[example_id]))
    if len(my_vars.seed_example_rule[example_id]) == 0:
        # print("deleted the empty entry!")
        del my_vars.seed_example_rule[example_id]
    # print("rules before deletion:")
    print(rules)
    # Delete rule
    removed = rules.pop()
    print("removed rule in majority noisy label:\n{}".format(removed))
    print("rules after deletion:")
    print(rules)
    return df, rules


def compute_hashable_key(series):
    """Returns a hashable (=immutable) representation of a pd.Series object"""
    cp = series.copy()
    # Ignore index for hashing because that makes hashes of rules, that are otherwise duplicates, unique
    cp.name = 1
    return hash(str(cp))
    # print("string for hashing:", tuple(cp))
    # hash_val = 0
    # for rule_id, val in cp.iteritems():
    #     # Compute hash separately for lower and upper bound
    #     if isinstance(val, Bounds):
    #         lower, upper = val
    #         hash_val += hash(lower) + hash(upper)
    #     else:
    #         hash_val += hash(val)
    # # return hash(str(cp).encode("utf-8"))
    # return hash_val


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

