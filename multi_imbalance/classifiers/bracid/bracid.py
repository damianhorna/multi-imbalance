from collections import Counter, deque, namedtuple
import warnings
import copy
from operator import itemgetter
import math

import pandas as pd
import tqdm as tqdm
from pandas.api.types import is_numeric_dtype
import sklearn.datasets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
import numpy as np
import enum
import dataclasses
from collections import defaultdict

from . import vars as my_vars

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bracid')


class ExampleClass(enum.Enum):
    SAFE = enum.auto()
    NOISY = enum.auto()
    BORDERLINE = enum.auto()


class Labels(str, enum.Enum):
    REST = "rest"


@dataclasses.dataclass
class ConfusionMatrix:
    TP: set = dataclasses.field(default_factory=set)
    TN: set = dataclasses.field(default_factory=set)
    FP: set = dataclasses.field(default_factory=set)
    FN: set = dataclasses.field(default_factory=set)

    @property
    def f1(self) -> float:
        tp = len(self.TP)
        fp = len(self.FP)
        fn = len(self.FN)
        return 2 * tp / (2 * tp + fp + fn)


# (ID of rule, distance of rule to the closest example) is stored per example in a named tuple
Data = namedtuple("Data", ["rule_id", "dist"])
Bounds = namedtuple("Bounds", ["lower", "upper"])
Support = namedtuple("Support", ["minority", "majority"])
Predictions = namedtuple("Predictions", ["label", "confidence"])
# Keep original rule and delete the corresponding duplicate rule which is at duplicate_idx in the list of rules
Duplicates = namedtuple("Duplicates",
                        ["original", "duplicate", "duplicate_idx"])


def get_min_max(df: pd.DataFrame):
    min_max = {}
    for column in df.columns:
        values = df[column].values
        min_max[column] = {
            "min": np.nanmin(values),
            "max": np.nanmax(values)
        }
    return pd.DataFrame(min_max)


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
    # logger.info(frequencies)
    most_common = frequencies[0]
    tag = ExampleClass.SAFE
    if most_common[1] == total_labels and most_common[0] != label:
        tag = ExampleClass.NOISY
    elif most_common[1] < total_labels:
        second_most_common = frequencies[1]
        # logger.info("most common: {} 2nd most common: {}".format(most_common, second_most_common))

        # Tie
        if most_common[1] == second_most_common[1] or most_common[
            0] != label:
            tag = ExampleClass.BORDERLINE
    # logger.info("neighbor labels: {} vs. {}".format(labels, label))
    # logger.info("tag:", tag)
    return tag


def _assert_is_numeric_dtype(col):
    if not is_numeric_dtype(col):
        raise ValueError(
            f'''{col.name} is not of numeric dtype. Found: {col.dtype}
{col}
''')


def normalize_dataframe(df):
    """Normalize numeric features (=columns) using min-max normalization"""
    for col_name in df.columns:
        col = df[col_name]
        _assert_is_numeric_dtype(col)
        min_val = col.min()
        max_val = col.max()
        df[col_name] = (col - min_val) / (max_val - min_val)
    return df


def normalize_series(col):
    """Normalizes a given series assuming it's data type is numeric"""
    _assert_is_numeric_dtype(col)
    min_val = col.min()
    max_val = col.max()
    col = (col - min_val) / (max_val - min_val)
    return col


def does_rule_cover_example(example, rule):
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
    for col_name, example_val in example.items():
        if col_name not in rule:
            continue
        rule_val = rule[col_name]
        if isinstance(rule_val, tuple):
            if not (rule_val[0] <= example_val <= rule_val[1]):
                return False
        elif rule_val != example_val:
            return False
    return True


def does_rule_cover_example_without_label(example, rule, dtypes,
                                          class_col_name):
    """
    Tests if a rule covers a given example. In contrast to does_rule_cover_example(), the class label is ignored.

    Parameters
    ----------
    example: pd.Series - example
    rule: pd.Series - rule
    dtypes: pd.Series - data types of the respective columns in the dataset
    class_col_name: str - name of the column in <example> that holds the class label of the example

    Returns
    -------
    bool.
    True if the rule covers the example else False.

    """
    is_covered = True
    for (col_name, example_val), dtype in zip(example.items(), dtypes):
        example_dtype = dtype
        if col_name in rule and col_name != class_col_name:
            # Cast object to tuple datatype -> this is only automatically done if it's not a string
            rule_val = (rule[col_name])
            # logger.info("rule_val", rule_val, "\nrule type:", type(rule_val))
            # assert is_numeric_dtype(example_dtype), f'{col_name} {example_dtype}'
            if not is_numeric_dtype(example_dtype):
                raise ValueError(f'{example_dtype} is not a numeric dtype')
            if rule_val[0] > example_val or rule_val[1] < example_val:
                is_covered = False
                break
    return is_covered


def is_empty(df):
    """Tests if a pd.DataFrame is empty or not"""
    return len(df.index) == 0


def most_specific_generalization(example, rule, class_col_name,
                                 dtypes):
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
    for (col_name, example_val), dtype in zip(example.items(), dtypes):
        if col_name == class_col_name:
            continue
        if col_name in rule:
            rule_val = rule[col_name]
            new_rule = rule_val
            # logger.info("rule_val", rule_val, "\nrule type:", type(rule_val))
            if example_val > rule_val[1]:
                # logger.info("new upper limit", (rule_val[0], example_val))
                new_rule = (rule_val[0], example_val)
            elif example_val < rule_val[0]:
                # logger.info("new lower limit", (example_val, rule_val[1]))
                new_rule = (example_val, rule_val[1])
                # logger.info("updated:", rule)
            if isinstance(rule_val, Bounds):
                rule[col_name] = Bounds(lower=new_rule[0],
                                        upper=new_rule[1])
            else:
                rule[col_name] = new_rule

    return rule


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
        logger.info(
            "column {} is NaN in rule:\n{}".format(col_name, rule_feat))
        dists = [(idx, 1.0) for idx, _ in example_feat.items()]
        zlst = list(zip(*dists))
        out = pd.Series(zlst[1], index=zlst[0], name=col_name)
        return out
    # For every row/example
    for idx, example_val in example_feat.items():
        # logger.info("processing", example_val)
        if pd.isnull(example_val):
            logger.warning("NaN(s) in svdm() in column '{}' in row {}".format(
                col_name, idx))
            dist = 1.0
        else:
            min_rule_val = min_max.at["min", col_name]
            max_rule_val = min_max.at["max", col_name]
            if np.isclose(min_rule_val, max_rule_val):
                logger.warning(
                    'Min and max for column "%s" are equal: %f and %f',
                    col_name, min_rule_val, max_rule_val)
                dist = 0
            # logger.info("min({})={}".format(col_name, min_rule_val))
            # logger.info("max({})={}".format(col_name, max_rule_val))
            elif example_val > upper_rule_val:
                # logger.info("example > upper")
                # logger.info("({} - {}) / ({} - {})".format(example_val, upper_rule_val, max_rule_val, min_rule_val))
                dist = (example_val - upper_rule_val) / (
                        max_rule_val - min_rule_val)
            elif example_val < lower_rule_val:
                # logger.info("example < lower")
                # logger.info("({} - {}) / ({} - {})".format(lower_rule_val, example_val, max_rule_val, min_rule_val))
                dist = (lower_rule_val - example_val) / (
                        max_rule_val - min_rule_val)
            else:
                dist = 0
        dists.append((idx, dist * dist))
    zlst = list(zip(*dists))
    out = pd.Series(zlst[1], index=zlst[0], name=col_name)
    return out


def hvdm(examples, rule, classes, min_max, class_col_name):
    """
    Computes the distance (Heterogenous Value Difference Metrics) between a rule/example and another example.
    Assumes that there's at least 1 feature shared between <rule> and <examples>.

    Parameters
    ----------
    examples: pd.DataFrame - examples
    rule: pd.Series - (m x n) rule
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
        if col_name in [class_col_name, my_vars.TAG, my_vars.COVERED]:
            continue
        # Extract column from both dataframes into numpy array
        example_feature_col = examples[col_name]
        # Compute numeric distance
        _assert_is_numeric_dtype(example_feature_col)
        dist_squared = di(example_feature_col, rule, min_max)
        dists.append(dist_squared)
    # Note: this line assumes that there's at least 1 feature
    distances = pd.DataFrame(list(zip(*dists)),
                             columns=[s.name for s in dists],
                             index=dists[0].index)
    # Sum up rows to compute HVDM - no need to square the distances as the order won't change
    distances[my_vars.DIST] = distances.select_dtypes(float).sum(1)
    distances = distances.sort_values(my_vars.DIST, ascending=True)
    return distances


def update_confusion_matrix(example, rule, positive_class,
                            class_col_name, conf_matrix):
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
    # logger.info("neighbors:\n{}".format(neighbor))
    predicted = rule[class_col_name]
    true = example[class_col_name]
    # logger.info("example label: {} vs. rule label: {}".format(predicted, true))
    predicted_id = example.name
    # Potentially remove example from confusion matrix
    conf_matrix.TP.discard(predicted_id)
    conf_matrix.TN.discard(predicted_id)
    conf_matrix.FP.discard(predicted_id)
    conf_matrix.FN.discard(predicted_id)
    # Add updated value
    if true == positive_class:
        if predicted == true:
            conf_matrix.TP.add(predicted_id)
            # logger.info("pred: {} <-> true: {} -> tp".format(predicted, true))
        else:
            conf_matrix.FN.add(predicted_id)
            # logger.info("pred: {} <-> true: {} -> fn".format(predicted, true))
    else:
        if predicted == true:
            conf_matrix.TN.add(predicted_id)
            # logger.info("pred: {} <-> true: {} -> tn".format(predicted, true))
        else:
            conf_matrix.FP.add(predicted_id)
            # logger.info("pred: {} <-> true: {} -> fp".format(predicted, true))
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
        tp = len(conf_matrix.TP)
        fp = len(conf_matrix.FP)
        fn = len(conf_matrix.FN)
        # tn = len(self.conf_matrix.TN)
        precision = 0
        recall = 0
        prec_denom = tp + fp
        rec_denom = tp + fn
        if prec_denom > 0:
            precision = tp / prec_denom
        if rec_denom > 0:
            recall = tp / rec_denom
        # logger.info("recall: {} precision: {}".format(recall, precision))
        f1_denom = precision + recall
        if f1_denom > 0:
            f1 = 2 * precision * recall / f1_denom
    return f1


def _are_duplicates(rule_i, rule_j):
    """Returns True if two rules are duplicates (= all values of the rules are identical) of each other and
    False otherwise"""
    # Same number of features in both rules
    if len(rule_i) != len(rule_j):
        return False

    def _compare(val_i, val_j):
        # Tuples/Bounds
        if isinstance(val_i, tuple):
            assert issubclass(Bounds, tuple)
            lower_i, upper_i = val_i
            lower_j, upper_j = val_j
            return np.isclose(lower_i, lower_j, atol=my_vars.PRECISION) \
                   and np.isclose(upper_i, upper_j, atol=my_vars.PRECISION)

        return val_i == val_j

    for (idx_i, val_i), (idx_j, val_j) in zip(rule_i.items(),
                                              rule_j.items()):
        # Same feature
        if idx_i != idx_j:
            return False
        if not _compare(val_i, val_j):
            return False
    return True


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
        # Assuming the value is x, we now store named tuples of Bounds(lower=x, upper=x) per row
        rules[col_name] = [Bounds(lower=row[col_name], upper=row[col_name])
                           for _, row in df.iterrows()]
    return rules


def sklearn_to_df(sklearn_dataset):
    """Converts sklearn dataset into a pd.dataFrame."""
    df = pd.DataFrame(sklearn_dataset.data,
                      columns=sklearn_dataset.feature_names)
    df['class'] = pd.Series(sklearn_dataset.target)
    return df


# def compute_hashable_key(series):
#     """Returns a hashable (=immutable) representation of a pd.Series object"""
#     # Ignore index for hashing because that makes hashes of rules, that are otherwise duplicates, unique
#     if my_vars.HASH in series:
#         return series[my_vars.HASH]
#     original_name = series.name
#     series.name = 1
#     hash_value = hash(str(series))
#     series.name = original_name
#     series[my_vars.HASH] = hash_value
#     return hash_value

def compute_hashable_key(series):
    """Returns a hashable (=immutable) representation of a pd.Series object"""
    # Ignore index for hashing because that makes hashes of rules, that are otherwise duplicates, unique
    original_name = series.name
    series.name = 1
    hash_value = hash(str(series))
    series.name = original_name
    return hash_value


def to_binary_classification_task(df, class_col_name, minority_label,
                                  merged_label="rest"):
    """
    Converts a classification task into a binary classification task merging all classes other than the minority one.

    Parameters
    ----------
    df: pd.DataFrame - dataset
    minority_label: str - name of the minority class. All other labels are merged into another label
    class_col_name: str - name of the column that holds the class labels
    merged_label: str - name of the class label into which the remaining classes will be merged

    Returns
    -------
    pd.DataFrame.
    Classification task with binary labels - the minority label and "rest".

    """
    unique_class_labels = set(df[class_col_name].tolist())
    unique_class_labels.remove(minority_label)
    labels_to_merge = dict(
        (label, merged_label) for label in unique_class_labels)
    df[class_col_name] = df[class_col_name].replace(labels_to_merge)
    return df


class BRACID(BaseEstimator, ClassifierMixin):

    def __init__(self, k: int = 5, minority_class=None):
        """
        Parameters
        ----------
        k: int - number of neighbors to consider
        minority_class: int, str or None - minority class in binary classification task

        Returns
        -------
        pd.DataFrame, list of pd.Series.
        Dataset with an additional column containing the tag, initially extracted rules.
        """
        # {example ei: set(rule ri for which ei is the seed)}
        self.seed_example_rule = {}
        # {rule ri: example ei is seed for ri}
        self.seed_rule_example = {}
        # {rule ri: set(example ei)}
        self.examples_covered_by_rule = {}
        # {example ei: tuple(rule ri, distance di)}
        self.closest_rule_per_example = {}
        # {rule ri: set(example ei, example ej)}
        self._closest_examples_per_rule = defaultdict(set)
        self.conf_matrix = ConfusionMatrix()
        # {hash of rule ri: set(ID of rule ri, ID of rule rj)}
        self.unique_rules = {}
        # {ID of rule ri: rule ri (=pd.Series)}
        self.all_rules = {}
        self.latest_rule_id = 0
        if (minority_class is not None
                and not isinstance(minority_class, (str, int))):
            raise ValueError(
                f'minority_class={minority_class} should be an int or str '
                f'but is {type(minority_class)}')
        self._minority_class = minority_class
        self.k = k
        self._class_column_name = "Class"
        self._rules = None

    def read_dataset(self, src, positive_class, excluded=[], skip_rows=0,
                     na_values=[], normalize=False, class_index=-1,
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
        pd.DataFrame, dict, pd.DataFrame, pd.DataFrame - dataset, SVDM lookup matrix which contains for nominal
        classes how often the value of a feature co-occurs with each class label, initial rule set, min/max values per
        numeric column

        """
        self._minority_class = positive_class
        # Add column names
        if header:
            df = pd.read_csv(src, skiprows=skip_rows, na_values=na_values)
        else:
            df = pd.read_csv(src, skiprows=skip_rows, na_values=na_values,
                             header=None)
            df.columns = [i for i in range(len(df.columns))]
        # Convert fancy index to regular index - otherwise the loop below won't skip the column with class labels
        if class_index < 0:
            class_index = len(df.columns) + class_index
        self.CLASSES = df.iloc[:, class_index].unique()
        class_col_name = df.columns[class_index]
        rules = extract_initial_rules(df, class_col_name)
        minmax = {}
        # Create lookup matrix for nominal features for SVDM + normalize numerical features columnwise, but ignore labels
        for col_name in df:
            if col_name == class_col_name:
                continue
            col = df[col_name]
            assert is_numeric_dtype(col), f'{col_name} {col.dtype} {col}'
            if normalize:
                df[col_name] = normalize_series(col)
            minmax[col_name] = {"min": col.min(), "max": col.max()}
        min_max = pd.DataFrame(minmax)
        return df, rules, min_max

    def add_tags_and_extract_rules(self, df, k, class_col_name, min_max,
                                   classes):
        """
        Extracts initial rules and assigns each example in the dataset a tag, either "SAFE" or "UNSAFE", "NOISY",
        "BORDERLINE".

        Parameters
        ----------
        df: pd.DataFrame - dataset
        k: int - number of neighbors to consider
        class_col_name: str - name of class label
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset.

        Returns
        -------
        pd.DataFrame, list of pd.Series.
        Dataset with an additional column containing the tag, initially extracted rules.

        """
        rules_df = extract_initial_rules(df, class_col_name)
        # self.latest_rule_id = rules_df.shape[0] - 1
        # 1 rule per example
        # assert(rules_df.shape[0] == df.shape[0])
        # The next 2 lines assume that the 1st example starts with ID 0 which isn't necessarily true
        # self.seed_example_rule = dict((x, {x}) for x in range(rules_df.shape[0]))
        # self.seed_rule_example = dict((x, x) for x in range(rules_df.shape[0]))
        for rule_id, _ in rules_df.iterrows():
            self.seed_rule_example[rule_id] = rule_id
            self.seed_example_rule[rule_id] = {rule_id}

        # Don't store that seeds are covered by initial rules - that's given implicitly
        # self.examples_covered_by_rule = dict((x, {x}) for x in range(rules_df.shape[0]))
        rules = []
        for rule_id, rule in rules_df.iterrows():
            # TODO: convert tuples into Bounds
            # converted_rule = pd.Series(name=rule_id)
            # for feat_name, val in rule.items():
            #     if isinstance(val, Bounds):
            #         logger.info("convert {} to Bounds".format(val))
            #         lower, upper = val
            #         converted_rule[feat_name] = Bounds(lower=lower, upper=upper)
            #         logger.info(converted_rule[feat_name])
            #         logger.info(isinstance(converted_rule[feat_name], Bounds))
            #     else:
            #         converted_rule[feat_name] = val
            # logger.info("converted rule")
            # logger.info(converted_rule)
            rules.append(rule)
            self.all_rules[rule_id] = rule
        tagged = self.add_tags(df, k, rules, class_col_name, min_max, classes)
        rules = deque(rules)
        return tagged, rules

    def add_tags(self, df, k, rules, class_col_name, min_max, classes):
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
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset.

        Returns
        -------
        pd.DataFrame.
        Dataset with an additional column containing the tag.

        """
        tags = []
        for rule in rules:
            logger.info(rule)
            rule_id = rule.name
            # Ignore current row
            examples_for_pairwise_distance = df.loc[df.index != rule_id]
            if examples_for_pairwise_distance.shape[0] > 0:
                # logger.info("pairwise distances for rule {}:".format(rule.name))
                # logger.info("compute distance to:\n{}".format(examples_for_pairwise_distance))
                neighbors, _, _ = self.find_nearest_examples(
                    examples_for_pairwise_distance, k, rule, class_col_name,
                    min_max, classes, label_type=my_vars.ALL_LABELS,
                    only_uncovered_neighbors=False)
                # logger.info("neighbors:\n{}".format(neighbors))
                labels = Counter(neighbors[class_col_name].values)
                tag = assign_tag(labels, rule[class_col_name])
                # logger.info("=>", tag)

                tags.append(tag)
        df[my_vars.TAG] = pd.Series(tags)
        return df

    def find_nearest_examples(self, df, k, rule, class_col_name, min_max,
                              classes, label_type=my_vars.ALL_LABELS,
                              only_uncovered_neighbors=True,
                              only_single_example=False):
        """
        Finds k-nearest examples for a given rule with the same class label as the rule.
        If less than k examples exist, a warning is issued.

        Parameters
        ----------
        df: pd.DataFrame - dataset
        k: int - number of neighbors to consider
        rule: pd.Series - rule
        class_col_name: str - name of class label
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
        ValueError: if invalid option for <label_type> is supplied

        Returns
        -------
        pd.DataFrame, pd.DataFrame, bool OR None, None, None, None
        k-nearest examples for the given rule, distances of the k nearest examples, true if some rule became the closest
        rule for an example. Returns None, None if there are no neighbors.

        """
        # logger.info("find neighbors with same label as rule ({}) and which aren't covered by the rule yet ({})"
        #       .format(use_same_label, only_uncovered_neighbors))
        class_label = rule[class_col_name]
        if label_type == my_vars.ALL_LABELS:
            examples_with_same_label = df.copy()
        elif label_type == my_vars.OPPOSITE_LABEL_TO_RULE:
            opposite_label = classes[0] if classes[0] != class_label else \
                classes[1]
            logger.info("opposite class label:", opposite_label)
            examples_with_same_label = df.loc[
                df[class_col_name] == opposite_label]
        elif label_type == my_vars.SAME_LABEL_AS_RULE:
            examples_with_same_label = df.loc[
                df[class_col_name] == class_label]
        else:
            raise ValueError(
                "'{}' is an invalid option for the label_type!".format(
                    label_type))
        # Only consider examples, that have the same label as the rule and aren't covered by the rule yet
        if only_uncovered_neighbors:
            covered_examples = self.examples_covered_by_rule.get(rule.name,
                                                                 set())
            # logger.info("examples that are already covered by the rule:", covered_examples)
            # Select only examples which aren't covered yet
            examples_with_same_label = examples_with_same_label.loc[
                ~examples_with_same_label.index.isin(covered_examples)]
            if is_empty(examples_with_same_label):
                return None, None, None
            # Check if any remaining examples are covered as well
            examples_with_same_label[
                my_vars.COVERED] = examples_with_same_label.loc[:, :] \
                .apply(does_rule_cover_example, axis=1,
                       args=(rule,))
            # Only keep the uncovered examples
            examples_with_same_label = examples_with_same_label.loc[
                examples_with_same_label[my_vars.COVERED] == False]
        # logger.info("neighbors:\n{}".format(examples_with_same_label))

        if is_empty(examples_with_same_label):
            return None, None, None

        neighbors = examples_with_same_label.shape[0]
        # logger.info("neighbors:", neighbors)
        if neighbors < k and not only_single_example:
            logging.debug("Only %i neighbors for\n%s",
                          examples_with_same_label.shape[0],
                          examples_with_same_label)
        dists = hvdm(examples_with_same_label, rule, classes, min_max,
                     class_col_name)
        neighbor_ids = dists.index[: k]
        is_closer = self._update_data_about_closest_rule(rule, dists)

        # logger.info("{} nearest neighbors:\n{}\n{}".format(k, dists, neighbor_ids))
        return df.loc[neighbor_ids], dists.loc[neighbor_ids], is_closer

    def _update_data_about_closest_rule(self, rule, dists):
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
            # logger.info("example id:{}\ndata:{}".format(example_id, row[my_vars.DIST]))
            dist = row[my_vars.DIST]
            old_rule_id = None
            has_changed = False
            # No closest rule exists for the example yet
            if example_id not in self.closest_rule_per_example:
                logger.info("old closest rule per example:",
                            self.closest_rule_per_example)
                logger.info(
                    "add new entry for example {}: {}".format(example_id, Data(
                        rule_id=rule.name, dist=dist)))
                self.closest_rule_per_example[example_id] = Data(
                    rule_id=rule.name, dist=dist)
                has_changed = True
            else:
                old_rule_id, old_dist = self.closest_rule_per_example[
                    example_id]
                # if old_rule_id is None:
                #     error = "name is None in the closest rule for example {}, i.e. name=... wasn't set in the rule!"\
                #         .format(example_id)
                #     raise Exception(error)
                # logger.info("old existing closest rule per example", self.closest_rule_per_example)
                # logger.info("get rule {} for example {}".format(old_rule_id, example_id))
                old_features = self.all_rules[old_rule_id].size
                features = rule.size
                # 1. New rule is closer
                if dist < old_dist:
                    logger.info(
                        "new rule is closer ({}) vs. old ({})".format(dist,
                                                                      old_dist))
                    self.closest_rule_per_example[example_id] = Data(
                        rule_id=rule.name, dist=dist)
                    has_changed = True
                # 2. Occam's razor, i.e. 2 rules are equally close, then prefer the simpler (= with fewer features) one.
                # If both are equally simple, keep the current one
                elif abs(
                        dist - old_dist) < my_vars.PRECISION and features < old_features:
                    logger.info(
                        "occam's razor: dist: {} and #old {} vs. #new {} features in rule {}".format(
                            abs(dist - old_dist),
                            old_features,
                            features, rule.name))
                    self.closest_rule_per_example[example_id] = Data(
                        rule_id=rule.name, dist=dist)
                    has_changed = True
            if has_changed:
                was_updated = True
                # logger.info("nearest rule was updated for example ({})".format(example_id))
                self.closest_examples_per_rule.setdefault(rule.name,
                                                          set()).add(
                    example_id)
                # Delete old entry and possibly the whole entry (if the old rule isn't closest to any example anymore),
                # but only if the new closest rule is a different one (it could still be the old rule which came closer
                # to the example after generalization)
                if old_rule_id is not None and rule.name != old_rule_id:
                    logger.info("update closest examples per rule")
                    logger.info("old", self.closest_examples_per_rule)
                    self.closest_examples_per_rule[old_rule_id].discard(
                        example_id)
                    if len(self.closest_examples_per_rule[old_rule_id]) == 0:
                        del self.closest_examples_per_rule[old_rule_id]
                    logger.info("new", self.closest_examples_per_rule)
            # Special case: rule covers example - an example could be covered by multiple rules theoretically
            if row[my_vars.DIST] == 0:
                self.examples_covered_by_rule.setdefault(rule.name, set()).add(
                    example_id)
        return was_updated

    def find_nearest_rule(self, rules, example, class_col_name, min_max,
                          classes, examples_covered_by_rule, label_type,
                          only_uncovered_neighbors):
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
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset.
        examples_covered_by_rule: dict - which rule covers which examples, i.e. {rule ri: set(example ei, example ej)}
        label_type: str - consider only examples of the specified type as neighbors. Valid values:
        scripts.vars.ALL_LABELS - ignore the label and choose the k-nearest examples across all class labels
        scripts.vars.SAME_LABEL_AS_RULE - consider only examples as k-nearest examples with they have the same label as
        <rule>
        scripts.vars.OPPOSITE_LABEL_TO_RULE - consider only examples as k-nearest examples with they have the opposite
        label of <rule>
        only_uncovered_neighbors: bool - True if only examples should be considered that aren't covered by <rule> yet.
        Otherwise, all neighbors are considered. An example is covered by a rule if the example satisfies all conditions
        imposed by <rule>.

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
        if example.name in self.closest_rule_per_example:
            min_rule_id, min_dist = self.closest_rule_per_example[example.name]
            # logger.info("entry exists for example {}: {}".format(example.name, self.closest_rule_per_example[example.name]))
        # hvdm() expects a dataFrame of examples, not a Series
        # Plus, data type is "object", but then numeric columns won't be detected in di(), so we need to infer them
        example_df = example.to_frame().T.infer_objects()
        was_updated = False
        for rule in rules:
            rule_id = rule.name
            # logger.info("Now checking rule with ID {}:\n{}".format(rule_id, rule))
            examples = len(examples_covered_by_rule.get(rule.name, set()))
            # > 0 (instead of 1) because seeds aren't stored in this dict, so we implicitly add 1
            covers_multiple_examples = True if examples > 0 else False

            # Ignore rule because current example was seed for it and the rule doesn't cover multiple examples
            # if not covers_multiple_examples and self.seed_example_rule[example.name] == rule_id:
            if not covers_multiple_examples and rule_id in self.seed_example_rule.get(
                    example.name, set()):
                # Ignore rule as it's the seed for the example
                # logger.info("rule {} is seed for example {}, so ignore it".format(rule_id, example.name))
                continue
            assert example_df.shape[0] == 1
            neighbors, dists, is_closest = \
                self.find_nearest_examples(example_df, k, rule, class_col_name,
                                           min_max, classes,
                                           label_type=label_type,
                                           only_uncovered_neighbors=only_uncovered_neighbors,
                                           only_single_example=True)
            if neighbors is None:
                logger.debug("No neighbors for rule:\n{}".format(rule))
                min_rule_id = None
                break
            dist = dists.iloc[0][my_vars.DIST]
            if min_dist is not None:
                if is_closest:
                    was_updated = True
                    min_dist = dist
                    min_rule_id = rule_id
            else:
                min_dist = dist
                min_rule_id = rule_id
                was_updated = True

        if min_rule_id is not None:
            logger.info(
                "nearest rule for example {}:rule {} with dist={}".format(
                    example.name, min_rule_id, min_dist))
            return self.all_rules[min_rule_id], min_dist, was_updated
        return None, None, None

    def evaluate_f1_initialize_confusion_matrix(self, df, rules,
                                                class_col_name, min_max,
                                                classes):
        """
        Computes the F1 score of the dataset for a given set of rules using leave-one-out cross-evaluation.
        Builds the initial confusion matrix.

        Parameters
        ----------
        df: pd.DataFrame - examples
        rules: list of pd.Series - list of rules
        class_col_name: str - name of the column in the series holding the class label
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
            # logger.info("Searching nearest rule for example:\n{}\n{}".format("------------------------------------", example))
            rule, rule_dist, _ = self.find_nearest_rule(rules, example,
                                                        class_col_name,
                                                        min_max, classes,
                                                        self.examples_covered_by_rule,
                                                        label_type=my_vars.ALL_LABELS,
                                                        only_uncovered_neighbors=False)

            # Update which rule predicts the label of the example
            logger.info(
                "minimum distance ({}) to example {} by rule: {}".format(
                    rule_dist, row_id, rule.name))
            self.closest_rule_per_example[example.name] = Data(
                rule_id=rule.name, dist=rule_dist)
            self.conf_matrix = update_confusion_matrix(example, rule,
                                                       self._minority_class,
                                                       class_col_name,
                                                       self.conf_matrix)
        return f1(self.conf_matrix)

    def evaluate_f1_update_confusion_matrix(self, df, new_rule, class_col_name,
                                            min_max, classes):
        """
        Computes the F1 score of the dataset for a given set of rules using leave-one-out cross-evaluation.
        Assumes that the initial confusion matrix already exists, hence evaluate_f1_initialize_confusion_matrix() should
        be called prior to it.

        Parameters
        ----------
        df: pd.DataFrame - examples
        new_rule: pd.Series - new rule whose effect on the F1 score should be evaluated
        class_col_name: str - name of the column in the series holding the class label
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
        # logger.info("checking new rule {}:".format(new_rule.name))
        logger.info(new_rule)
        # Go through all examples and check if the new rule's distance to any example is smaller than the current minimum
        # distance, i.e. if the new rule is closer to an example than any other rules
        for example_id, example in df.iterrows():
            # logger.info("Potentially update nearest rule for example {}:\n{}"
            #       .format(example.name, "------------------------------------"))

            _, new_dist, is_closest = self.find_nearest_rule([new_rule],
                                                             example,
                                                             class_col_name,
                                                             min_max, classes,
                                                             self.examples_covered_by_rule,
                                                             label_type=my_vars.ALL_LABELS,
                                                             only_uncovered_neighbors=False)
            if new_dist is not None:
                # logger.info("current min value", self.closest_rule_per_example[example_id].dist)
                # logger.info("new dist", new_dist)
                # logger.info("updated?", is_closest)
                # cur_min_dist = self.closest_rule_per_example[example_id][1]
                # Note that find_nearest_examples() has already updated self.closest_rule_per_example in
                # _update_data_about_closest_rule(), so we only need to check for equality of floats
                if is_closest:
                    # logger.info("****************************")
                    # logger.info("update mapping for example", example.name)
                    # logger.info("****************************")
                    # logger.info("old mapping:", self.closest_rule_per_example[example_id])
                    # logger.info("old examples per rule", self.closest_examples_per_rule)
                    old_rule_id = self.closest_rule_per_example[
                        example_id].rule_id
                    self.closest_examples_per_rule.setdefault(new_rule.name,
                                                              set()).add(
                        example_id)
                    if old_rule_id in self.closest_examples_per_rule and new_rule.name != old_rule_id:
                        self.closest_examples_per_rule[old_rule_id].discard(
                            example_id)
                        if len(self.closest_examples_per_rule[
                                   old_rule_id]) == 0:
                            del self.closest_examples_per_rule[old_rule_id]
                    self.closest_rule_per_example[example_id] = Data(
                        rule_id=new_rule.name, dist=new_dist)
                    # logger.info("new mapping", self.closest_rule_per_example[example_id])
                    # logger.info("new examples per rule", self.closest_examples_per_rule)
                    # logger.info("old confusion matrix:", self.conf_matrix)
                    self.conf_matrix = update_confusion_matrix(example,
                                                               new_rule,
                                                               self._minority_class,
                                                               class_col_name,
                                                               self.conf_matrix)
                    # logger.info("new confusion matrix:", self.conf_matrix)
        return f1(self.conf_matrix)

    def evaluate_f1_temporarily(self, df, new_rule, new_rule_id,
                                class_col_name, min_max, classes):
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
        # logger.info("\nevaluate f1 temporarily:")
        # logger.info("+++++++++++++++++++++++++")
        # logger.info("checking new rule", new_rule.name)
        # logger.info(new_rule)
        # initial_closest_rule_per_example = copy.deepcopy(self.closest_rule_per_example)
        closest_rule_per_example = copy.deepcopy(self.closest_rule_per_example)
        closest_examples_per_rule = copy.deepcopy(
            self.closest_examples_per_rule)
        covered_examples = copy.deepcopy(self.examples_covered_by_rule)
        backup_closest_examples_per_rule = copy.deepcopy(
            self.closest_examples_per_rule)
        backup_closest_rule = copy.deepcopy(self.closest_rule_per_example)
        backup_covered = copy.deepcopy(self.examples_covered_by_rule)
        conf_matrix = copy.deepcopy(self.conf_matrix)
        has_changed = False
        updated_example_ids = []
        # Go through all examples and check if the new rule's distance to any example is smaller than the current minimum
        # distance, i.e. if the new rule is closer to an example than any other rules
        for example_id, example in df.iterrows():
            # logger.info("Potentially update nearest rule for example {}:\n{}".format(example_id,
            #                                                                    "------------------------------------"))
            _, new_dist, was_updated = self.find_nearest_rule([new_rule],
                                                              example,
                                                              class_col_name,
                                                              min_max, classes,
                                                              self.examples_covered_by_rule,
                                                              label_type=my_vars.ALL_LABELS,
                                                              only_uncovered_neighbors=False)
            if was_updated:
                has_changed = True

            if new_dist is not None:
                # logger.info("current min value", closest_rule_per_example[example_id][1])
                # logger.info("new dist", new_dist)
                # cur_min_dist = closest_rule_per_example[example_id][1]
                # if new_dist < cur_min_dist:
                if was_updated:
                    logger.info("*****************************")
                    logger.info("update mapping for example", example.name)
                    logger.info("*****************************")
                    # logger.info("old mapping:", closest_rule_per_example[example_id])
                    old_rule_id = closest_rule_per_example[example_id].rule_id
                    closest_rule_per_example[example_id] = Data(
                        rule_id=new_rule_id, dist=new_dist)
                    updated_example_ids.append(example_id)
                    # logger.info("new mapping", closest_rule_per_example[example_id])
                    # logger.info(closest_rule_per_example)
                    # logger.info("old closest examples per rule", closest_examples_per_rule)
                    closest_examples_per_rule.setdefault(new_rule_id,
                                                         set()).add(example_id)
                    # logger.info("intermediate closest examples per rule", closest_examples_per_rule)
                    if old_rule_id in closest_examples_per_rule and new_rule_id != old_rule_id:
                        # logger.info("delete")
                        closest_examples_per_rule[old_rule_id].discard(
                            example_id)
                        if len(closest_examples_per_rule[old_rule_id]) == 0:
                            del closest_examples_per_rule[old_rule_id]
                    # logger.info("new closest examples per rule", closest_examples_per_rule)
                    # logger.info("old confusion matrix:", conf_matrix)
                    conf_matrix = update_confusion_matrix(example,
                                                          new_rule,
                                                          self._minority_class,
                                                          class_col_name,
                                                          conf_matrix)
                    # logger.info("new confusion matrix:", conf_matrix)
                    # logger.info("new distance", new_dist)
                    if new_dist == 0:
                        # logger.info("new rule id", new_rule_id)
                        # logger.info("before covered examples by rule:", covered_examples)
                        covered_examples.setdefault(new_rule_id, set()).add(
                            example_id)
                        # Delete entry for old rule only if the new rule is different from the old one
                        if old_rule_id in covered_examples and old_rule_id != new_rule_id:
                            covered_examples[old_rule_id].discard(example_id)
                            if len(covered_examples[old_rule_id]) == 0:
                                del covered_examples[old_rule_id]
                        # logger.info("after covered examples by rule:", covered_examples)

        # Reset data because it was updated in find_nearest_examples() in find_nearest_rule(), namely in
        # _update_data_about_closest_rule()
        if has_changed:
            self.closest_rule_per_example = backup_closest_rule
            self.closest_examples_per_rule = backup_closest_examples_per_rule
            self.examples_covered_by_rule = backup_covered
        return f1(
            conf_matrix), conf_matrix, closest_rule_per_example, closest_examples_per_rule, covered_examples, \
               updated_example_ids

    def is_duplicate(self, new_rule, existing_rule_ids):
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
                possible_duplicate = self.all_rules[rule_id]
                if _are_duplicates(new_rule, possible_duplicate):
                    duplicate_rule_id = possible_duplicate.name
                    break
        return duplicate_rule_id

    def find_duplicate_rule_id(self, generalized_rule, rule_hash):
        """
        Checks if a new rule is unique and otherwise returns the ID of the rule that has the same hash as the new rule.

        Parameters
        ----------
        generalized_rule: pd.Series - new rule that was generalized
        rule_hash: int - hash value of the rule

        Returns
        -------
        int.
        The ID of the duplicate rule or my_vars.UNIQUE_RULE if the new rule is unique.

        """
        duplicate_rule_id = my_vars.UNIQUE_RULE
        # Hash collisions might occur, so there could be multiple rules with the same hash value
        if rule_hash in self.unique_rules:
            existing_rule_ids = self.unique_rules[rule_hash]
            logger.info("existing rule ids", existing_rule_ids)
            for rid in existing_rule_ids:
                logger.info(rid)
                logger.info("possible duplicate:", self.all_rules[rid])
            duplicate_rule_id = self.is_duplicate(generalized_rule,
                                                  existing_rule_ids)
        return duplicate_rule_id

    def _delete_old_rule_hash(self, rule):
        """
        Deletes the hash of the old rule.

        Parameters
        ----------
        rule: pd.Series - rule that was deleted

        """
        rule_hash = compute_hashable_key(rule)
        logger.info("delete old hash of {}: {}".format(rule.name, rule_hash))
        logger.info("before update:", self.unique_rules)
        # logger.info("remove old hash of rule {}: {}".format(rule.name, old_hash))
        # rules_with_same_hash = self.unique_rules[old_hash]
        # if len(rules_with_same_hash) > 1:
        #     self.unique_rules[old_hash].discard(rule.name)
        # else:
        #     del self.unique_rules[old_hash]

        rules_with_same_hash = self.unique_rules.get(rule_hash, set())
        if len(rules_with_same_hash) > 1:
            self.unique_rules[rule_hash].discard(rule.name)
        # If a rule was extended, it wasn't added to self.unique_rules, so the additional check is necessary
        elif rule_hash in self.unique_rules:
            del self.unique_rules[rule_hash]
        logger.info("after update:", self.unique_rules)

    def merge_rule_statistics_of_duplicate(self, existing_rule,
                                           duplicate_rule):
        """
        Merges the statistics of a rule, that was just generalized and became a duplicate of an existing rule, with the
        statistics of the existing rule, s.t. the generalized rule is deleted and all statistics are updated for the
        existing rule.
        IMPORTANT: <duplicate_rule> is the base rule from which <duplicate_rule> was generalized

        Parameters
        ----------
        existing_rule: pd.Series - existing rule whose statistics will be updated
        duplicate_rule: pd.Series - base rule that was generalized and became a duplicate, thus it's statistics will be
                        deleted once they were added to <existing_rule>

        """
        logger.info("existing rule", existing_rule.name)
        logger.info("duplicate rule", duplicate_rule.name)
        # 1. Update existing rule
        duplicate_seed_example_id = self.seed_rule_example[duplicate_rule.name]
        # existing_seed_example_id = self.seed_rule_example[existing_rule.name]

        logger.info("seed example per rule:", self.seed_rule_example)
        # self.seed_rule_example[existing_rule.name] = duplicate_seed_example_id

        logger.info("rules for which the examples are seeds:",
                    self.seed_example_rule)
        # self.seed_example_rule[existing_seed_example_id].add(duplicate_rule.name)

        logger.info("updating which rule covers which examples:",
                    self.examples_covered_by_rule)
        covered = self.examples_covered_by_rule.get(duplicate_rule.name, set())
        if len(covered) > 0:
            self.examples_covered_by_rule[existing_rule.name] = \
                self.examples_covered_by_rule.get(existing_rule.name,
                                                  set()).union(covered)
        logger.info("after merging:", self.examples_covered_by_rule)

        affected_examples = self.closest_examples_per_rule.get(
            duplicate_rule.name, set())

        logger.info("closest rule per example", self.closest_rule_per_example)
        for example_id in affected_examples:
            _, dist = self.closest_rule_per_example[example_id]
            self.closest_rule_per_example[example_id] = Data(
                rule_id=existing_rule.name, dist=dist)
        logger.info("after update:", self.closest_rule_per_example)

        logger.info("closest examples per rule:",
                    self.closest_examples_per_rule)
        self.closest_examples_per_rule[existing_rule.name] = \
            self.closest_examples_per_rule.get(existing_rule.name,
                                               set()).union(affected_examples)

        # 2. Delete statistics of duplicate rule
        del self.seed_rule_example[duplicate_rule.name]
        if len(self.seed_example_rule[duplicate_seed_example_id]) > 1:
            self.seed_example_rule[duplicate_seed_example_id].discard(
                duplicate_rule.name)
        else:
            del self.seed_example_rule[duplicate_seed_example_id]
        logger.info("seed example rule updated", self.seed_example_rule)
        logger.info("seed rule example updated", self.seed_rule_example)

        if duplicate_rule.name in self.examples_covered_by_rule:
            del self.examples_covered_by_rule[duplicate_rule.name]

        if duplicate_rule.name in self.closest_examples_per_rule:
            del self.closest_examples_per_rule[duplicate_rule.name]
        logger.info("closest examples per rule after merging:",
                    self.closest_examples_per_rule)

        del self.all_rules[duplicate_rule.name]

        self._delete_old_rule_hash(duplicate_rule)

    def add_one_best_rule(self, df, neighbors, rule, rules, f1, class_col_name,
                          min_max, classes):
        """
        Implements AddOneBestRule() from the paper, i.e. Algorithm 3.

        Parameters
        ----------
        neighbors: pd.DataFrame - nearest examples for <rule>
        rule: pd.Series - rule whose effect on the F1 score should be evaluated
        rules: list of pd.Series - list of all rules in the rule set RS and <rule> is at the end in that list
        class_col_name: str - name of the column in the series holding the class label
        min_max: pd.DataFrame - min and max value per numeric feature.
        classes: list of str - class labels in the dataset.

        Returns
        -------
        bool, list of pd.Series, float or bool, None, float.
        True if a generalized version of the rule improves the F1 score, False otherwise. Returns the updated list of
        rules - all rules in that list are unique, i.e. if the best found rule becomes identical with any existing one
        (that isn't updated), it'll be ignored. The new F1 score using the generalized rule.
        Returns False, <rules>, <f1> if <neighbors> is None

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
        best_hash = None
        logger.info("rule:\n{}".format(rule))
        logger.info("best f1:", best_f1)
        # No neighbors
        if neighbors is None:
            return False, rules, best_f1
        dtypes = neighbors.dtypes
        for example_id, example in neighbors.iterrows():
            logger.info(
                "add_1 generalize rule for example {}".format(example.name))
            generalized_rule = most_specific_generalization(example, rule,
                                                            class_col_name,
                                                            dtypes)
            # logger.info("generalized rule:\n{}".format(generalized_rule))
            current_f1, current_conf_matrix, current_closest_rule, current_closest_examples_per_rule, current_covered, _ \
                = self.evaluate_f1_temporarily(df, generalized_rule,
                                               generalized_rule.name,
                                               class_col_name, min_max,
                                               classes)
            logger.info(current_f1, best_f1)

            if current_f1 >= best_f1 and current_f1 != 1.0:
                logger.info("{} >= {}".format(current_f1, f1))
                best_f1 = current_f1
                best_generalization = generalized_rule
                best_closest_examples_per_rule = current_closest_examples_per_rule
                best_covered = current_covered
                improved = True
                best_conf_matrix = current_conf_matrix
                best_closest_rule_dist = current_closest_rule
                best_hash = compute_hashable_key(generalized_rule)

        if improved:
            logger.info("improvement!")
            # Replace old rule with new one. Note that <rule> (see parameters) is the last rule in <rules>
            idx = -1
            # replace_rule = False
            # Only update existing if its generalization isn't a duplicate - if it is, just delete the existing rule
            duplicate_rule_id = self.find_duplicate_rule_id(
                best_generalization, best_hash)
            # Generalized rule isn't a duplicate
            if duplicate_rule_id == my_vars.UNIQUE_RULE:
                # Delete old hash entry first before adding a new one
                self._delete_old_rule_hash(rule)
                logger.info("replace rule (add_one_best)",
                            best_generalization.name)
                logger.info("###############")
                logger.info("###############")
                # Note that a hash collision could've occurred, i.e. there are different rules with the same hash, so just
                # add to the set of IDs instead of assuming an empty set
                self.unique_rules.setdefault(best_hash, set()).add(
                    best_generalization.name)
                logger.info("updated unique rules:", self.unique_rules)
                rules[idx] = best_generalization
                self.all_rules[best_generalization.name] = best_generalization
                # logger.info("updated best rule per example for example {}:\n{}"
                #             .format(best_generalization.name, (
                #     rule.name,
                #     best_closest_rule_dist[best_generalization.name])))
                self.closest_rule_per_example = best_closest_rule_dist
                logger.info("closest rule per example updated",
                            self.closest_rule_per_example)
                self.closest_examples_per_rule = best_closest_examples_per_rule
                logger.info("closest examples per rule updated",
                            self.closest_examples_per_rule)
                self.examples_covered_by_rule = best_covered
                logger.info("covered examples by rule updated",
                            self.examples_covered_by_rule)
                self.conf_matrix = best_conf_matrix
                logger.info("updated conf matrix", self.conf_matrix)
                logger.info("best f1:", best_f1)
            # Generalized rule is a duplicate
            else:
                logger.info("Duplicate rules!!!!")
                logger.info("Duplicate rule: \n{}".format(duplicate_rule_id))
                logger.info(self.all_rules[duplicate_rule_id])
                logger.info(
                    "best rule according to add_best_rule():\n{}".format(
                        best_generalization))
                logger.info("so doN't add the new rule")
                # Remove current rule that was generalized, which was added to the end of the list
                del rules[idx]
                # Important: use the original rule here because otherwise the generated hash will result in the one
                # that we want to keep because after generalization its hash became the same as the existing rule's hash
                self.merge_rule_statistics_of_duplicate(
                    self.all_rules[duplicate_rule_id], rule)
        return improved, rules, best_f1

    def add_all_good_rules(self, df, neighbors, rule, rules, f1,
                           class_col_name, min_max, classes):
        """
        Implements AddAllGoodRules() from the paper, i.e. Algorithm 3.

        Parameters
        ----------
        df: pd.DataFrame - examples
        neighbors: pd.DataFrame - nearest examples for <rule>
        rule: pd.Series - rule whose effect on the F1 score should be evaluated
        rules: list of pd.Series - list of all rules in the rule set RS
        class_col_name: str - name of the column in the series holding the class label
        min_max: pd.DataFrame - min and max value per numeric feature.
        classes: list of str - class labels in the dataset.

        Returns
        -------
        bool, list of pd.Series, float ; or bool, pd.Series, None.
        True if a generalized version of the rule improves the F1 score, False otherwise.  Returns the updated list of
        rules - all rules in that list are unique, i.e. if a new better rule becomes identical with any existing one
        (that isn't updated), it'll be ignored. New F1-score for the generalized and possibly added rules.
        Returns False, <rules>, <f1> if <neighbors> is None

        """
        # To be able to compute the hash of the rule that replaced <rule> in iteration 0 -
        # it's necessary to delete old hashes later
        replaced_rule = None
        improved = False

        best_f1 = f1
        is_first_rule = True
        iteration = 0
        changed_rules = []
        # Keep track of how many rules were added to access the original rule in the list in O(1)
        added_rules = 0
        # Duplicate rules
        duplicates = None
        if neighbors is None:
            return False, rules, best_f1
        dtypes = neighbors.dtypes
        logger.info("initial neighbors")
        logger.info(neighbors)
        while not is_empty(neighbors):
            for example_id, example in neighbors.iterrows():
                logger.info(
                    "\nadd_all generalize rule {} for example {}".format(
                        rule.name, example_id))
                logger.info("-------------------------------------------")
                # Assume that the generalized rule will be added, so we generate a new ID for it in advance
                original_rule_id = rule.name
                self.latest_rule_id += 1

                # logger.info("old rule:\n{}".format(rule))
                generalized_rule = most_specific_generalization(example,
                                                                rule,
                                                                class_col_name,
                                                                dtypes)

                # Remove current example
                neighbors.drop(example_id, inplace=True)

                # Contrary to add_one_best_rule(), we can already remove duplicates already here because we accept any
                # improvement instead of just the best improvement
                rule_hash = compute_hashable_key(generalized_rule)
                # Check if the generalized rule is a duplicate of an existing one
                logger.info("possible new rule")
                logger.info(generalized_rule)
                duplicate_rule_id = self.find_duplicate_rule_id(
                    generalized_rule, rule_hash)
                # Generalized rule isn't a duplicate
                if duplicate_rule_id == my_vars.UNIQUE_RULE:
                    current_f1, current_conf_matrix, current_closest_rule, current_closest_examples, current_covered, \
                    updated_example_ids = \
                        self.evaluate_f1_temporarily(df, generalized_rule,
                                                     self.latest_rule_id,
                                                     class_col_name,
                                                     min_max, classes)
                    # Generalized rule is better
                    if current_f1 >= best_f1:
                        improved = True
                        logger.info("{} >= {}".format(current_f1, f1))
                        best_f1 = current_f1
                        logger.info("improvement with the following rule:")
                        logger.info(generalized_rule)

                        # Only update the mapping if the new rule has become the closest rule for >= 1 example,
                        # which might not be the case. For example, if an example is already covered by a different
                        # rule, and the newly generalized one would cover it too, but has the same number of features
                        # as the rule that already covers it, then there's no need to update anything in my opinion.
                        # The logic that decides if a new rule is closer is handled in find_nearest_rule() and that
                        # result is passed back as a Boolean variable
                        if is_first_rule:
                            is_first_rule = False
                            logger.info(
                                "replace rule {} which had as original id {}".format(
                                    generalized_rule.name,
                                    original_rule_id))
                            logger.info("###############")
                            logger.info("###############")
                            changed_rules.append((example, generalized_rule))
                            # Note that a hash collision could've occurred, i.e. there are different rules with the same
                            # hash, so just
                            # add to the set of IDs instead of assuming an empty set
                            self.unique_rules.setdefault(rule_hash, set()).add(
                                generalized_rule.name)
                            logger.info(
                                "added {} for rule {}".format(rule_hash,
                                                              generalized_rule.name))
                            replaced_rule = generalized_rule

                            # Rule to be replaced is at the end
                            idx = -1

                            # Delete old hash entry first before adding a new one
                            self._delete_old_rule_hash(rule)

                            # Replace old rule with new one - old rule is at the end of the list
                            rules[idx] = generalized_rule
                            self.all_rules[
                                generalized_rule.name] = generalized_rule

                            # Generalized rule replaces existing one, but we updated statistics assuming that
                            # so use the original one's ID and allow reuse of this new ID
                            wrong_rule_id = self.latest_rule_id
                            logger.info(
                                "wrong ID used to temporarily compute f1:",
                                wrong_rule_id)
                            self.latest_rule_id -= 1
                            logger.info("old closest rule per example:",
                                        current_closest_rule)

                            # We updated statistics assuming that the new rule ID would be used, but this isn't the
                            # case, so we need to rollback and merge the results of the original rule ID with
                            # those of the new rule ID. Note that the new rule might not have become the closest rule for
                            # any example, so maybe no entry needs to be updated

                            # if wrong_rule_id in current_closest_rule:
                            #     current_closest_rule[example_id] = Data(rule_id=original_rule_id,
                            #                                             dist=current_closest_rule[example_id].dist)
                            # self.closest_rule_per_example = current_closest_rule
                            # logger.info("after update", current_closest_rule)

                            logger.info("old closest examples per rule",
                                        current_closest_examples)
                            # Again, the new rule might have not become the closest rule, so only update if it did
                            closest_examples_for_new_rule = current_closest_examples.get(
                                wrong_rule_id, set())
                            if len(closest_examples_for_new_rule) > 0:
                                for eid in closest_examples_for_new_rule:
                                    current_closest_rule[eid] = Data(
                                        rule_id=original_rule_id,
                                        dist=current_closest_rule[eid].dist)
                                self.closest_rule_per_example = current_closest_rule
                                logger.info("new closest rule per example:",
                                            current_closest_rule)

                                current_closest_examples[original_rule_id] = \
                                    current_closest_examples.get(
                                        original_rule_id, set()).union(
                                        closest_examples_for_new_rule)
                            logger.info(
                                "intermediate closest examples per rule",
                                current_closest_examples)
                            # A new rule might not be closest to any example
                            if wrong_rule_id in current_closest_examples:
                                del current_closest_examples[wrong_rule_id]
                            self.closest_examples_per_rule = current_closest_examples
                            logger.info("new closest examples per rule",
                                        self.closest_examples_per_rule)

                            logger.info("old covered rules", current_covered)
                            # Again, the new rule might have not become the closest rule, so only update if it did
                            covered_examples_by_new_rule = current_covered.get(
                                wrong_rule_id, set())
                            if len(covered_examples_by_new_rule) > 0:
                                current_covered[original_rule_id] = \
                                    current_covered.get(original_rule_id,
                                                        set()).union(
                                        covered_examples_by_new_rule)
                            logger.info("intermediate covered rules",
                                        current_covered)

                            if wrong_rule_id in current_covered:
                                del current_covered[wrong_rule_id]
                            self.examples_covered_by_rule = current_covered
                            logger.info("new current covered rules",
                                        self.examples_covered_by_rule)

                            for eid in updated_example_ids:
                                logger.info("closest rule",
                                            self.closest_rule_per_example[eid])
                                self.closest_rule_per_example[eid] = \
                                    Data(rule_id=original_rule_id,
                                         dist=self.closest_rule_per_example[
                                             eid].dist)
                                logger.info("updated rule ID closest rule",
                                            self.closest_rule_per_example[eid])

                            # Confusion matrix won't change, so no need to update it
                            self.conf_matrix = current_conf_matrix

                            # Sort remaining neighbors ascendingly w.r.t. the distance to the generalized rule
                            # Note that we use ALL_LABELS in find_nearest_rule() because we just want to re-sort the
                            # neighbors
                            # Otherwise no distance will be returned for examples with other labels than <generalized_rule>
                            dists = []
                            for neighbor_id, neighbor in neighbors.iterrows():
                                _, dist, _ = self.find_nearest_rule(
                                    [generalized_rule], neighbor,
                                    class_col_name,
                                    min_max, classes,
                                    self.examples_covered_by_rule,
                                    label_type=my_vars.ALL_LABELS,
                                    only_uncovered_neighbors=False)
                                dists.append((neighbor_id, dist))
                            logger.info("recomputed distances:")
                            logger.info(dists)
                            # At least 1 example still exists after dropping the previous one
                            if len(dists) > 0:
                                dists.sort(key=itemgetter(1))
                                example_ids, dists = map(list, (zip(*dists)))
                                neighbors = neighbors.loc[example_ids]
                            # Stop current loop because neighbors' distance was recomputed based on the generalized rule
                            break
                        else:
                            # Add generalized rule instead of replacing the original one
                            logger.info(
                                "add rule {}!!!".format(self.latest_rule_id))
                            logger.info("###############")
                            logger.info("###############")
                            logger.info(generalized_rule)
                            logger.info("closest new rule per example",
                                        current_closest_rule)
                            self.closest_rule_per_example = current_closest_rule
                            self.closest_examples_per_rule = current_closest_examples
                            self.conf_matrix = current_conf_matrix
                            self.examples_covered_by_rule = current_covered

                            logger.info("original rule id:",
                                        generalized_rule.name)
                            new_rule_id = self.latest_rule_id
                            generalized_rule.name = new_rule_id
                            logger.info("new rule id", generalized_rule.name)

                            logger.info("before adding unique hash:",
                                        self.unique_rules)
                            # new_hash = compute_hashable_key(generalized_rule)
                            is_added = True
                            if rule_hash not in self.unique_rules:
                                self.unique_rules[rule_hash] = {new_rule_id}
                            else:
                                generalized_rule.name = self.latest_rule_id
                                logger.info(
                                    "hash collision when adding new rule!")
                                logger.info("new rule:")
                                logger.info(generalized_rule)
                                # Hash collisions might occur, so there could be multiple rules with the same hash value
                                existing_rule_ids = self.unique_rules[
                                    rule_hash]
                                existing_rule_id = self.is_duplicate(
                                    generalized_rule, existing_rule_ids)
                                # No duplicate exists
                                if existing_rule_id != my_vars.UNIQUE_RULE:
                                    is_added = False
                            logger.info("after adding unique hash:",
                                        self.unique_rules)
                            # Only add if the generalized rule is no duplicate
                            if is_added:

                                # Note that a hash collision could've occurred, i.e. there are different rules with the same
                                # hash, so just
                                # add to the set of IDs instead of assuming an empty set
                                self.unique_rules.setdefault(rule_hash,
                                                             set()).add(
                                    generalized_rule.name)
                                logger.info(
                                    "added {} for rule {}".format(rule_hash,
                                                                  generalized_rule.name))
                                added_rules += 1
                                logger.info(
                                    "before updating seed: example_rule:",
                                    self.seed_example_rule)
                                self.seed_example_rule.setdefault(example_id,
                                                                  set()).add(
                                    new_rule_id)
                                logger.info(
                                    "after updating seed: example_rule:",
                                    self.seed_example_rule)
                                logger.info(
                                    "before updating seed: rule_example_rule:",
                                    self.seed_rule_example)
                                self.seed_rule_example[
                                    new_rule_id] = example_id
                                logger.info(
                                    "after updating seed: rule_example_rule:",
                                    self.seed_rule_example)

                                # Use the newly generated ID as name
                                generalized_rule.name = self.latest_rule_id
                                self.all_rules[
                                    generalized_rule.name] = generalized_rule
                                changed_rules.append(
                                    (example, generalized_rule))
                                rules.append(generalized_rule)
                                logger.info("new rule id:",
                                            generalized_rule.name)
                                logger.info("added rule for example {}:\n{}"
                                            .format(example_id, (
                                    generalized_rule.name,
                                    current_closest_rule[example_id])))
                                logger.info("covered:",
                                            self.examples_covered_by_rule)
                                logger.info("closest rule per example",
                                            self.closest_rule_per_example)
                                logger.info("closest examples per rule",
                                            self.closest_examples_per_rule)
                                logger.info("conf matrix", self.conf_matrix)

                            else:
                                # Rule was a duplicate, so reset the last ID
                                self.latest_rule_id -= 1
                    else:
                        # F1 score wasn't improved, so allow a reuse of this rule ID
                        self.latest_rule_id -= 1
                # Generalized rule is a duplicate
                else:
                    # Don't delete duplicates here because generalizing <rule> for other examples might lead to
                    # improvements - delete after loop

                    # Note that duplicate rules haven't been added yet, so there's no need to update any statistics - it's
                    # only necessary to delete duplicates if <rule>'s generalization isn't closer to at least 1 example -
                    # be it in iteration 0 or later

                    # Initially, the original rule was added at the last position in the list, but an arbitrary number of
                    # new rules could've been added in the meantime; +1 because it's 0-based
                    idx_original_rule = added_rules + 1
                    # Remove current rule that was generalized, which was added to the end of the list
                    logger.info(
                        "remove rule {} after some rules might've been added".format(
                            rules[-idx_original_rule].name))
                    logger.info("keep rule {} and remove rule {}".format(
                        self.all_rules[duplicate_rule_id].name, rule.name))
                    logger.info("existing rule")
                    logger.info(self.all_rules[duplicate_rule_id])
                    # del rules[-idx_original_rule]
                    # Might be None if we're still in iteration 0, but the generalized rule is already a duplicate
                    # if replaced_rule is None:
                    #     logger.info("iteration 0, replaced rule is None", replaced_rule)
                    #     # TODO: don't do anything????
                    #
                    #     duplicates = Duplicates(original=self.all_rules[duplicate_rule_id],
                    #                             duplicate=generalized_rule, duplicate_idx=idx_original_rule)
                    #     # deleting rule 4 isn't possible here because it could be potentially generalized for other examples
                    #     # merge_rule_statistics_of_duplicate(self.all_rules[duplicate_rule_id], generalized_rule)
                    # else:
                    #     logger.info("iteration !=0, replaced rule is not None", replaced_rule)
                    #     # Important: use the original rule here because otherwise the generated hash will result in the one
                    #     # that we want to keep because after generalization its hash became the same as the existing rule's
                    #     # hash
                    #     # merge_rule_statistics_of_duplicate(self.all_rules[duplicate_rule_id], replaced_rule)
                    # Potential infinite loop here if it's the last neighbor and only a duplicate is found, so we
                    # need to end the loop. If there are more neighbors available, continue with them
                    if not neighbors.empty:
                        last_row = neighbors.iloc[-1, :]
                        if last_row.equals(example):
                            neighbors = pd.DataFrame()
                            break
                        else:
                            continue
            iteration += 1
            logger.info("end of iteration {} in add_all()".format(iteration))
            logger.info("#####################\n")
        # IMPORTANT: don't delete here -> improved=False, meaning that bracid() will remove the rule after extending it
        # If generalizations of <rule> improved F1-score, but introduced duplicates, delete <rule>
        # if not improved:
        #     orig_rule, duplicate_rule, duplicate_idx = duplicates
        #     logger.info("generalizing <rule> {} was introducing only duplicates, so <rule> will be removed"
        #           .format(duplicate_rule.name))
        #
        #     logger.info("delete rule {} and keep rule {}".format(duplicate_rule.name, orig_rule.name))
        #     # <rule> is at the end of the list, potentially other generalizations of <rule> were added thereafter
        #     del rules[-duplicate_idx]
        #     merge_rule_statistics_of_duplicate(orig_rule, duplicate_rule)
        # else:
        #     logger.info("no duplicates deleted in add_all() because the following rules were closer to examples:")
        #     for example, rule in changed_rules:
        #         logger.info("rule {} for example {}".format(rule.name, example.name))

        # if not was_replaced:
        #     # Initially, the original rule was added at the last position in the list, but an arbitrary number of
        #     # new rules could've been added in the meantime; +1 because it's 0-based
        #     idx_original_rule = added_rules + 1
        #     del rules[-idx_original_rule]
        #     merge_rule_statistics_of_duplicate(self.all_rules[duplicate_rule_id], generalized_rule)
        logger.info(self.closest_examples_per_rule)
        logger.info(self.closest_rule_per_example)
        logger.info(self.examples_covered_by_rule)
        logger.info(self.conf_matrix)
        return improved, rules, best_f1

    def extend_rule(self, df, k, rule, class_col_name, min_max, classes):
        """
        Extends a rule in terms of numeric features according to algorithm 4 of the paper, i.e. extend().

        Parameters
        ----------
        df: pd.DataFrame - dataset
        k: int - number of neighbors with opposite label of <rule> to consider
        rule: pd.Series - rule
        class_col_name: str - name of class label
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset. It's assumed to be binary.

        Returns
        -------
        pd.Series.
        Extended rule.

        """
        neighbors, _, _ = self.find_nearest_examples(df, k, rule,
                                                     class_col_name, min_max,
                                                     classes,
                                                     label_type=my_vars.OPPOSITE_LABEL_TO_RULE,
                                                     only_uncovered_neighbors=True)
        # logger.info("neighbors")
        # logger.info(neighbors)
        # logger.info("rule before extension:\n{}".format(rule))
        # dtypes = rule.apply(type).tolist()
        # logger.info("data types", dtypes)
        if neighbors is not None:
            for col_name, col_val in rule.items():
                # Only numeric features - they're stored in a named tuple
                if isinstance(col_val, Bounds):
                    lower_rule, upper_rule = col_val
                    # logger.info("lower: {} upper: {}".format(lower_rule, upper_rule))
                    # logger.info("neighbors")
                    # logger.info(neighbors)
                    remaining_lower = neighbors.loc[
                        neighbors[col_name] < lower_rule]
                    remaining_upper = neighbors.loc[
                        neighbors[col_name] > upper_rule]
                    # logger.info("neighbors meeting lower constraint:\n{}".format(remaining_lower))
                    # logger.info("neighbors meeting upper constraint:\n{}".format(remaining_upper))
                    new_lower = 0
                    new_upper = 0
                    # Extend left towards nearest neighbor
                    if not is_empty(remaining_lower):
                        lower_example = remaining_lower[col_name].max()
                        # logger.info("lower val", lower_example)
                        new_lower = 0.5 * (lower_rule - lower_example)
                    # Extend right towards nearest neighbor
                    if not is_empty(remaining_upper):
                        upper_example = remaining_upper[col_name].min()
                        # logger.info("upper val", upper_example)
                        new_upper = 0.5 * (upper_example - upper_rule)
                    rule[col_name] = Bounds(lower=lower_rule - new_lower,
                                            upper=upper_rule + new_upper)
                    # logger.info("rule after extension of current column:\n{}".format(rule))
        self.all_rules[rule.name] = rule
        return rule

    def delete_rule_statistics(self, df, rule, rules, final_rules,
                               class_col_name, min_max, classes):
        """
        Deletes all statistics related to a specific rule.

        Parameters
        ----------
        df: pd.DataFrame - dataset
        rule: pd.Series - rule that was removed
        rules: list of pd.Series - list of candidate rules
        final_rules: dict - dictionary of final rules with rule IDs as keys and rules (pd.Series) as value
        class_col_name: str - name of class label
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset.

        Raises
        ------
        AssertionError: if there's no rule left in <rules> and <final_rules> that has the same class label as <example>

        """
        # Only delete rule statistics if a rule wasn't added to the set of final rules earlier. This could happen if a
        # rule of the minority class was extended in extend_rule() - afterwards it would still be deleted from the set of
        # candidate rules
        logger.info("rule {} might get deleted now".format(rule.name))
        logger.info("available rules", len(rules))
        logger.info("available final rules", len(final_rules))
        if rule.name not in final_rules:

            logger.info("Rule that was deleted: \n{}".format(rule.name))
            logger.info("delete seed rule_example entry: {}:{} "
                        .format(rule.name, self.seed_rule_example[rule.name]))
            old_seed_example_id = self.seed_rule_example[rule.name]
            del self.seed_rule_example[rule.name]
            logger.info("remaining entries:", self.seed_rule_example)

            logger.info("delete seed example_rule entry:",
                        self.seed_example_rule[old_seed_example_id])
            del self.seed_example_rule[old_seed_example_id]
            logger.info("remaining entries:", self.seed_example_rule)

            logger.info("updating which rule covers which examples:",
                        self.examples_covered_by_rule)
            if rule.name in self.examples_covered_by_rule:
                del self.examples_covered_by_rule[rule.name]
            logger.info("after update", self.examples_covered_by_rule)

            affected_example_ids = self.closest_examples_per_rule.get(
                rule.name, set())
            logger.info("closest rule per example before update:",
                        self.closest_examples_per_rule)
            logger.info("affected examples", affected_example_ids)
            # TODO: should the distances of each example to each rule be stored in memory for fast look-up????
            # To find the closest rule, check the remaining rules as well as the final rules
            for example_id in affected_example_ids:
                # Delete existing entry because otherwise find_nearest_rule() won't update the distance properly as one
                # rule,the one that was just deleted, was closer
                logger.info("deleted closest rule for example {}: {}:"
                            .format(example_id,
                                    self.closest_rule_per_example[example_id]))
                del self.closest_rule_per_example[example_id]
                example = df.loc[example_id]
                logger.info(
                    "example: {} old rule: {}".format(example[class_col_name],
                                                      rule[class_col_name]))
                # Closest rule
                rem_rule, rem_dist, rem_is_updated = self.find_nearest_rule(
                    rules, example, class_col_name, min_max,
                    classes, self.examples_covered_by_rule,
                    label_type=my_vars.SAME_LABEL_AS_RULE,
                    only_uncovered_neighbors=False)
                fin_rule, fin_dist, fin_is_updated = self.find_nearest_rule(
                    final_rules.values(), example, class_col_name,
                    min_max, classes,
                    self.examples_covered_by_rule,
                    label_type=my_vars.SAME_LABEL_AS_RULE,
                    only_uncovered_neighbors=False)
                closest_rule = rem_rule
                closest_dist = rem_dist
                # Sanity check: it's impossible that both rule sets were empty, thus a neighbor must exist
                # assert(rem_rule is not None or fin_rule is not None)
                # Only True if a final rule was closer than one of the candidate rules - if not, there must've been at
                # least 1 candidate rule and it's closer than any of the final rules
                if fin_is_updated:
                    closest_rule = fin_rule
                    closest_dist = fin_dist
                if rem_rule is None and fin_rule is None:
                    raise AssertionError(
                        "no rules remain with the label '{}'".format(
                            example[class_col_name]))
                logger.info(rem_rule)
                logger.info(fin_rule)
                # logger.info("nearest rule")
                # logger.info(closest_rule)
                logger.info("new nearest rule: {} with dist {}".format(
                    closest_rule.name, closest_dist))
                self.closest_rule_per_example[example_id] = Data(
                    rule_id=closest_rule.name, dist=closest_dist)
                logger.info(self.closest_rule_per_example)
                self.closest_examples_per_rule.setdefault(closest_rule.name,
                                                          set()).add(
                    example_id)
            logger.info("closest rule per example after update:",
                        self.closest_examples_per_rule)

            logger.info("closest examples per rule before update:",
                        self.closest_examples_per_rule)
            if rule.name in self.closest_examples_per_rule:
                del self.closest_examples_per_rule[rule.name]
            logger.info("closest examples per rule after update:",
                        self.closest_examples_per_rule)

            logger.info("all rules before", self.all_rules)
            del self.all_rules[rule.name]
            logger.info("all rules after", self.all_rules)

            logger.info("unique rules before", self.unique_rules)
            logger.info("rule_id", rule.name)
            self._delete_old_rule_hash(rule)
            # rule_hash = compute_hashable_key(rule)
            # rules_with_same_hash = self.unique_rules.get(rule_hash, set())
            # if len(rules_with_same_hash) > 1:
            #     self.unique_rules[rule_hash].discard(rule.name)
            # # If a rule was extended, it wasn't added to self.unique_rules, so the additional check is necessary
            # elif rule_hash in self.unique_rules:
            #     del self.unique_rules[rule_hash]
            logger.info("all rules after", self.unique_rules)
        else:
            logger.info(
                "Rule that was deleted, but added to final rules earlier - hence, it's not deleted: \n{}\n\n\n"
                    .format(rule.name))

    def _in_final_rules(self, rule, final_rules, final_rules_hashes):
        rule_hash = compute_hashable_key(rule)
        if not rule_hash in final_rules_hashes:
            return False
        for final_rule in final_rules:
            if _are_duplicates(rule, final_rule):
                return True
        return False

    def _fit_binary(self, df):
        if self._minority_class is None:
            # raise ValueError('Provide the minority class for binary prediction')
            self._minority_class = \
                df[self._class_column_name].value_counts().index[-1]
            logger.info(
                'No minority class specified for binary classification. Choosing %s',
                self._minority_class)
        self._rules = self.bracid(df, self.k, self._class_column_name,
                                  self._min_max, self._classes,
                                  self._minority_class)
        self._model = self.train_binary(self._rules, df, self._minority_class,
                                        self._class_column_name)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        assert y.shape[0] > 0, 'y cannot be empty'
        y_nominal = [str(v) for v in y]
        self._classes = list(set(y_nominal))
        self._is_binary_classification = len(self._classes) <= 2
        self._columns = [f'Column_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self._columns)
        self._min_max = get_min_max(df)
        df[self._class_column_name] = y_nominal
        if self._is_binary_classification:
            return self._fit_binary(df)
        return self._fit_multiclass(df)

    def _predict_binary(self, df, predict_proba=False):
        preds_df = self.predict_binary(self._model, df, self._rules,
                                       self._classes,
                                       self._class_column_name, self._min_max,
                                       for_multiclass=False)
        if predict_proba:
            preds = preds_df[my_vars.PREDICTION_CONFIDENCE].values
        else:
            preds = preds_df[my_vars.PREDICTED_LABEL].values
        return preds

    def _fit_multiclass(self, df):
        minority_labels = self._classes
        logger.info("one-vs-all labels in the order they'll be processed",
                    minority_labels)
        multiclass_fit_data = {}
        for minority_label in minority_labels:
            classes = [minority_label, Labels.REST]
            # Convert to a binary problem
            train = df.copy()
            train = to_binary_classification_task(train,
                                                  self._class_column_name,
                                                  minority_label,
                                                  merged_label=Labels.REST)
            logger.info("####training set#####"
                        f"{train}"
                        "classes to be used in current run"
                        f"{train[self._class_column_name].unique()}"
                        )
            rules = self.bracid(train, self.k, self._class_column_name,
                                self._min_max, classes,
                                minority_label)
            model = self.train_binary(rules, train, minority_label,
                                      self._class_column_name)
            multiclass_fit_data[minority_label] = {
                "rules": rules,
                "model": model,
                "classes": classes
            }
        self._multiclass_fit_data = multiclass_fit_data

    def _predict_multiclass(self, df, predict_proba=False):
        minority_labels = self._classes
        # Create empty DataFrame with the same number of rows as df
        res = pd.DataFrame(index=df.index)
        # Negative confidence because even if confidence for the first label is 0, it should be used instead of no label
        res[my_vars.PREDICTION_CONFIDENCE] = -1
        res[my_vars.PREDICTED_LABEL] = ""
        all_rules = {}
        for minority_label in minority_labels:
            fit_data = self._multiclass_fit_data[minority_label]
            rules = fit_data["rules"]
            model = fit_data["model"]
            classes = fit_data["classes"]
            preds_df = self.predict_binary(model, df, rules, classes,
                                           self._class_column_name,
                                           self._min_max,
                                           for_multiclass=True)
            all_rules[minority_label] = rules
            # Update predicted label and confidence if confidence is higher than the currently best confidence
            PC = my_vars.PREDICTION_CONFIDENCE
            PL = my_vars.PREDICTED_LABEL
            conf_lower = res[PC] < preds_df[PC]
            not_rest = preds_df[PL] != Labels.REST
            conf_lower_not_rest = conf_lower & not_rest
            res.loc[conf_lower_not_rest, PL] = preds_df[PL]
            res.loc[conf_lower_not_rest, PC] = preds_df[PC]

        if predict_proba:
            prediction = res[my_vars.PREDICTION_CONFIDENCE]
        else:
            prediction = res[my_vars.PREDICTED_LABEL]
        return prediction.values

    def _predict(self, X, y=None, predict_proba=False):
        df = pd.DataFrame(X, columns=self._columns)
        if self._is_binary_classification:
            return self._predict_binary(df, predict_proba=predict_proba)
        else:
            return self._predict_multiclass(df, predict_proba=predict_proba)

    def predict(self, X, y=None):
        return self._predict(X, y, predict_proba=False)

    def predict_proba(self, X, y=None):
        return self._predict(X, y, predict_proba=True)

    def bracid(self, df, k, class_col_name, min_max, classes, minority_label):
        """
        Implements the actual BRACID algorithm according to Algorithm 1 in the paper.

        Parameters
        ----------
        df: pd.DataFrame - dataset
        k: int - number of neighbors with opposite label of the current example to consider
        class_col_name: str - name of class label
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset. It's assumed to be binary.
        minority_label: str - class label of the minority class. Note that all other labels are grouped into another class
        so that there's a binary classification task.

        Returns
        -------
        dictionary of pd.Series.
        Dictionary of rules that classify the training data most accurately according to F1 score. Keys are the rule IDs and
        values the corresponding rules.

        """
        self._minority_class = minority_label
        self.init_statistics(df)
        logger.info("minority class label:", self._minority_class)
        df, rules = self.add_tags_and_extract_rules(df, k, class_col_name,
                                                    min_max, classes)
        logger.info("initial rules")
        logger.info(rules)
        # {rule_id: rule}
        final_rules = {}
        final_rules_hashes = set()
        iteration = 0
        keep_running = True
        for rule in rules:
            rule_hash = compute_hashable_key(rule)
            logger.info("rule/ hash:", rule_hash)
            self.unique_rules.setdefault(rule_hash, set()).add(rule.name)
        f1_ = self.evaluate_f1_initialize_confusion_matrix(df, rules,
                                                           class_col_name,
                                                           min_max, classes)
        while keep_running:
            improved = False
            pbar = tqdm.tqdm(desc=f'BRACID for class {minority_label}')
            loop_iterations = 0
            while len(rules) > 0:
                loop_iterations += 1
                pbar.set_description(
                    f'({minority_label}) There are {len(rules)} rules left for evaluation',
                    refresh=True)
                logger.info("\nthere are {} rules left for evaluation:".format(
                    len(rules)))
                logger.info("hashes:", self.unique_rules)
                rule = rules.popleft()
                rule_id = rule.name
                logger.info(
                    "rule {} is currently being processed:\n{}".format(rule_id,
                                                                       rule))
                # Add current rule at the end

                # if self._in_final_rules(rule, final_rules, final_rules_hashes):
                #     # Doesn't seem to execute
                #     logger.info(f"rule {rule} is present in final_rules. Skipping...")
                #     continue

                rules.append(rule)
                # logger.info("it was now added to the end of all rules:\n{}".format(rules))
                logger.info(self.seed_rule_example)
                seed_id = self.seed_rule_example[rule_id]
                # logger.info("seed id", seed_id)
                # logger.info(df)
                seed = df.loc[seed_id]
                # logger.info("seed\n{}".format(seed))
                seed_label = seed[class_col_name]
                seed_tag = seed[my_vars.TAG]
                # logger.info("seed label:", seed_label)
                logger.info("closest rule per example",
                            self.closest_rule_per_example)
                logger.info("closest examples per rule",
                            self.closest_examples_per_rule)
                logger.info("covered examples", self.examples_covered_by_rule)
                # Minority class label
                if seed_label == minority_label:
                    neighbors, dists, _ = self.find_nearest_examples(df, k,
                                                                     rule,
                                                                     class_col_name,
                                                                     min_max,
                                                                     classes,
                                                                     label_type=my_vars.SAME_LABEL_AS_RULE,
                                                                     only_uncovered_neighbors=True)
                    # Neighbors exist
                    # if neighbors is not None:
                    if seed_tag == ExampleClass.SAFE:
                        improved, generalized_rules, f1_ = self.add_one_best_rule(
                            df, neighbors, rule, rules, f1_,
                            class_col_name, min_max, classes)
                    else:
                        improved, generalized_rules, f1_ = self.add_all_good_rules(
                            df, neighbors, rule, rules, f1_,
                            class_col_name, min_max, classes)
                    if not improved:
                        # Don't extend for outlier
                        if iteration != 0:
                            extended_rule = self.extend_rule(df, k, rule,
                                                             class_col_name,
                                                             min_max, classes)
                            final_rules[extended_rule.name] = extended_rule
                            # Delete rule
                            removed = rules.pop()
                            logger.info(
                                "removed rule after extension:\n{}".format(
                                    removed))
                            self.delete_rule_statistics(df, removed, rules,
                                                        final_rules,
                                                        class_col_name,
                                                        min_max,
                                                        classes)
                    else:
                        # Use updated rules
                        rules = generalized_rules
                # Majority label
                else:
                    n = k
                    if seed_tag == ExampleClass.SAFE:
                        n = 1
                    neighbors, dists, _ = self.find_nearest_examples(df, n,
                                                                     rule,
                                                                     class_col_name,
                                                                     min_max,
                                                                     classes,
                                                                     label_type=my_vars.SAME_LABEL_AS_RULE,
                                                                     only_uncovered_neighbors=True)
                    # Neighbors exist
                    # if neighbors is not None:
                    improved, generalized_rules, f1_ = self.add_one_best_rule(
                        df, neighbors, rule, rules, f1_, class_col_name,
                        min_max, classes)
                    if not improved:
                        # Treat as noise
                        if iteration == 0:
                            example_id = self.seed_rule_example[rule_id]
                            df, rules = self.treat_majority_example_as_noise(
                                df, example_id, rules, rule_id)
                        else:
                            final_rules[rule.name] = rule
                            # Delete rule
                            removed = rules.pop()
                            logger.info(
                                "removed rule after adding majority final rule:\n{}".format(
                                    removed))
                            self.delete_rule_statistics(df, removed, rules,
                                                        final_rules,
                                                        class_col_name,
                                                        min_max,
                                                        classes)
                    else:
                        # Use updated rules
                        rules = generalized_rules
                iteration += 1
                logger.info(
                    "end of iteration {} in bracid()".format(iteration))
                logger.info("#####################\n")
                pbar.update(1)
            if not improved:
                keep_running = False

            pbar.set_description(
                f'There were {loop_iterations} iterations for label={minority_label}',
                refresh=True)
            pbar.close()

        return final_rules

    def treat_majority_example_as_noise(self, df, example_id, rules, rule_id):
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
        logger.info("########noise!##############")
        # Delete rule and corresponding seed (=noisy example)
        # example_id = self.seed_rule_example[rule_id]
        logger.info("delete example id", example_id)
        del self.seed_rule_example[rule_id]
        self.seed_example_rule[example_id].discard(rule_id)
        logger.info("before deleting the example")
        logger.info(df)
        df.drop(example_id, inplace=True)
        logger.info("after")
        logger.info(df)
        # logger.info("remaining entries for {}: {}".format(rule_id, self.seed_example_rule[example_id]))
        if len(self.seed_example_rule[example_id]) == 0:
            # logger.info("deleted the empty entry!")
            del self.seed_example_rule[example_id]
        # logger.info("rules before deletion:")
        logger.info(rules)
        # Delete rule
        removed = rules.pop()
        logger.info(
            "removed rule in majority noisy label:\n{}".format(removed))
        logger.info("rules after deletion:")
        logger.info(rules)
        return df, rules

    def train_binary(self, rules, training_examples, minority_label,
                     class_col_name):
        """
        Trains the model used for predicting class labels of unknown examples. To this end, the support of the derived rules
        is computed in the model. Note that BRACID was used to derive <rules>.
        Deals only with binary labels, i.e. anything that isn't <minority_label>, will be assigned the same class label
        (that is different from <minority_label>).

        Parameters
        ----------
        rules: dict - rules for the dataset, where rule IDs are keys and the rules (pd.Series) are values
        training_examples: pd.DataFrame - training set where each row represents a training example
        minority_label: str - label of the minority class
        class_col_name: str - name of the column with the class label in <training_examples>

        Returns
        -------
        dict.
        The model used for computing the support of each rule. It contains as keys the rule IDs and as value a named tuple
        indicating the support (= % of covered examples whose labels are predicted correctly) for the minority and majority
        labels, respectively

        """
        self._minority_class = minority_label
        model = {}
        logger.info("closest rule per example in train():",
                    self.closest_rule_per_example)
        for rule_id in rules:
            rule = rules[rule_id]
            logger.info(rule)
            if self._minority_class == rule[class_col_name]:
                logger.info(
                    "rule {} predicts minority label '{}'".format(rule.name,
                                                                  rule[
                                                                      class_col_name]))
            else:
                logger.info(
                    "rule {} predicts majority label '{}'".format(rule.name,
                                                                  rule[
                                                                      class_col_name]))
            training_examples[my_vars.COVERED] = training_examples.loc[:, :] \
                .apply(does_rule_cover_example_without_label, axis=1,
                       args=(rule, training_examples.dtypes, class_col_name))
            all_covered_examples = training_examples.loc[
                training_examples[my_vars.COVERED] == True]
            logger.info("all covered")
            logger.info(all_covered_examples)
            # Examples whose labels were predicted correctly by the rule - True or False
            correct = \
                all_covered_examples.loc[
                    rule[class_col_name] == all_covered_examples[
                        class_col_name]]
            logger.info("correctly covered")
            logger.info(correct)
            counts = correct.shape[0]
            total = all_covered_examples.shape[0]
            # Support = #covered examples that were predicted correctly by that rule / all covered examples by that rule
            support = counts / total
            rest = 1 - support
            logger.info(
                "support(rule {}) = {}/{} = {}".format(rule_id, counts, total,
                                                       support))
            if rule[class_col_name] == self._minority_class:
                model[rule_id] = Support(minority=support, majority=rest)
            else:
                model[rule_id] = Support(minority=rest, majority=support)
        return model

    def predict_binary(self, model, test_examples, rules, classes,
                       class_col_name, min_max, for_multiclass=False):
        """
        Predicts the binary class labels of unknown examples. Sums up the support of (potentially multiple) rules that are
        closest to an example.

        Parameters
        ----------
        model: dict - used for predicting unknown class labels. It contains as keys the rule IDs and as value a named tuple
        indicating the support (= % of covered examples whose labels are predicted correctly) for the minority and majority
        labels respectively
        test_examples: pd.DataFrame - unlabeled examples for which the class labels will be predicted
        rules: dict - rules for the dataset, where rule IDs are keys and the rules (pd.Series) are values
        classes: list of str - list of class labels - only 2 are considered though, namely minority label and rest
        class_col_name: str - name of the column with the class label in <training_examples>
        min_max: pd:DataFrame - contains min/max values per numeric feature
        for_multiclass: bool - True if the binary classification will be used for predicting multiclass labels. In this case
        the label for the minority class will always be returned (because we need to choose the minority label across all
        classes with the highest support, so we don't care about the "rest" label (=majority) at all)

        Returns
        -------
        pd.DataFrame.
        <test_examples> with 2 additional columns storing predicted label and BRACID's confidence using the column names
        PREDICTED_LABEL and PREDICTION_CONFIDENCE respectively.
        If <for_multiclass> == False:
            Confidence = max (support for minority, support for majority) / (support for minority + support for majority)
        Else:
            Confidence = support for minority / (support for minority + support for majority)

        """
        # {example_id: Predictions(...)}
        preds = {}
        # Compute support and predicted label
        supports = self.compute_rule_support_per_example(rules, test_examples,
                                                         model, class_col_name,
                                                         min_max, classes)
        majority_label = classes[0]
        if self._minority_class == classes[0]:
            majority_label = classes[1]
        minority_label = self._minority_class
        # Compute confidence based on support
        for example_id in supports:
            predicted_label = minority_label
            has_minority = True
            if supports[example_id].minority <= supports[example_id].majority:
                predicted_label = majority_label
                has_minority = False
            if has_minority or for_multiclass:
                confidence = supports[example_id].minority / (
                        supports[example_id].minority + supports[
                    example_id].majority)
                predicted_label = minority_label
            else:
                confidence = supports[example_id].majority / (
                        supports[example_id].minority + supports[
                    example_id].majority)
            preds[example_id] = Predictions(label=predicted_label,
                                            confidence=confidence)
        # Store confidence and predicted label in data frame
        test_examples[my_vars.PREDICTED_LABEL] = minority_label
        test_examples[my_vars.PREDICTION_CONFIDENCE] = 0
        for example_id, row in test_examples.iterrows():
            data = preds[example_id]
            test_examples.loc[example_id, [my_vars.PREDICTED_LABEL,
                                           my_vars.PREDICTION_CONFIDENCE]] = data.label, \
                                                                             data.confidence
        return test_examples

    def compute_rule_support_per_example(self, rules, examples, model,
                                         class_col_name, min_max, classes):
        """
        Computes the support of the various rules for each unlabeled example. Even if class labels exist, they are ignored.
        Support computation takes into account that multiple rules might cover an example or that multiple rules are
        closest to an example.

        Parameters
        ----------
        rules: dict - rules for the dataset, where rule IDs are keys and the rules (pd.Series) are values
        examples: pd.DataFrame - unlabeled examples for which the class labels will be predicted
        model: dict - used for predicting unknown class labels. It contains as keys the rule IDs and as value a named tuple
        indicating the support (= % of covered examples whose labels are predicted correctly) for the minority and majority
        labels respectively
        class_col_name: str - name of the column with the class label in <training_examples>
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset

        Returns
        -------
        dict.
        Support per example aggregated over the various rules for majority and minority class:
        {example_id: Support(minority=X, majority=Y)} where X and Y are floats.

        """
        logger.info("model")
        logger.info(model)
        # Turn off a pandas warning about making changes to a copy of the dataFrame when using .apply() below
        pd.options.mode.chained_assignment = None  # default='warn'
        # {example_id: Support(...)}
        supports = {}
        uncovered_example_ids = set(examples.index.values)
        assert (len(uncovered_example_ids) == examples.shape[0])
        for rule_id in rules:
            rule = rules[rule_id]
            examples[my_vars.COVERED] = examples.loc[:, :] \
                .apply(does_rule_cover_example_without_label, axis=1,
                       args=(rule, examples.dtypes, class_col_name))
            all_covered_examples = examples.loc[
                examples[my_vars.COVERED] == True]
            for example_id, example in all_covered_examples.iterrows():
                uncovered_example_ids.discard(example_id)
                # logger.info("rule {} covers example {}".format(rule_id, example_id))
                if example_id not in supports:
                    supports[example_id] = Support(minority=0, majority=0)
                # logger.info("old support", supports[example_id])
                new_minority = supports[example_id].minority + model[
                    rule_id].minority
                new_majority = supports[example_id].majority + model[
                    rule_id].majority
                supports[example_id] = Support(minority=new_minority,
                                               majority=new_majority)
                # logger.info("updated support", supports[example_id])
        # Compute distances for the remaining uncovered examples and take ties into account
        if uncovered_example_ids:
            k = len(uncovered_example_ids)
            # {example_id1: [Data(rule_id, dist, Data(rule_id, dist),...]}
            uncovered_examples = {}
            # {example_id1: set(rule_id1, rule_id5)}
            closest_rule_ids_per_example = {}
            for eid in uncovered_example_ids:
                uncovered_examples[eid] = [Data(rule_id=-1, dist=math.inf)]
                closest_rule_ids_per_example[eid] = set()
            remaining_examples = examples.loc[list(uncovered_example_ids)]

            # find_nearest_rule() updates internal statistics of the actual model which we don't desire, so restore them
            # later
            rule_per_example = copy.deepcopy(self.closest_rule_per_example)
            examples_per_rule = copy.deepcopy(self.closest_examples_per_rule)
            covered_examples = copy.deepcopy(self.examples_covered_by_rule)

            for rule_id in rules:
                rule = rules[rule_id]
                # TODO: write as a class and don't update in find_nearest_rule (or nearest_examples), but return the
                #         # updated entries to decide depending on the scenario whether to update the data or not
                neighbors, dists, is_closest = \
                    self.find_nearest_examples(remaining_examples, k, rule,
                                               class_col_name, min_max,
                                               classes,
                                               label_type=my_vars.ALL_LABELS,
                                               only_uncovered_neighbors=False)
                if neighbors is not None:
                    for example_id, row in dists.iterrows():
                        dist = dists.loc[example_id][my_vars.DIST]
                        if example_id not in uncovered_examples:
                            uncovered_examples[example_id] = [
                                Data(rule_id=rule_id, dist=dist)]
                            closest_dist = math.inf
                        else:
                            closest_dist = uncovered_examples[example_id][
                                0].dist
                        # New closest rule
                        if dist < closest_dist:
                            uncovered_examples[example_id] = [
                                Data(rule_id=rule_id, dist=dist)]
                        # Add tie
                        elif abs(dist - closest_dist) < my_vars.PRECISION:
                            uncovered_examples[example_id].append(
                                Data(rule_id=rule_id, dist=dist))

            # Restore actual model
            self.closest_rule_per_example = rule_per_example
            self.closest_examples_per_rule = examples_per_rule
            self.examples_covered_by_rule = covered_examples
            # Update support for uncovered examples
            for example_id in uncovered_examples:
                if example_id not in supports:
                    supports[example_id] = Support(minority=0, majority=0)
                minority_supp = 0
                majority_supp = 0
                for rule_id, _ in uncovered_examples[example_id]:
                    minority_supp += model[rule_id].minority
                    majority_supp += model[rule_id].majority
                supports[example_id] = Support(minority=minority_supp,
                                               majority=majority_supp)
        return supports

    def init_statistics(self, df):
        """
        Initializes the global variables required in bracid() with default values.

        Parameters
        ----------
        df: pd.DataFrame - dataset.

        """
        self.all_rules = {}
        self.unique_rules = {}
        self.seed_example_rule = {}
        self.seed_rule_example = {}
        self.closest_rule_per_example = {}
        self.closest_examples_per_rule = {}
        self.conf_matrix = ConfusionMatrix()
        self.examples_covered_by_rule = {}
        # Initial rule (with the highest index) will be derived from seed examples, so we already know the maximum ID now
        max_example_id = df.index.max()
        self.latest_rule_id = max_example_id

    @property
    def closest_examples_per_rule(self):
        return self._closest_examples_per_rule

    @closest_examples_per_rule.setter
    def closest_examples_per_rule(self, v):
        if v is None:
            self._closest_examples_per_rule = None
            return
        if not isinstance(v, dict):
            raise ValueError(
                f'closest_examples_per_rule cannot be assigned a non-dict object of type {type(v)}: {v}')
        if self._closest_examples_per_rule is None:
            self._closest_examples_per_rule = defaultdict(set, v)
            return
        assert isinstance(self._closest_examples_per_rule, defaultdict)
        self._closest_examples_per_rule.clear()
        self._closest_examples_per_rule.update(v)

    def extract_rules_and_train_and_predict_multiclass(self, train_set,
                                                       test_set, min_max,
                                                       class_col_name, k):
        """
        Wrapper function that extracts the BRACID rules, trains a model based on these discovered rules, and predicts the
        labels for a multiclass classification task. Converts a multiclass problem into a one-vs-all scheme.

        Parameters
        ----------
        train_set: pd.DataFrame - training set where each row represents a training example
        test_set: pd.DataFrame - test set where each row represents a test example for which the label should be predicted
        min_max: pd:DataFrame - contains min/max values per numeric feature
        class_col_name: str - name of the column with the class label in <training_examples>
        k: int - number of neighbors with opposite label of the current example to consider

        Returns
        -------
        dict of dict of pd.Series, pd.dataFrame.
        Dictionary of rules that classify the training data most accurately according to F1 score. Keys are the class
        labels and values are dicts with the keys being the rule IDs and values the corresponding rules.
        DataFrame contains 2 additional columns in <test_examples>, namely PREDICTED_LABEL and PREDICTION_CONFIDENCE
        containing the predicted label and BRACID's confidence for assigning it.
        Confidence = max (support for minority) / (support for minority + support for majority)
        Final label per example is calculated according to the maximum confidence.
        For example, if there are 3 classes A, B, C, and we have an instance e for which to predict the label,
        we choose the label according to:
        max_{i in |{A, B, C}|} sup(K_i, e)
        , where sup() is defined in the paper.

        """
        minority_labels = train_set[class_col_name].unique()
        logger.info("one-vs-all labels in the order they'll be processed",
                    minority_labels)
        res = test_set.copy()
        # Negative confidence because even if confidence for the first label is 0, it should be used instead of no label
        res[my_vars.PREDICTION_CONFIDENCE] = -1
        res[my_vars.PREDICTED_LABEL] = ""
        all_rules = {}
        for minority_label in minority_labels:
            classes = [minority_label, Labels.REST]
            # Convert to a binary problem
            train = train_set.copy()
            test = test_set.copy()
            train = to_binary_classification_task(train, class_col_name,
                                                  minority_label,
                                                  merged_label=Labels.REST)
            logger.info("####training set#####")
            logger.info(train)
            logger.info("classes to be used in current run",
                        train[class_col_name].unique())
            logger.info("####test set######")
            logger.info(test)
            rules = self.bracid(train, k, class_col_name, min_max, classes,
                                minority_label)
            model = self.train_binary(rules, train, minority_label,
                                      class_col_name)
            preds_df = self.predict_binary(model, test, rules, classes,
                                           class_col_name, min_max,
                                           for_multiclass=True)
            all_rules[minority_label] = rules
            # Update predicted label and confidence if confidence is higher than the currently best confidence
            res.loc[((res[my_vars.PREDICTION_CONFIDENCE] < preds_df[
                my_vars.PREDICTION_CONFIDENCE]) &
                     (preds_df[my_vars.PREDICTED_LABEL] != Labels.REST)),
                    my_vars.PREDICTED_LABEL] = preds_df[
                my_vars.PREDICTED_LABEL]
            res.loc[((res[my_vars.PREDICTION_CONFIDENCE] < preds_df[
                my_vars.PREDICTION_CONFIDENCE]) &
                     (preds_df[my_vars.PREDICTED_LABEL] != Labels.REST)),
                    my_vars.PREDICTION_CONFIDENCE] = preds_df[
                my_vars.PREDICTION_CONFIDENCE]
            # logger.info("predicted when using {} as class".format(minority_label))
            # logger.info(res[res.columns[-3:]])
            # logger.info("bla")
        return all_rules, res

    def extract_rules_and_train_and_predict_binary(self, train_set, test_set,
                                                   min_max, classes,
                                                   minority_label,
                                                   class_col_name, k):
        """
        Wrapper function that extracts the BRACID rules, trains a model based on these discovered rules, and predicts the
        labels for a binary classification task.

        Parameters
        ----------
        train_set: pd.DataFrame - training set where each row represents a training example
        test_set: pd.DataFrame - test set where each row represents a test example for which the label should be predicted
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset. It's assumed to be binary.
        minority_label: str - label of the minority class
        class_col_name: str - name of the column with the class label in <training_examples>
        k: int - number of neighbors with opposite label of the current example to consider

        Returns
        -------
        dict of pd.Series, pd.dataFrame.
        Dictionary of rules that classify the training data most accurately according to F1 score. Keys are the rule IDs and
        values the corresponding rules.
        DataFrame contains 2 additional columns in <test_examples>, namely PREDICTED_LABEL and PREDICTION_CONFIDENCE
        containing the predicted label and BRACID's confidence for assigning it.
        Confidence = max (support for minority, support for majority) / (support for minority + support for majority)

        """
        rules = self.bracid(train_set, k, class_col_name, min_max, classes,
                            minority_label)
        model = self.train_binary(rules, train_set, minority_label,
                                  class_col_name)
        preds_df = self.predict_binary(model, test_set, rules, classes,
                                       class_col_name, min_max)
        return rules, preds_df

    def cv_binary(self, dataset, k, class_col_name, min_max, classes,
                  minority_label, folds=10, seed=13):
        """
        Performs cross-validation on a given binary dataset.

        Parameters
        ----------
        dataset: pd.DataFrame - dataset
        k: int - number of neighbors with opposite label of <rule> to consider
        class_col_name: str - name of class label
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset. It's assumed to be binary
        minority_label: str - class label of the minority class. Note that all other labels are grouped into another class
        so that there's a binary classification task.
        folds: int - number of folds in cross-validation
        seed: int - seed for PRNG for reproduction of the results

        Returns
        -------
        float, np.ndarray of shape (n_classes,).
        Micro-averaged F1 score, class-wise F1 score.

        """
        df = dataset.copy()
        # Shuffle dataset
        df = df.sample(frac=1, random_state=seed)
        examples = df.shape[0]
        examples_per_fold = math.ceil(examples / folds)
        logger.info("pick {} examples per fold".format(examples_per_fold))

        predicted = []
        true = []
        # Create folds for CV
        for i in range(folds):
            logger.info("fold", i + 1)
            test_set = df.iloc[
                       i * examples_per_fold: i * examples_per_fold + examples_per_fold]
            # logger.info("test set: {}".format(test_set.shape))
            # logger.info(test_set)
            train_set = df.drop(df.index[
                                i * examples_per_fold: i * examples_per_fold + examples_per_fold])
            # logger.info("training set: {}".format(train_set.shape))
            # logger.info(train_set)
            _, preds_df = self.extract_rules_and_train_and_predict_binary(
                train_set, test_set, min_max, classes,
                minority_label, class_col_name, k)
            predicted.extend(preds_df[my_vars.PREDICTED_LABEL].values)
            true.extend(preds_df[class_col_name].values)
        micro_f1 = sklearn.metrics.f1_score(true, predicted, labels=classes,
                                            average="micro")
        classwise_f1 = sklearn.metrics.f1_score(true, predicted,
                                                labels=classes, average=None)
        # logger.info("order of classes", classes)
        # logger.info("class-wise F1-scores", classwise_f1)
        # logger.info("micro-averaged F1-score:", micro_f1)
        return micro_f1, classwise_f1

    def cv_multiclass(self, dataset, k, class_col_name, min_max, classes,
                      folds=10, seed=13):
        """
        Performs cross-validation on a given dataset, but handles multiclass
        problems using the one-vs-all scheme, i.e. if there are m classes in the dataset, m classifiers are trained
        distinguishing a class from the rest. The final label is assigned to an instance based on the highest support.
        For example, if there are 3 classes A, B, C, and we have an instance e for which to predict the label,
        we choose the label according to:
        max_{i in |{A, B, C}|} sup(K_i, e)
        , where sup() is defined in the paper.

        Parameters
        ----------
        dataset: pd.DataFrame - dataset
        k: int - number of neighbors with opposite label of <rule> to consider
        class_col_name: str - name of class label
        min_max: pd:DataFrame - contains min/max values per numeric feature
        classes: list of str - class labels in the dataset.
        folds: int - number of folds in cross-validation
        seed: int - seed for PRNG for reproduction of the results

        Returns
        -------
        float, np.ndarray of shape (n_classes,), np.ndarray of shape (folds, n_classes), np.ndarray of shape (folds,
        n_classes)
        Micro-averaged F1 score, class-wise F1 score, true labels of the test set per fold, predicted labels of the test
        set per fold

        """
        df = dataset.copy()
        # Shuffle dataset
        df = df.sample(frac=1, random_state=seed)
        examples = df.shape[0]
        examples_per_fold = math.ceil(examples / folds)
        logger.info("pick {} examples per fold".format(examples_per_fold))

        predicted_total = []
        true_total = []
        predicted_foldwise = []
        true_foldwise = []
        # Create folds for CV
        for i in range(folds):
            test_set = df.iloc[
                       i * examples_per_fold: i * examples_per_fold + examples_per_fold]
            train_set = df.drop(df.index[
                                i * examples_per_fold: i * examples_per_fold + examples_per_fold])
            _, preds_df = self.extract_rules_and_train_and_predict_multiclass(
                train_set, test_set, min_max, class_col_name, k)
            preds = preds_df[my_vars.PREDICTED_LABEL].values
            true = preds_df[class_col_name].values
            # logger.info("true labels:", true)
            # logger.info("predicted labels:", preds)
            predicted_total.extend(preds)
            predicted_foldwise.append(preds)
            true_total.extend(true)
            true_foldwise.append(true)
        micro_f1 = f1_score(true_total, predicted_total, labels=classes,
                            average="micro")
        classwise_f1 = f1_score(true_total, predicted_total, labels=classes,
                                average=None)
        return micro_f1, classwise_f1, np.array(true_foldwise), np.array(
            predicted_foldwise)
