from unittest import TestCase

import pandas as pd

from multi_imbalance.classifiers.bracid.bracid import BRACID, Bounds, Support
import multi_imbalance.classifiers.bracid.vars as my_vars
from tests.classifiers.bracid.classes_ import _0, _1


class TestPredict(TestCase):
    """Tests predict() from utils.py"""

    def test_predict_covered(self):
        """Predict the class labels of covered examples"""
        test_set = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": ["", "", "", "", "", ""]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        classes = [_0, _1]
        bracid = BRACID(k=-1, minority_class=classes[0])
        class_col_name = "Class"
        rules = {
            2: pd.Series({"B": Bounds(lower=1.25, upper=4.0), "C": Bounds(lower=0.5, upper=1.5), "Class": _1}, name=2),
            6: pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0), "Class": _1}, name=6),
            5: pd.Series({"B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5), "Class": _1}, name=5),
            0: pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0), "Class": _0}, name=0),
        }
        model = {
            2: Support(minority=0.75, majority=0.25),
            6: Support(minority=0.2, majority=0.8),
            5: Support(minority=1.0, majority=0.0),
            0: Support(minority=0, majority=1),
        }

        # Last 2 parameters aren't be used in this test
        df = bracid.predict_binary(model, test_set, rules, classes, class_col_name, None, for_multiclass=False)
        correct = pd.DataFrame(
            {my_vars.PREDICTED_LABEL: [_1, _1, _0, _1, _1, _1], my_vars.PREDICTION_CONFIDENCE: [0.9, 0.6, 0.875, 0.683333, 0.9, 0.6]}
        )
        pd.testing.assert_series_equal(correct[my_vars.PREDICTED_LABEL], df[my_vars.PREDICTED_LABEL])
        pd.testing.assert_series_equal(correct[my_vars.PREDICTION_CONFIDENCE], df[my_vars.PREDICTION_CONFIDENCE], check_less_precise=4)

    def test_predict_uncovered(self):
        """Predict the class labels of uncovered examples with handling ties (2 rules are equally distant) for
        example 4, namely rules 0 and 6"""
        # Assumptions: these are the data for the training set NOT for the test set
        classes = [_0, _1]
        bracid = BRACID(k=-1, minority_class=classes[0])
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        test_set = pd.DataFrame(
            {
                "A": ["low", "high", "high", "low", "low", "high"],
                "B": [4.1, 6.1, 5.4, 0.15, 0.05, 0.075],
                "C": [0.3, 4, 0.1, 0.4, 0.3, 5],
                "Class": ["", "", "", "", "", ""],
            }
        )
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        class_col_name = "Class"
        rules = {
            2: pd.Series({"B": Bounds(lower=1.25, upper=4.0), "C": Bounds(lower=0.5, upper=1.5), "Class": _1}, name=2),
            6: pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0), "Class": _1}, name=6),
            5: pd.Series({"B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5), "Class": _1}, name=5),
            0: pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0), "Class": _0}, name=0),
        }
        bracid.all_rules = rules
        model = {
            2: Support(minority=0.75, majority=0.25),
            6: Support(minority=0.2, majority=0.8),
            5: Support(minority=1.0, majority=0.0),
            0: Support(minority=0, majority=1),
        }
        correct_covered = {}
        correct_examples_per_rule = {}
        correct_rule_per_example = {}

        df = bracid.predict_binary(model, test_set, rules, classes, class_col_name, min_max, for_multiclass=False)
        correct = pd.DataFrame(
            {my_vars.PREDICTED_LABEL: [_0, _0, _0, _1, _1, _1], my_vars.PREDICTION_CONFIDENCE: [0.75, 1, 0.75, 0.9, 0.9, 0.9]}
        )

        # Test that predictions didn't change internal statistics of the model
        self.assertDictEqual(correct_covered, bracid.examples_covered_by_rule)
        self.assertDictEqual(correct_examples_per_rule, bracid.closest_examples_per_rule)
        self.assertDictEqual(correct_rule_per_example, bracid.closest_rule_per_example)

        pd.testing.assert_series_equal(correct[my_vars.PREDICTED_LABEL], df[my_vars.PREDICTED_LABEL])
        pd.testing.assert_series_equal(correct[my_vars.PREDICTION_CONFIDENCE], df[my_vars.PREDICTION_CONFIDENCE], check_less_precise=4)

    def test_predict_mixed(self):
        """Predict the class labels of uncovered and covered examples while handling ties"""
        # Assumptions: these are the data for the training set NOT for the test set
        classes = [_0, _1]
        bracid = BRACID(k=-1, minority_class=classes[0])
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        test_set = pd.DataFrame(
            {
                "A": ["low", "low", "high", "low", "low", "high"],
                "B": [4.1, 1, 5.4, 0.15, 0.05, 0.075],
                "C": [0.3, 2, 0.1, 0.4, 0.3, 5],
                "Class": ["", "", "", "", "", ""],
            }
        )
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        class_col_name = "Class"
        rules = {
            2: pd.Series({"B": Bounds(lower=1.25, upper=4.0), "C": Bounds(lower=0.5, upper=1.5), "Class": _1}, name=2),
            6: pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0), "Class": _1}, name=6),
            5: pd.Series({"B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5), "Class": _1}, name=5),
            0: pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0), "Class": _0}, name=0),
        }
        bracid.all_rules = rules
        model = {
            2: Support(minority=0.75, majority=0.25),
            6: Support(minority=0.2, majority=0.8),
            5: Support(minority=1.0, majority=0.0),
            0: Support(minority=0, majority=1),
        }
        correct_covered = {}
        correct_examples_per_rule = {}
        correct_rule_per_example = {}

        df = bracid.predict_binary(model, test_set, rules, classes, class_col_name, min_max, for_multiclass=False)
        correct = pd.DataFrame(
            {my_vars.PREDICTED_LABEL: [_0, _1, _0, _1, _1, _1], my_vars.PREDICTION_CONFIDENCE: [0.75, 0.6, 0.75, 0.9, 0.9, 0.9]}
        )

        # Test that predictions didn't change internal statistics of the model
        self.assertDictEqual(correct_covered, bracid.examples_covered_by_rule)
        self.assertDictEqual(correct_examples_per_rule, bracid.closest_examples_per_rule)
        self.assertDictEqual(correct_rule_per_example, bracid.closest_rule_per_example)

        pd.testing.assert_series_equal(correct[my_vars.PREDICTED_LABEL], df[my_vars.PREDICTED_LABEL])
        pd.testing.assert_series_equal(correct[my_vars.PREDICTION_CONFIDENCE], df[my_vars.PREDICTION_CONFIDENCE], check_less_precise=4)
