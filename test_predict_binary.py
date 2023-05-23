from unittest import TestCase
from collections import Counter

import pandas as pd
import numpy as np

from scripts.bracid import BRACID, Bounds, Support, Predictions
import scripts.vars as my_vars


class TestPredict(TestCase):
    """Tests predict() from utils.py"""

    def test_predict_covered(self):
        """Predict the class labels of covered examples"""
        test_set = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                 "C": [3, 2, 1, .5, 3, 2],
                                 "Class": ["", "", "", "", "", ""]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        bracid = BRACID()
        classes = ["apple", "banana"]
        class_col_name = "Class"
        bracid.minority_class = classes[0]
        rules = {
            2: pd.Series({"B": Bounds(lower=1.25, upper=4.0), "C": Bounds(lower=0.5, upper=1.5),
                          "Class": "banana"}, name=2),
            6: pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": "banana"}, name=6),
            5: pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5),
                          "Class": "banana"}, name=5),
            0: pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": "apple"}, name=0),
        }
        model = {2: Support(minority=0.75, majority=0.25), 6: Support(minority=0.2, majority=0.8),
                 5: Support(minority=1.0, majority=0.0), 0: Support(minority=0, majority=1)}

        # Last 2 parameters aren't be used in this test
        df = bracid.predict_binary(model, test_set, rules, classes, class_col_name, None, None, for_multiclass=False)
        correct = pd.DataFrame(
            {
                my_vars.PREDICTED_LABEL:["banana", "banana", "apple", "banana", "banana", "apple"],
                my_vars.PREDICTION_CONFIDENCE: [0.9, 0.9, 0.875, 0.683333, 0.9, 1]
            })
        np.testing.assert_array_equal(correct[my_vars.PREDICTED_LABEL].values, df[my_vars.PREDICTED_LABEL].values)
        np.testing.assert_allclose(correct[my_vars.PREDICTION_CONFIDENCE], df[my_vars.PREDICTION_CONFIDENCE], atol=my_vars.PRECISION)

    def test_predict_uncovered(self):
        """Predict the class labels of uncovered examples with handling ties (2 rules are equally distant) for
        example 4, namely rules 0 and 6"""
        # Assumptions: these are the data for the training set NOT for the test set
        bracid = BRACID()
        lookup = \
            {
                "A":
                    {
                        'high': 2,
                        'low': 4,
                        my_vars.CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        'banana': 2
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 2
                                    })
                            }
                    }
            }
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        classes = ["apple", "banana"]
        test_set = pd.DataFrame({"A": ["low", "high", "high", "low", "low", "high"],
                                 "B": [4.1, 6.1, 5.4, 0.15, 0.05, 0.075],
                                 "C": [0.3, 4, 0.1, .4, 0.3, 5],
                                 "Class": ["", "", "", "", "", ""]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        class_col_name = "Class"
        bracid.minority_class = classes[0]
        bracid.examples_covered_by_rule = {}
        bracid.closest_examples_per_rule = {}
        bracid.closest_rule_per_example = {}
        rules = {
            2: pd.Series({"B": Bounds(lower=1.25, upper=4.0), "C": Bounds(lower=0.5, upper=1.5),
                          "Class": "banana"}, name=2),
            6: pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": "banana"}, name=6),
            5: pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5),
                          "Class": "banana"}, name=5),
            0: pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": "apple"}, name=0),
        }
        bracid.all_rules = rules
        model = {2: Support(minority=0.75, majority=0.25), 6: Support(minority=0.2, majority=0.8),
                 5: Support(minority=1.0, majority=0.0), 0: Support(minority=0, majority=1)}
        correct_covered = {}
        correct_examples_per_rule = {}
        correct_rule_per_example = {}

        df = bracid.predict_binary(model, test_set, rules, classes, class_col_name, lookup, min_max, for_multiclass=False)
        correct = pd.DataFrame(
            {
                my_vars.PREDICTED_LABEL: ["apple", "apple", "apple", "banana", "banana", "apple"],
                my_vars.PREDICTION_CONFIDENCE: [0.75, 1, 0.75, 0.9, 0.9, 1]
            })

        # Test that predictions didn't change internal statistics of the model
        self.assertEqual(correct_covered, bracid.examples_covered_by_rule)
        self.assertEqual(correct_examples_per_rule, bracid.closest_examples_per_rule)
        self.assertEqual(correct_rule_per_example, bracid.closest_rule_per_example)

        np.testing.assert_array_equal(correct[my_vars.PREDICTED_LABEL].values, df[my_vars.PREDICTED_LABEL].values)
        np.testing.assert_allclose(correct[my_vars.PREDICTION_CONFIDENCE], df[my_vars.PREDICTION_CONFIDENCE])

    def test_predict_mixed(self):
        """Predict the class labels of uncovered and covered examples while handling ties"""
        # Assumptions: these are the data for the training set NOT for the test set
        bracid = BRACID()
        lookup = \
            {
                "A":
                    {
                        'high': 2,
                        'low': 4,
                        my_vars.CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        'banana': 2
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 2
                                    })
                            }
                    }
            }
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        classes = ["apple", "banana"]
        test_set = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"],
                                 "B": [4.1, 1, 5.4, 0.15, 0.05, 0.075],
                                 "C": [0.3, 2, 0.1, .4, 0.3, 5],
                                 "Class": ["", "", "", "", "", ""]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        class_col_name = "Class"
        bracid.minority_class = classes[0]
        bracid.examples_covered_by_rule = {}
        bracid.closest_examples_per_rule = {}
        bracid.closest_rule_per_example = {}
        rules = {
            2: pd.Series({"B": Bounds(lower=1.25, upper=4.0), "C": Bounds(lower=0.5, upper=1.5),
                          "Class": "banana"}, name=2),
            6: pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": "banana"}, name=6),
            5: pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5),
                          "Class": "banana"}, name=5),
            0: pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": "apple"}, name=0),
        }
        bracid.all_rules = rules
        model = {2: Support(minority=0.75, majority=0.25), 6: Support(minority=0.2, majority=0.8),
                 5: Support(minority=1.0, majority=0.0), 0: Support(minority=0, majority=1)}
        correct_covered = {}
        correct_examples_per_rule = {}
        correct_rule_per_example = {}

        correct_predictions = {0: Predictions(label='apple', confidence=0.75),
                               1: Predictions(label='banana', confidence=0.9),  # Covered by rules 0 and 6
                               2: Predictions(label='apple', confidence=0.75),
                               3: Predictions(label='banana', confidence=0.9),
                               4: Predictions(label='banana', confidence=0.9),
                               5: Predictions(label='apple', confidence=1.0)}

        df = bracid.predict_binary(model, test_set, rules, classes, class_col_name, lookup, min_max,
                                   for_multiclass=False)
        correct = pd.DataFrame(
            {
                my_vars.PREDICTED_LABEL: ["apple", "banana", "apple", "banana", "banana", "apple"],
                my_vars.PREDICTION_CONFIDENCE: [0.75, 0.9, 0.75, 0.9, 0.9, 1]
            })

        # Test that predictions didn't change internal statistics of the model
        self.assertEqual(correct_covered, bracid.examples_covered_by_rule)
        self.assertEqual(correct_examples_per_rule, bracid.closest_examples_per_rule)
        self.assertEqual(correct_rule_per_example, bracid.closest_rule_per_example)

        np.testing.assert_array_equal(correct[my_vars.PREDICTED_LABEL].values, df[my_vars.PREDICTED_LABEL].values)
        np.testing.assert_allclose(correct[my_vars.PREDICTION_CONFIDENCE], df[my_vars.PREDICTION_CONFIDENCE])
