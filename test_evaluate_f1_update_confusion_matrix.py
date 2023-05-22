from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.bracid import BRACID, Data, Bounds
import scripts.vars as my_vars


class TestEvaluateF1UpdateConfusionMatrix(TestCase):
    """Tests evaluate_f1_update_confusion_matrix() in utils.py"""

    def test_evaluate_f1_update_confusion_matrix_updated(self):
        """Tests what happens if input has a numeric and a nominal feature and a rule that predicts an example is
        updated"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        class_col_name = "Class"
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
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        bracid.minority_class = "apple"

        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5)
        ]
        bracid.closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625)}
        bracid.examples_covered_by_rule = {}
        bracid.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: {3, 4}, my_vars.TN: {2, 5}, my_vars.FN: set()}
        new_rule = pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.0), "C": Bounds(lower=3, upper=3),
                              "Class": "banana"}, name=0)
        # tagged, initial_rules = add_tags_and_extract_rules(df, 2, class_col_name, lookup, min_max, classes)
        correct_f1 = 0.8
        f1 = bracid.evaluate_f1_update_confusion_matrix(df, new_rule, class_col_name, lookup, min_max, classes)
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.0),
            5: Data(rule_id=2, dist=0.67015625)}
        print(f1, correct_f1)
        self.assertTrue(f1 == correct_f1)
        for example_id in bracid.closest_rule_per_example:
            rule_id, dist = bracid.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)
        correct_conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: {3}, my_vars.TN: {2, 4, 5}, my_vars.FN: set()}
        self.assertTrue(bracid.conf_matrix == correct_conf_matrix)

    def test_evaluate_f1_update_confusion_matrix_not_updated(self):
        """Tests what happens if input has a numeric and a nominal feature and a rule that predicts an example is
        not updated as F1 score doesn't improve"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        class_col_name = "Class"
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
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        bracid.minority_class = "apple"

        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5)
        ]
        bracid.examples_covered_by_rule = {}
        bracid.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625)}
        bracid.conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        new_rule = pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4)
        correct_f1 = 2*1*0.5/1.5

        f1 = bracid.evaluate_f1_update_confusion_matrix(df, new_rule, class_col_name, lookup, min_max, classes)
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625)}
        self.assertTrue(f1 == correct_f1)
        for example_id in bracid.closest_rule_per_example:
            rule_id, dist = bracid.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)
        correct_conf_matrix = {my_vars.TP: {0, 1}, my_vars.FP: set(), my_vars.TN: {2, 5}, my_vars.FN: {3, 4}}
        self.assertTrue(bracid.conf_matrix == correct_conf_matrix)
