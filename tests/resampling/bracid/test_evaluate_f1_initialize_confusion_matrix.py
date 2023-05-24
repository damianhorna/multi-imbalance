from unittest import TestCase
from collections import Counter

import pandas as pd

# from scripts.utils import evaluate_f1_initialize_confusion_matrix, Bounds
from multi_imbalance.resampling.bracid.bracid import BRACID, Bounds, ConfusionMatrix
import multi_imbalance.resampling.bracid.vars as my_vars
from tests.resampling.bracid.classes_ import _0, _1


class TestEvaluateF1InitializeConfusionMatrix(TestCase):
    """Tests evaluate_f1_initialize_confusion_matrix() in utils.py"""

    def test_evaluate_f1_initialize_confusion_matrix(self):
        """Tests what happens if input has a numeric and a nominal feature"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
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
                                        _1: 2
                                    }),
                                'low':
                                    Counter({
                                        _1: 2,
                                        _0: 2
                                    })
                            }
                    }
            }
        classes = [_0, _1]
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": _1}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": _1}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": _1}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": _1}, name=5)
        ]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        bracid.minority_class = _0
        # Reset as other tests changed the data
        bracid.closest_rule_per_example = {}
        bracid.closest_examples_per_rule = {}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}

        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        bracid.examples_covered_by_rule = {}

        # tagged, initial_rules = add_tags_and_extract_rules(df, 2, class_col_name, min_max, classes)
        correct_f1 = 2*1*0.5/1.5
        f1 = bracid.evaluate_f1_initialize_confusion_matrix(df, rules, class_col_name, min_max, classes)
        correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.015625),
            5: (2, 0.67015625)}
        correct_closest_examples_per_rule = {1: {0, 3}, 0: {1, 4}, 5: {2}, 2: {5}}
        # correct_conf_matrix = {'tp': {0, 1}, 'fp': {3, 4}, 'tn': {2, 5}, 'fn': set()}
        correct_conf_matrix = ConfusionMatrix(TP={0, 1}, FP={3, 4}, TN={2, 5}, FN=set())
        self.assertEqual(f1, correct_f1)
        self.assertEqual(correct_closest_examples_per_rule, bracid.closest_examples_per_rule)
        for example_id in bracid.closest_rule_per_example:
            rule_id, dist = bracid.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id][0] and
                            abs(dist - correct_closest_rule_per_example[example_id][1]) < 0.001)
        self.assertEqual(bracid.conf_matrix, correct_conf_matrix)
