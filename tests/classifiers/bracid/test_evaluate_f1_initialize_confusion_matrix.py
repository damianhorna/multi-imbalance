from unittest import TestCase
from collections import Counter

import pandas as pd

from multi_imbalance.resampling.bracid.bracid import BRACID, Bounds, ConfusionMatrix, Data
from tests.resampling.bracid.classes_ import _0, _1
from tests.resampling.bracid.assertions import assert_almost_equal


class TestEvaluateF1InitializeConfusionMatrix(TestCase):
    """Tests evaluate_f1_initialize_confusion_matrix() in utils.py"""

    def test_evaluate_f1_initialize_confusion_matrix(self):
        """Tests what happens if input has a numeric and a nominal feature"""
        bracid = BRACID()
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        classes = [_0, _1]
        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0},
                      name=1),
            pd.Series({"B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": _1}, name=2),
            pd.Series({"B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": _1}, name=3),
            pd.Series({"B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": _1}, name=4),
            pd.Series({"B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": _1}, name=5)
        ]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        bracid.minority_class = _0
        bracid.all_rules = {i: rule for i, rule in enumerate(rules)}

        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        bracid.examples_covered_by_rule = {}

        f1 = bracid.evaluate_f1_initialize_confusion_matrix(df, rules, class_col_name, min_max, classes)
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=5, dist=0.00390625),
            2: Data(rule_id=3, dist=0.393125),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=5, dist=0.013906250000000002),
            5: Data(rule_id=1, dist=0.00390625)}
        correct_closest_examples_per_rule = {1: {0, 3, 5}, 5: {1, 4}, 3: {2}}
        correct_conf_matrix = ConfusionMatrix(TP={0}, TN={2, 4}, FP={3, 5}, FN={1})
        correct_f1 = correct_conf_matrix.f1
        self.assertEqual(f1, correct_f1)
        self.assertDictEqual(correct_closest_examples_per_rule, bracid.closest_examples_per_rule)
        assert_almost_equal(self, bracid.closest_rule_per_example, correct_closest_rule_per_example)
        self.assertEqual(bracid.conf_matrix, correct_conf_matrix)
