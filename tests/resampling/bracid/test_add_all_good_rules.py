from unittest import TestCase
from collections import Counter, deque

import pandas as pd

# from scripts.utils import add_all_good_rules, find_nearest_examples, evaluate_f1_initialize_confusion_matrix, Data, \
#     Bounds, compute_hashable_key

from multi_imbalance.resampling.bracid.bracid import BRACID, Bounds, Data, ConfusionMatrix
import multi_imbalance.resampling.bracid.vars as my_vars
from tests.resampling.bracid.classes_ import _0, _1

class TestAddAllGoodRules(TestCase):
    """Tests add_all_good_rules() in utils.py"""

    def test_add_all_good_rules(self):
        """Tests that rule set is updated when a generalized rule improves F1"""
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        bracid = BRACID()
        class_col_name = "Class"
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        bracid.minority_class = _1
        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0},
                      name=1),
            pd.Series({"B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": _1}, name=3),
            pd.Series({"B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": _1}, name=4),
            pd.Series({"B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": _1}, name=5),
            pd.Series({"B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": _1}, name=2)  # Current rule to be tested is always at the end
        ]
        test_idx = -1
        bracid.latest_rule_id = len(rules) - 1
        bracid.examples_covered_by_rule = {}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[test_idx], 3: rules[2], 4: rules[3], 5: rules[4]}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        bracid.unique_rules = {}
        for rule in rules:
            hash_val = bracid.compute_hashable_key(rule)
            bracid.unique_rules.setdefault(hash_val, set()).add(rule.name)

        initial_correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=5, dist=0.00390625),
            2: Data(rule_id=3, dist=0.393125),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=5, dist=0.013906250000000002),
            5: Data(rule_id=1, dist=0.00390625)}

        initial_f1 = bracid.evaluate_f1_initialize_confusion_matrix(df, rules, class_col_name, min_max, classes)
        correct_confusion_matrix = ConfusionMatrix(TP={2, 4}, TN={0}, FP={1}, FN={3, 5})
        correct_rules = 7
        self.assertEqual(bracid.conf_matrix, correct_confusion_matrix)

        def assert_almost_equal(bracid_closest_rule_per_example, initial_correct_rules):
            # Make sure confusion matrix, closest rule per example are correct at the beginning
            for example_id in bracid_closest_rule_per_example:
                rule_id, dist = bracid_closest_rule_per_example[example_id]
                with self.subTest(f'{example_id}: rule_id'):
                    self.assertEqual(rule_id, initial_correct_rules[example_id].rule_id)
                with self.subTest(f'{example_id}: dist'):
                    self.assertAlmostEqual(dist, initial_correct_rules[example_id].dist, delta=0.001)

        assert_almost_equal(bracid.closest_rule_per_example, initial_correct_closest_rule_per_example)

        # bracid_closest_rule_per_example = {example_id: Data(*bracid.closest_rule_per_example[example_id]) for example_id in bracid.closest_rule_per_example}
        # self.assertDictEqual(initial_correct_closest_rule_per_example, bracid_closest_rule_per_example)

        correct_initial_f1 = correct_confusion_matrix.f1
        self.assertAlmostEqual(initial_f1, correct_initial_f1)
        k = 3
        neighbors, dists, _ = bracid.find_nearest_examples(df, k, rules[test_idx], class_col_name, min_max, classes,
                                                    label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=
                                                    True)
        improved, updated_rules, f1 = bracid.add_all_good_rules(df, neighbors, rules[test_idx], rules, initial_f1,
                                                         class_col_name, min_max, classes)
        self.assertTrue(improved)
        print("f1", f1)
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=6, dist=0.0), 2: Data(rule_id=6, dist=0.0),
            3: Data(rule_id=2, dist=0.0),
            4: Data(rule_id=5, dist=0.013906250000000002),
            5: Data(rule_id=6, dist=0.0)}
        correct_confusion_matrix = ConfusionMatrix(TP={2, 3, 4, 5}, TN={0}, FP={1}, FN=set())
        correct_covered = {2: {3}, 6: {1, 2, 5}}
        correct_f1 = 0.888
        self.assertAlmostEqual(correct_f1, f1, delta=0.001)
        assert_almost_equal(
            bracid.closest_rule_per_example,
            correct_closest_rule_per_example)

        self.assertEqual(bracid.conf_matrix, correct_confusion_matrix)
        self.assertEqual(len(updated_rules), correct_rules)
        self.assertEqual(bracid.latest_rule_id, correct_rules - 1)
        self.assertDictEqual(correct_covered, bracid.examples_covered_by_rule)
