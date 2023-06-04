from unittest import TestCase
import copy

import pandas as pd
import pytest

from multi_imbalance.classifiers.bracid.bracid import BRACID, Data, Bounds, ConfusionMatrix
from tests.classifiers.bracid.classes_ import _0, _1
from tests.classifiers.bracid.assertions import assert_almost_equal


class TestEvaluateF1Temporarily(TestCase):
    """Tests evaluate_f1_temporarily() in utils.py"""

    @pytest.mark.skip(reason="TODO: fix this test")
    def test_evaluate_f1_temporarily(self):
        """Tests that the global variables won't be updated despite local changes"""
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": [_0, _0, _1, _1, _1, _1]})
        bracid = BRACID()
        class_col_name = "Class"
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        bracid.minority_class = _0
        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0}, name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0}, name=1),
            pd.Series({"B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1), "Class": _1}, name=2),
            pd.Series({"B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5), "Class": _1}, name=3),
            pd.Series({"B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3), "Class": _1}, name=4),
            pd.Series({"B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2), "Class": _1}, name=5),
        ]

        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # bracid.seed_rule_example = {i: {i} for i, _ in enumerate(rules)}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        # bracid.seed_example_rule = {i: {i} for i, _ in enumerate(rules)}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        # bracid.all_rules = {i: rule for i, rule in enumerate(rules)}

        bracid.closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625),
        }
        bracid.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
        bracid.closest_examples_per_rule = {
            0: {1, 2, 4, 5},
            1: {0, 3},
        }
        bracid.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
        correct_closest_rules = copy.deepcopy(bracid.closest_rule_per_example)
        correct_closest_examples = copy.deepcopy(bracid.closest_examples_per_rule)
        bracid.conf_matrix = ConfusionMatrix(TP={0, 1}, FP={3, 4}, TN={2, 5}, FN=set())
        new_rule = pd.Series({"B": Bounds(lower=0.5, upper=1.0), "C": Bounds(lower=3, upper=3), "Class": _1}, name=0)
        correct_f1 = 0.8

        f1, conf_matrix, closest_rules, closest_examples, covered, updated_example_ids = bracid.evaluate_f1_temporarily(
            df, new_rule, new_rule.name, class_col_name, min_max, classes
        )
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.0),
            5: Data(rule_id=2, dist=0.67015625),
        }
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=0, dist=0.6025),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.0),
            5: Data(rule_id=0, dist=0.010000000000000002),
        }
        correct_covered = {0: {4}}
        correct_updated_examples = [2, 4, 5]
        self.assertListEqual(updated_example_ids, correct_updated_examples)
        self.assertEqual(f1, correct_f1)
        assert_almost_equal(self, closest_rules, correct_closest_rule_per_example)
        self.assertDictEqual(closest_examples, bracid.closest_examples_per_rule)
        correct_conf_matrix = ConfusionMatrix(TP={0, 1}, FP={3}, TN={2, 4, 5}, FN=set())
        self.assertEqual(conf_matrix, correct_conf_matrix)
        # But now check that global variables remained unaffected by the changes
        correct_conf_matrix = ConfusionMatrix(TP={0, 1}, FP={3, 4}, TN={2, 5}, FN=set())
        self.assertEqual(bracid.conf_matrix, correct_conf_matrix)
        self.assertEqual(correct_closest_rules, bracid.closest_rule_per_example)
        self.assertEqual(correct_closest_examples, bracid.closest_examples_per_rule)
        self.assertEqual(correct_covered, covered)
