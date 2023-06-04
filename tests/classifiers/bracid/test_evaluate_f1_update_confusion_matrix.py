from unittest import TestCase

import pandas as pd

from multi_imbalance.classifiers.bracid.bracid import BRACID, Data, Bounds, ConfusionMatrix
from tests.classifiers.bracid.classes_ import _0, _1
from tests.classifiers.bracid.assertions import assert_almost_equal


class TestEvaluateF1UpdateConfusionMatrix(TestCase):
    """Tests evaluate_f1_update_confusion_matrix() in utils.py"""

    def test_evaluate_f1_update_confusion_matrix_updated(self):
        """Tests what happens if input has a numeric and a nominal feature and a rule that predicts an example is
        updated"""
        bracid = BRACID(k=-1, minority_class=_0)
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0}, name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0}, name=1),
            pd.Series({"B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1), "Class": _1}, name=2),
            pd.Series({"B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5), "Class": _1}, name=3),
            pd.Series({"B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3), "Class": _1}, name=4),
            pd.Series({"B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2), "Class": _1}, name=5),
        ]
        bracid.closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625),
        }
        bracid.examples_covered_by_rule = {}
        bracid.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.conf_matrix = ConfusionMatrix(TP={0, 1}, FP={3, 4}, TN={2, 5}, FN=set())
        new_rule = pd.Series({"B": Bounds(lower=0.5, upper=1.0), "C": Bounds(lower=3, upper=3), "Class": _1}, name=0)
        # tagged, initial_rules = add_tags_and_extract_rules(df, 2, class_col_name, min_max, classes)
        correct_f1 = 0.8
        f1 = bracid.evaluate_f1_update_confusion_matrix(df, new_rule, class_col_name, min_max, classes)
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=0, dist=0.6025),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.0),
            5: Data(rule_id=0, dist=0.010000000000000002),
        }
        print(f1, correct_f1)
        self.assertAlmostEqual(f1, correct_f1)
        assert_almost_equal(self, bracid.closest_rule_per_example, correct_closest_rule_per_example)
        correct_conf_matrix = ConfusionMatrix(TP={0, 1}, TN={2, 4, 5}, FP={3}, FN=set())
        self.assertEqual(bracid.conf_matrix, correct_conf_matrix)

    def test_evaluate_f1_update_confusion_matrix_not_updated(self):
        """Tests what happens if input has a numeric and a nominal feature and a rule that predicts an example is
        not updated as F1 score doesn't improve"""
        bracid = BRACID(k=-1, minority_class=_0)
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0}, name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0}, name=1),
            pd.Series({"B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1), "Class": _1}, name=2),
            pd.Series({"B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5), "Class": _1}, name=3),
            pd.Series({"B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3), "Class": _1}, name=4),
            pd.Series({"B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2), "Class": _1}, name=5),
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
            5: Data(rule_id=2, dist=0.67015625),
        }
        bracid.conf_matrix = ConfusionMatrix(TP={0, 1}, FP=set(), TN={2, 5}, FN={3, 4})
        new_rule = pd.Series({"B": (0.5, 0.5), "C": (3, 3), "Class": _1}, name=4)
        correct_f1 = 2 * 1 * 0.5 / 1.5

        f1 = bracid.evaluate_f1_update_confusion_matrix(df, new_rule, class_col_name, min_max, classes)
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=4, dist=0.013906250000000002),
        }
        self.assertAlmostEqual(f1, correct_f1)
        assert_almost_equal(self, bracid.closest_rule_per_example, correct_closest_rule_per_example)
        correct_conf_matrix = ConfusionMatrix(TP={0, 1}, TN={2, 5}, FP=set(), FN={3, 4})
        self.assertEqual(bracid.conf_matrix, correct_conf_matrix)
