from unittest import TestCase

import pandas as pd
import pytest

import multi_imbalance.classifiers.bracid.vars as my_vars
from tests.classifiers.bracid.classes_ import _0, _1
from multi_imbalance.classifiers.bracid.bracid import BRACID, Data, Bounds


class TestFindNeighbors(TestCase):
    """Test find_neighbors() from utils.py"""

    @pytest.mark.skip(reason="TODO: remove test or test logging output")
    def test_find_neighbors_too_few(self):
        """Test that warning is thrown if too few neighbors exist"""
        k = 3
        bracid = BRACID(k=k, minority_class=-1)
        dataset = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [2, 2], "Class": [_0, _1]})
        rule = pd.Series({"A": (0.1, 1), "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0})
        classes = [_0, _1]
        class_col_name = "Class"
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}, "C": {"min": 1, "max": 2}})
        self.assertWarns(
            UserWarning,
            bracid.find_nearest_examples,
            dataset,
            k,
            rule,
            class_col_name,
            min_max,
            classes,
            label_type=my_vars.SAME_LABEL_AS_RULE,
            only_uncovered_neighbors=False,
        )

    def test_find_neighbors_numeric_nominal_label_type(self):
        """Tests what happens if input has a numeric and a nominal feature and we vary label_type as parameter"""
        k = 3
        bracid = BRACID(k=k, minority_class=-1)
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0}, name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0}, name=1),
            pd.Series({"B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1), "Class": _1}, name=2),
            pd.Series({"B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5), "Class": _1}, name=3),
            pd.Series({"B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3), "Class": _1}, name=4),
            pd.Series({"B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2), "Class": _1}, name=5),
        ]
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.closest_rule_per_example = {}
        bracid.closest_examples_per_rule = {}
        correct_all = df.iloc[[0, 1, 5]]
        correct_same = df.iloc[[5, 3, 4]]
        correct_opposite = df.iloc[[0, 1]]
        rule = pd.Series({"B": (1, 1), "Class": _1}, name=0)
        classes = [_0, _1]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        neighbors_all, _, _ = bracid.find_nearest_examples(
            df, k, rule, class_col_name, min_max, classes, label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False
        )
        neighbors_same, _, _ = bracid.find_nearest_examples(
            df, k, rule, class_col_name, min_max, classes, label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=False
        )
        neighbors_opposite, _, _ = bracid.find_nearest_examples(
            df, k, rule, class_col_name, min_max, classes, label_type=my_vars.OPPOSITE_LABEL_TO_RULE, only_uncovered_neighbors=False
        )
        print(neighbors_all)
        print(neighbors_same)
        print(neighbors_opposite)
        pd.testing.assert_frame_equal(neighbors_all, correct_all)
        pd.testing.assert_frame_equal(neighbors_same, correct_same)
        pd.testing.assert_frame_equal(neighbors_opposite, correct_opposite)

    def test_find_neighbors_numeric_nominal_covered(self):
        """Tests what happens if input has a numeric and a nominal feature and some examples are already covered
        by the rule"""
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        for k in [1, 2, 3, 4]:
            with self.subTest(f"k={k}"):
                bracid = BRACID(k=k, minority_class=-1)
                if k == 1:
                    correct = df.iloc[[5]]
                elif k == 2:
                    correct = df.iloc[[5, 3]]
                elif k == 3:
                    correct = df.iloc[[5, 3, 4]]
                elif k >= 4:
                    # Examples at indices 2 and 4 are already covered by the rule, so don't return them as neighbors
                    bracid.examples_covered_by_rule = {0: {2, 4}}
                    correct = df.iloc[[5, 3]]
                rule = pd.Series({"B": Bounds(lower=1, upper=1), "Class": _1}, name=0)
                classes = [_0, _1]
                min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})

                neighbors, _, _ = bracid.find_nearest_examples(
                    df, k, rule, class_col_name, min_max, classes, label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=True
                )
                pd.testing.assert_frame_equal(correct, neighbors)

    def test_find_neighbors_numeric_nominal_stats(self):
        """Tests that global statistics are updated accordingly"""
        k = 4
        bracid = BRACID(k=k, minority_class=-1)
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        rule = pd.Series({"B": (1, 1), "Class": _1}, name=0)
        bracid.closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625),
        }
        # Reset because other tests added data, so if you only run this test it would work, but not if other
        # tests are run prior to that
        bracid.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0}, name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0}, name=1),
            pd.Series({"B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1), "Class": _1}, name=2),
            pd.Series({"B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5), "Class": _1}, name=3),
            pd.Series({"B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3), "Class": _1}, name=4),
            pd.Series({"B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2), "Class": _1}, name=5),
        ]
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        correct = df.iloc[[5, 3, 4, 2]]

        classes = [_0, _1]
        bracid.minority_class = _1
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        correct_covered = {}
        correct_examples_per_rule = {0: {1, 2, 3, 4, 5}, 1: {0}}
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=0, dist=0.09),
            3: Data(rule_id=0, dist=0.0025000000000000005),
            4: Data(rule_id=0, dist=0.0025000000000000005),
            5: Data(rule_id=0, dist=0.0006250000000000001),
        }
        neighbors, _, _ = bracid.find_nearest_examples(
            df, k, rule, class_col_name, min_max, classes, label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=False
        )
        pd.testing.assert_frame_equal(neighbors, correct)
        self.assertDictEqual(correct_covered, bracid.examples_covered_by_rule)
        self.assertDictEqual(correct_examples_per_rule, bracid.closest_examples_per_rule)
        for example_id, (rule_id, dist) in correct_closest_rule_per_example.items():
            features = bracid.all_rules[rule_id].size
            self.assertIn(example_id, bracid.closest_rule_per_example)
            other_id, other_dist = bracid.closest_rule_per_example[example_id]
            other_features = bracid.all_rules[other_id].size
            self.assertEqual(rule_id, other_id)
            self.assertEqual(features, other_features)
            self.assertAlmostEqual(dist, other_dist, delta=0.0001)

    def test_find_neighbors_numeric_nominal_covers(self):
        """Tests that the stats for a newly covered rule are updated (dist = 0)"""
        """Tests that global statistics are updated accordingly"""
        k = 4
        bracid = BRACID(k=k, minority_class=_1)
        df = pd.DataFrame({"B": [1, 1, 1, 1, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0}, name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0}, name=1),
            pd.Series({"B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1), "Class": _1}, name=2),
            pd.Series({"B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5), "Class": _1}, name=3),
            pd.Series({"B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3), "Class": _1}, name=4),
            pd.Series({"B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2), "Class": _1}, name=5),
        ]
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: (5, 0.67015625),
            3: (1, 0.038125),
            4: (0, 0.015625),
            5: (2, 0.67015625),
        }
        bracid.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
        correct = df.iloc[[2, 3, 5, 4]]
        rule = pd.Series({"B": (1, 1), "Class": _1}, name=0)
        classes = [_0, _1]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        bracid.examples_covered_by_rule = {1: {2}}
        correct_covered = {1: {2}, 0: {2, 3}}
        correct_examples_per_rule = {0: {1, 2, 3, 4, 5}, 1: {0}}
        correct_closest_rule_per_example = {
            0: (1, 0.010000000000000002),
            1: (0, 0.010000000000000002),
            2: Data(rule_id=0, dist=0.0),
            3: Data(rule_id=0, dist=0.0),
            4: Data(rule_id=0, dist=0.0025000000000000005),
            5: Data(rule_id=0, dist=0.0006250000000000001),
        }
        neighbors, _, _ = bracid.find_nearest_examples(
            df, k, rule, class_col_name, min_max, classes, label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=False
        )
        pd.testing.assert_frame_equal(neighbors, correct)
        self.assertEqual(correct_covered, bracid.examples_covered_by_rule)
        self.assertEqual(correct_examples_per_rule, bracid.closest_examples_per_rule)
        for example_id, (rule_id, dist) in correct_closest_rule_per_example.items():
            self.assertIn(example_id, bracid.closest_rule_per_example)
            other_id, other_dist = bracid.closest_rule_per_example[example_id]
            self.assertEqual(rule_id, other_id)
            self.assertAlmostEqual(dist, other_dist, delta=0.0001)
