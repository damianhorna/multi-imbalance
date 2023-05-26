from unittest import TestCase

import pandas as pd

from multi_imbalance.classifiers.bracid.bracid import BRACID, Data
import multi_imbalance.classifiers.bracid.vars as my_vars
from tests.classifiers.bracid.classes_ import _0, _1
from tests.classifiers.bracid.assertions import assert_almost_equal

class TestFindNearestRule(TestCase):
    """Tests find_nearest_rule() in utils.py"""

    def test_find_nearest_rule_no_ties(self):
        """Tests that the nearest rule is found per example assuming no ties"""
        bracid = BRACID(k=-1, minority_class = _0)
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        rules = [
            pd.Series({"B": (1, 1), "C": (3, 3), "Class": _0}, name=0),
            pd.Series({"B": (1, 1), "C": (2, 2), "Class": _0}, name=1),
            pd.Series({"B": (4, 4), "C": (1, 1), "Class": _1}, name=2),
            pd.Series({"B": (1.5, 1.5), "C": (0.5, 0.5), "Class": _1}, name=3),
            pd.Series({"B": (0.5, 0.5), "C": (3, 3), "Class": _1}, name=4),
            pd.Series({"B": (0.75, 0.75), "C": (2, 2), "Class": _1}, name=5)
        ]
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}, 6: {8}}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.examples_covered_by_rule = {6: {8}}
        for example_id, example in df.iterrows():
            rule, dist, was_updated = bracid.find_nearest_rule(rules, example, class_col_name, min_max, classes,
                                                        bracid.examples_covered_by_rule,
                                                        label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
            self.assertTrue(was_updated)

        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=5, dist=0.00390625),
            2: Data(rule_id=3, dist=0.393125),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=5, dist=0.013906250000000002),
            5: Data(rule_id=1, dist=0.00390625)}
        correct_closest_examples_per_rule = {1: {0, 3, 5}, 5: {1, 4}, 3: {2}}
        print(bracid.closest_rule_per_example)
        print(correct_closest_rule_per_example)

        assert_almost_equal(self, bracid.closest_rule_per_example, correct_closest_rule_per_example)
        self.assertEqual(correct_closest_examples_per_rule, bracid.closest_examples_per_rule)

    def test_find_nearest_rule_ties(self):
        """Tests that ties (multiple rules cover an example) are resolved properly"""
        bracid = BRACID(k=-1, minority_class = _0)
        df = pd.DataFrame({"B": [1, 1, 2],
                           "C": [1, 2, 3],
                           "Class": [_0, _1, _1]})
        class_col_name = "Class"
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        rules = [
            pd.Series({"B": (1, 2), "C": (1, 3), "Class": _0}, name=0),
            pd.Series({"B": (1, 2), "C": (1, 3), "Class": _0}, name=1),
            pd.Series({"B": (0, 3), "C": (1, 4), "Class": _0}, name=2),
        ]
        # Reset because other tests change the data
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2]}

        for example_id, example in df.iterrows():
            rule, dist, was_updated = bracid.find_nearest_rule(rules, example, class_col_name, min_max, classes,
                                                        bracid.examples_covered_by_rule,
                                                        label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
            # print("eid: {} rule:\n{}\ndist: {} updated: {}".format(example_id, rule, dist, was_updated))
            print("eid: {} rule: {} dist: {} updated: {}".format(example_id, rule.name, dist, was_updated))
            self.assertTrue(was_updated)
        print("closest rules")
        print(bracid.closest_rule_per_example)
        correct_closest_rule_per_example = {0: Data(rule_id=1, dist=0.0), 1: Data(rule_id=0, dist=0.0), 2: Data(rule_id=0, dist=0.0)}
        correct_closest_examples_per_rule = {1: {0}, 0: {1, 2}}
        print(bracid.closest_rule_per_example)
        print(bracid.closest_examples_per_rule)
        assert_almost_equal(self, bracid.closest_rule_per_example, correct_closest_rule_per_example)
        self.assertEqual(correct_closest_examples_per_rule, bracid.closest_examples_per_rule)
