from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.bracid import BRACID, Data
import scripts.vars as my_vars
from unit_tests.classes_ import _0, _1


class TestFindNearestRule(TestCase):
    """Tests find_nearest_rule() in utils.py"""

    def test_find_nearest_rule_no_ties(self):
        """Tests that the nearest rule is found per example assuming no ties"""
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
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        bracid.minority_class = _0
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": _0}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": _0}, name=1),
            pd.Series({"A": "high", "B": (4, 4), "C": (1, 1), "Class": _1}, name=2),
            pd.Series({"A": "low", "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": _1}, name=3),
            pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": _1}, name=4),
            pd.Series({"A": "high", "B": (0.75, 0.75), "C": (2, 2), "Class": _1}, name=5)
        ]
        # Reset because other tests change the data
        bracid.closest_examples_per_rule = {}
        bracid.closest_rule_per_example = {}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}, 6: {8}}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.examples_covered_by_rule = {6: {8}}
        bracid.unique_rules = {}
        bracid.conf_matrix = {}
        for example_id, example in df.iterrows():
            rule, dist, was_updated = bracid.find_nearest_rule(rules, example, class_col_name, min_max, classes,
                                                        bracid.examples_covered_by_rule,
                                                        label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
            # print("eid: {} rule:\n{}\ndist: {} updated: {}".format(example_id, rule, dist, was_updated))
            self.assertTrue(was_updated)

        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625)}
        correct_closest_examples_per_rule = {
            0: {1, 4},
            1: {0, 3},
            2: {5},
            5: {2}
        }
        print(bracid.closest_rule_per_example)
        print(correct_closest_rule_per_example)
        # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
        for example_id in bracid.closest_rule_per_example:
            rule_id, dist = bracid.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id].rule_id and
                            abs(dist - correct_closest_rule_per_example[example_id].dist) < 0.001)
        self.assertEqual(correct_closest_examples_per_rule, bracid.closest_examples_per_rule)

    def test_find_nearest_rule_ties(self):
        """Tests that ties (multiple rules cover an example) are resolved properly"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["low", "low", "low"], "B": [1, 1, 2],
                           "C": [1, 2, 3],
                           "Class": [_0, _1, _1]})
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
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        bracid.minority_class = _0
        rules = [
            pd.Series({"A": "low", "B": (1, 2), "C": (1, 3), "Class": _0}, name=0),
            pd.Series({"B": (1, 2), "C": (1, 3), "Class": _0}, name=1),
            pd.Series({"B": (0, 3), "C": (1, 4), "Class": _0}, name=2),
        ]
        # Reset because other tests change the data
        bracid.closest_examples_per_rule = {}
        bracid.closest_rule_per_example = {}
        bracid.examples_covered_by_rule = {}
        bracid.unique_rules = {}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2}
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2]}
        bracid.conf_matrix = {}

        for example_id, example in df.iterrows():
            rule, dist, was_updated = bracid.find_nearest_rule(rules, example, class_col_name, min_max, classes,
                                                        bracid.examples_covered_by_rule,
                                                        label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
            # print("eid: {} rule:\n{}\ndist: {} updated: {}".format(example_id, rule, dist, was_updated))
            print("eid: {} rule: {} dist: {} updated: {}".format(example_id, rule.name, dist, was_updated))
            self.assertTrue(was_updated)
        print("closest rules")
        print(bracid.closest_rule_per_example)
        # Note: it's permissible that rule 1 covers example 1 (although example 1 is the seed for rule 1)
        # because rule 1 already covers example 0
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.0),
            1: Data(rule_id=1, dist=0.0),
            2: Data(rule_id=1, dist=0.0),
        }
        correct_closest_examples_per_rule = {
            1: {0, 1, 2},
        }
        print(bracid.closest_rule_per_example)
        print(bracid.closest_examples_per_rule)
        # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
        for example_id in bracid.closest_rule_per_example:
            rule_id, dist = bracid.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id].rule_id and
                            abs(dist - correct_closest_rule_per_example[example_id].dist) < 0.001)
        self.assertEqual(correct_closest_examples_per_rule, bracid.closest_examples_per_rule)
