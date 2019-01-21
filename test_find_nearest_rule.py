from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import find_nearest_rule, Data
import scripts.vars as my_vars


class TestFindNearestRule(TestCase):
    """Tests find_nearest_rule() in utils.py"""

    def test_find_nearest_rule_no_ties(self):
        """Tests that the nearest rule is found per example assuming no ties"""
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
        my_vars.minority_class = "apple"
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "apple"}, name=1),
            pd.Series({"A": "high", "B": (4, 4), "C": (1, 1), "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": (1.5, 1.5), "C": (0.5, 0.5), "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": (0.5, 0.5), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": (0.75, 0.75), "C": (2, 2), "Class": "banana"}, name=5)
        ]
        # Reset because other tests change the data
        my_vars.closest_examples_per_rule = {}
        my_vars.closest_rule_per_example = {}
        my_vars.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}, 6: {8}}
        my_vars.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8}
        my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        my_vars.examples_covered_by_rule = {6: {8}}
        my_vars.unique_rules = {}
        my_vars.conf_matrix = {}
        for example_id, example in df.iterrows():
            rule, dist, was_updated = find_nearest_rule(rules, example, class_col_name, lookup, min_max, classes,
                                                        my_vars.examples_covered_by_rule,
                                                        label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
            # print("eid: {} rule:\n{}\ndist: {} updated: {}".format(example_id, rule, dist, was_updated))
            self.assertTrue(was_updated is True)

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
        print(my_vars.closest_rule_per_example)
        print(correct_closest_rule_per_example)
        # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id].rule_id and
                            abs(dist - correct_closest_rule_per_example[example_id].dist) < 0.001)
        self.assertTrue(correct_closest_examples_per_rule == my_vars.closest_examples_per_rule)

    def test_find_nearest_rule_ties(self):
        """Tests that ties (multiple rules cover an example) are resolved properly"""
        df = pd.DataFrame({"A": ["low", "low", "low"], "B": [1, 1, 2],
                           "C": [1, 2, 3],
                           "Class": ["apple", "banana", "banana"]})
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
        my_vars.minority_class = "apple"
        rules = [
            pd.Series({"A": "low", "B": (1, 2), "C": (1, 3), "Class": "apple"}, name=0),
            pd.Series({"B": (1, 2), "C": (1, 3), "Class": "apple"}, name=1),
            pd.Series({"B": (0, 3), "C": (1, 4), "Class": "apple"}, name=2),
        ]
        # Reset because other tests change the data
        my_vars.closest_examples_per_rule = {}
        my_vars.closest_rule_per_example = {}
        my_vars.examples_covered_by_rule = {}
        my_vars.unique_rules = {}
        my_vars.seed_example_rule = {0: {0}, 1: {1}, 2: {2}}
        my_vars.seed_rule_example = {0: 0, 1: 1, 2: 2}
        my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[2]}
        my_vars.conf_matrix = {}

        for example_id, example in df.iterrows():
            rule, dist, was_updated = find_nearest_rule(rules, example, class_col_name, lookup, min_max, classes,
                                                        my_vars.examples_covered_by_rule,
                                                        label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
            # print("eid: {} rule:\n{}\ndist: {} updated: {}".format(example_id, rule, dist, was_updated))
            print("eid: {} rule: {} dist: {} updated: {}".format(example_id, rule.name, dist, was_updated))
            self.assertTrue(was_updated is True)
        print("closest rules")
        print(my_vars.closest_rule_per_example)
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
        print(my_vars.closest_rule_per_example)
        print(my_vars.closest_examples_per_rule)
        # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id].rule_id and
                            abs(dist - correct_closest_rule_per_example[example_id].dist) < 0.001)
        self.assertTrue(correct_closest_examples_per_rule == my_vars.closest_examples_per_rule)
