from unittest import TestCase
from collections import Counter, deque

import pandas as pd

from scripts.utils import add_all_good_rules, find_nearest_examples, evaluate_f1_initialize_confusion_matrix, Data, \
    Bounds, compute_hashable_key
import scripts.vars as my_vars


class TestAddAllGoodRules(TestCase):
    """Tests add_all_good_rules() in utils.py"""

    def test_add_all_good_rules(self):
        """Tests that rule set is updated when a generalized rule improves F1"""
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
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        my_vars.minority_class = "banana"
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2)  # Current rule to be tested is always at the end
        ]
        test_idx = -1
        my_vars.latest_rule_id = len(rules) - 1
        my_vars.examples_covered_by_rule = {}
        my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[test_idx], 3: rules[2], 4: rules[3], 5: rules[4]}
        my_vars.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        my_vars.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        my_vars.unique_rules = {}
        for rule in rules:
            hash_val = compute_hashable_key(rule)
            my_vars.unique_rules.setdefault(hash_val, set()).add(rule.name)

        initial_correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625)}
        initial_f1 = evaluate_f1_initialize_confusion_matrix(df, rules, class_col_name, lookup, min_max, classes)
        correct_confusion_matrix = {my_vars.TP: {2, 5}, my_vars.FP: set(), my_vars.TN: {0, 1}, my_vars.FN: {3, 4}}
        correct_rules = 8
        self.assertTrue(my_vars.conf_matrix == correct_confusion_matrix)

        # Make sure confusion matrix, closest rule per example are correct at the beginning
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == initial_correct_closest_rule_per_example[example_id].rule_id and
                            abs(dist - initial_correct_closest_rule_per_example[example_id].dist) < 0.001)

        correct_initial_f1 = 2 * 0.5 * 1 / 1.5
        self.assertTrue(initial_f1 == correct_initial_f1)
        k = 3
        neighbors, dists, _ = find_nearest_examples(df, k, rules[test_idx], class_col_name, lookup, min_max, classes,
                                                    label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=
                                                    True)
        improved, updated_rules, f1 = add_all_good_rules(df, neighbors, rules[test_idx], rules, initial_f1,
                                                         class_col_name, lookup, min_max, classes)
        self.assertTrue(improved is True)
        print("f1", f1)
        # correct_covered = {2: {0, 1, 2, 3, 4, 5}}
        correct_covered = {6: {0, 1, 2, 4, 5}, 7: {3}}
        correct_confusion_matrix = {my_vars.TP: {2, 3, 4, 5}, my_vars.FP: {0, 1}, my_vars.TN: set(), my_vars.FN: set()}
        # correct_closest_rule_per_example = {
        #     0: Data(rule_id=2, dist=0.0),
        #     1: Data(rule_id=2, dist=0.0),
        #     2: Data(rule_id=2, dist=0.0),
        #     3: Data(rule_id=2, dist=0.0),
        #     4: Data(rule_id=2, dist=0.0),
        #     5: Data(rule_id=2, dist=0.0)}
        correct_closest_rule_per_example = {
            0: Data(rule_id=6, dist=0.0),
            1: Data(rule_id=6, dist=0.0),
            2: Data(rule_id=6, dist=0.0),
            3: Data(rule_id=7, dist=0.0),
            4: Data(rule_id=6, dist=0.0),
            5: Data(rule_id=6, dist=0.0)
        }
        correct_f1 = 0.8
        self.assertTrue(correct_f1 == f1)
        for example_id in my_vars.closest_rule_per_example:
            rule_id, dist = my_vars.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id].rule_id and
                            abs(dist - correct_closest_rule_per_example[example_id].dist) < 0.001)
        self.assertTrue(my_vars.conf_matrix == correct_confusion_matrix)
        # latest_rule_id must be 7 as 2 new rules were added to the 5 initial rules
        self.assertTrue(len(updated_rules) == correct_rules and my_vars.latest_rule_id == (correct_rules - 1))
        self.assertTrue(correct_covered == my_vars.examples_covered_by_rule)

    def test_add_all_good_rules_bug(self):
        """Tests a case that fails in test_bracid_stops()"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"],
                           "tag": [my_vars.BORDERLINE, my_vars.BORDERLINE, my_vars.SAFE, my_vars.BORDERLINE,
                                   my_vars.BORDERLINE, my_vars.BORDERLINE]})
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
        test_idx = 5
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        my_vars.minority_class = "banana"
        initial_f1 = 0.8571428571428571
        # All initial values are taken from the console to reproduce the errors
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3.0, upper=3.0),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2.0, upper=2.0),
                       "Class": "banana"}, name=5),
            pd.Series({"A": "low", "B": Bounds(lower=1.0, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                       "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1.0, upper=1.5), "C": Bounds(lower=0.5, upper=2.0),
                       "Class": "apple"}, name=1),
            pd.Series({"B": Bounds(lower=1.5, upper=4.0), "C": Bounds(0.5, upper=1.0), "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3)
        ]
        rules = deque(rules)
        my_vars.unique_rules = {}
        for rule in rules:
            rule_hash = compute_hashable_key(rule)
            my_vars.unique_rules.setdefault(rule_hash, set()).add(rule.name)
        print(my_vars.unique_rules)

        my_vars.all_rules = {0: rules[2], 1: rules[3], 2: rules[4], 3: rules[test_idx], 4: rules[0], 5: rules[1]}
        neighbors = df.loc[[1, 0, 4]]
        my_vars.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        my_vars.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        my_vars.closest_rule_per_example = {
            1: Data(rule_id=0, dist=0.0),
            4: Data(rule_id=0, dist=0.015625),
            3: Data(rule_id=2, dist=0.0),
            5: Data(rule_id=2, dist=0.04515625),
            2: Data(rule_id=5, dist=0.67015625),
            0: Data(rule_id=1, dist=0.010000000000000002)
        }
        my_vars.closest_examples_per_rule = {0: {1, 4}, 1: {0}, 2: {3, 5}, 5: {2}}
        my_vars.conf_matrix = {my_vars.TP: {2, 3, 5}, my_vars.FP: set(), my_vars.TN: {0, 1}, my_vars.FN: {4}}
        my_vars.examples_covered_by_rule = {0: {1}, 2: {3}}
        my_vars.latest_rule_id = 5

        # Process rule 3: first it is replaced, then its generalization becomes rule 6 and it'll cover examples 0 and 4
        improved, updated_rules, f1 = add_all_good_rules(df, neighbors, rules[test_idx], rules, initial_f1,
                                                         class_col_name, lookup, min_max, classes)
        correct_f1 = 0.888888888888889
        rule_6 = pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                            "Class": "banana"}, name=6)
        correct_conf_matrix = {my_vars.TP: {2, 3, 4, 5}, my_vars.FP: {0}, my_vars.TN: {1}, my_vars.FN: set()}
        correct_closest_rule = {1: Data(rule_id=0, dist=0.0), 4: Data(rule_id=6, dist=0.0),
                                3: Data(rule_id=2, dist=0.0), 5: Data(rule_id=2, dist=0.04515625),
                                2: Data(rule_id=5, dist=0.67015625), 0: Data(rule_id=6, dist=0.0)}
        correct_closest_examples = {0: {1}, 2: {3, 5}, 5: {2}, 6: {0, 4}}
        correct_covered_examples = {0: {1}, 2: {3}, 6: {0, 4}}
        correct_rule_6_hash = compute_hashable_key(rule_6)
        self.assertTrue(abs(correct_f1 - f1) < my_vars.PRECISION)
        self.assertTrue(improved is True)
        self.assertTrue(updated_rules[-1].equals(rule_6))
        self.assertTrue(my_vars.conf_matrix == correct_conf_matrix)
        self.assertTrue(my_vars.closest_rule_per_example == correct_closest_rule)
        self.assertTrue(my_vars.closest_examples_per_rule == correct_closest_examples)
        self.assertTrue(my_vars.examples_covered_by_rule == correct_covered_examples)
        self.assertTrue(len(my_vars.unique_rules) == 7 and correct_rule_6_hash in my_vars.unique_rules)

        # Process rule 4 whose generalization becomes a duplicate of rule 6, so delete it
        # Rule 7 is generalized from rule 4, so it's added as well
        rule_4 = updated_rules.popleft()
        updated_rules.append(rule_4)
        neighbors = df.loc[[0, 1, 3]]
        improved, updated_rules, f1 = add_all_good_rules(df, neighbors, rule_4, updated_rules, f1,
                                                         class_col_name, lookup, min_max, classes)
        print(updated_rules)
        self.assertTrue(abs(correct_f1 - f1) < my_vars.PRECISION)
        self.assertTrue(improved is True)
        self.assertTrue(rule_4.name not in my_vars.all_rules)
        self.assertTrue(my_vars.conf_matrix == correct_conf_matrix)
        self.assertTrue(my_vars.closest_rule_per_example == correct_closest_rule)
        self.assertTrue(my_vars.closest_examples_per_rule == correct_closest_examples)
        self.assertTrue(my_vars.examples_covered_by_rule == correct_covered_examples)
        self.assertTrue(len(my_vars.unique_rules) == 7 and rule_4.name not in my_vars.all_rules)
        print("improved?", improved)
        print("updated rules")
        print("f1", f1)

        # Process rule 5

