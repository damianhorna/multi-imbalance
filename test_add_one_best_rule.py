from unittest import TestCase
from collections import Counter

import pandas as pd

# from scripts.utils import add_one_best_rule, find_nearest_examples, compute_hashable_key, Data, Bounds
from scripts.bracid import BRACID, Bounds, Data, ConfusionMatrix
import scripts.vars as my_vars


class TestAddOneBestRule(TestCase):
    """Tests add_one_best_rule() from utils.py"""

    def test_add_one_best_rule_update(self):
        """Tests that rule set is updated when a generalized rule improves F1"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        bracid = BRACID()
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
        bracid.minority_class = "apple"
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0)  # Current rule is always at the end of the list
        ]
        bracid.closest_examples_per_rule = {
            0: {1, 4},
            1: {0, 3},
            2: {5},
            5: {2}
        }
        bracid.closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625)}
        test_idx = -1
        # Reset because other tests change the data
        bracid.examples_covered_by_rule = {}
        bracid.all_rules = {0: rules[test_idx], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[0]}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        bracid.unique_rules = {}
        for rule in rules:
            rule_hash = bracid.compute_hashable_key(rule)
            bracid.unique_rules.setdefault(rule_hash, set()).add(rule.name)

        # Actually, correctly it should've been
        # bracid.conf_matrix = ConfusionMatrix(TP= {0, 1}, FP= set(), TN= {2, 5}, FN= {3, 4})
        # at the start (i.e. F1=0.66666), but to see if it changes, it's changed
        # bracid.conf_matrix = ConfusionMatrix(TP= {0}, FP= set(), TN= {1, 2, 5}, FN= {3, 4})
        bracid.conf_matrix = ConfusionMatrix(TP={0}, FP=set(), TN={1, 2, 5}, FN={3, 4})
        initial_f1 = 0.1
        k = 3
        neighbors, dists, _ = bracid.find_nearest_examples(df, k, rules[test_idx], class_col_name, lookup, min_max, classes,
                                                    label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=
                                                    True)
        improved, updated_rules, f1 = bracid.add_one_best_rule(df, neighbors, rules[test_idx], rules, initial_f1,
                                                        class_col_name, lookup, min_max, classes)

        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.0),
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
        correct_f1 = 2 * 0.5 * 1 / 1.5
        self.assertTrue(improved is True)
        self.assertTrue(abs(correct_f1 - f1) < my_vars.PRECISION)
        correct_generalized_rule = pd.Series({"A": "low", "B": (1, 1), "C": (2.0, 3), "Class": "apple"}, name=0)
        # correct_confusion_matrix = ConfusionMatrix(TP= {0, 1}, FP= set(), TN= {2, 5}, FN= {3, 4})
        correct_confusion_matrix = ConfusionMatrix(TP={0, 1}, FP=set(), TN={2, 5}, FN={3, 4})
        # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
        for example_id in bracid.closest_rule_per_example:
            rule_id, dist = bracid.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id].rule_id and
                            abs(dist - correct_closest_rule_per_example[example_id].dist) < 0.001)
        self.assertTrue(updated_rules[test_idx].equals(correct_generalized_rule))
        self.assertTrue(bracid.conf_matrix == correct_confusion_matrix)
        self.assertTrue(correct_closest_examples_per_rule == bracid.closest_examples_per_rule)

    def test_add_one_best_rule_update_stats(self):
        """Tests that rule set is updated when a generalized rule improves F1 and also the mapping of closest rule per
        example changes"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        bracid = BRACID()
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
        test_idx = -1
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        bracid.minority_class = "apple"
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0)  # Current rule is always at the end of the list
        ]
        bracid.closest_examples_per_rule = {
            0: {4},
            1: {0, 1, 3},   # Change compared to previous test case
            2: {5},
            5: {2}
        }
        bracid.closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=1, dist=0.010000000000000002),   # Change compared to previous test case
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625)}
        # Reset because other tests change the data
        # bracid.examples_covered_by_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}, 6: {8}}
        bracid.examples_covered_by_rule = {}
        bracid.all_rules = {0: rules[test_idx], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        bracid.unique_rules = {}
        bracid.unique_rules = {}
        for rule in rules:
            rule_hash = bracid.compute_hashable_key(rule)
            bracid.unique_rules.setdefault(rule_hash, set()).add(rule.name)

        # Actually, correctly it should've been
        # bracid.conf_matrix = ConfusionMatrix(TP= {0, 1}, FP= set(), TN= {2, 5}, FN= {3, 4})
        # at the start (i.e. F1=0.66666), but to see if it changes, it's changed
        # bracid.conf_matrix = ConfusionMatrix(TP= {0}, FP= set(), TN= {1, 2, 5}, FN= {3, 4})
        bracid.conf_matrix = ConfusionMatrix(TP={0}, FP=set(), TN={1, 2, 5}, FN={3, 4})
        initial_f1 = 0.1
        k = 3
        neighbors, dists, _ = bracid.find_nearest_examples(df, k, rules[test_idx], class_col_name, lookup, min_max, classes,
                                                    label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=
                                                    True)
        improved, updated_rules, f1 = bracid.add_one_best_rule(df, neighbors, rules[test_idx], rules, initial_f1,
                                                        class_col_name, lookup, min_max, classes)

        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.0),
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
        correct_f1 = 2 * 0.5 * 1 / 1.5
        self.assertTrue(abs(correct_f1 - f1) < my_vars.PRECISION)
        self.assertTrue(improved is True)
        correct_generalized_rule = pd.Series({"A": "low", "B": (1, 1), "C": (2.0, 3), "Class": "apple"}, name=0)
        # correct_confusion_matrix = ConfusionMatrix(TP= {0, 1}, FP= set(), TN= {2, 5}, FN= {3, 4})
        correct_confusion_matrix = ConfusionMatrix(TP={0, 1}, FP=set(), TN={2, 5}, FN={3, 4})
        # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
        for example_id in bracid.closest_rule_per_example:
            rule_id, dist = bracid.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id].rule_id and
                            abs(dist - correct_closest_rule_per_example[example_id].dist) < 0.001)
        self.assertTrue(updated_rules[test_idx].equals(correct_generalized_rule))
        self.assertTrue(bracid.conf_matrix == correct_confusion_matrix)
        print(correct_closest_examples_per_rule)
        print(bracid.closest_examples_per_rule)
        self.assertTrue(correct_closest_examples_per_rule == bracid.closest_examples_per_rule)

    def test_add_one_best_rule_no_update(self):
        """Tests that rule set is not updated when no generalized rule improves F1"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        bracid = BRACID()
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
        test_idx = -1
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        bracid.minority_class = "apple"
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0)   # Current rule is always at the end of the list
        ]
        bracid.closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625)}
        bracid.all_rules = {0: rules[test_idx], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[0]}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        bracid.conf_matrix = ConfusionMatrix(TP={0, 1}, FP= set(), TN= {2, 5}, FN= {3, 4})
        bracid.examples_covered_by_rule = {}
        # F1 is actually 0.6666, but setting it to 0.8 makes it not update any rule
        initial_f1 = 0.8
        k = 3
        bracid.unique_rules = {}
        for rule in rules:
            rule_hash = bracid.compute_hashable_key(rule)
            bracid.unique_rules.setdefault(rule_hash, set()).add(rule.name)

        neighbors, dists, _ = bracid.find_nearest_examples(df, k, rules[test_idx], class_col_name, lookup, min_max, classes,
                                                    label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=
                                                    True)
        improved, updated_rules, f1 = bracid.add_one_best_rule(df, neighbors, rules[test_idx], rules, initial_f1,
                                                        class_col_name, lookup, min_max, classes)
        correct_closest_rule_per_example = {
            0: Data(rule_id=1, dist=0.010000000000000002),
            1: Data(rule_id=0, dist=0.010000000000000002),
            2: Data(rule_id=5, dist=0.67015625),
            3: Data(rule_id=1, dist=0.038125),
            4: Data(rule_id=0, dist=0.015625),
            5: Data(rule_id=2, dist=0.67015625)}
        self.assertTrue(improved is False)
        correct_f1 = initial_f1
        self.assertTrue(abs(correct_f1 - f1) < my_vars.PRECISION)
        correct_generalized_rule = pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0)
        # correct_confusion_matrix = ConfusionMatrix(TP= {0, 1}, FP= set(), TN= {2, 5}, FN= {3, 4})
        correct_confusion_matrix = ConfusionMatrix(TP={0, 1}, FP= set(), TN={2, 5}, FN={3, 4})
        # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
        for example_id in bracid.closest_rule_per_example:
            rule_id, dist = bracid.closest_rule_per_example[example_id]
            self.assertTrue(rule_id == correct_closest_rule_per_example[example_id].rule_id and
                            abs(dist - correct_closest_rule_per_example[example_id].dist) < 0.001)
        print(rules[test_idx])
        print(correct_generalized_rule)
        print("updated")
        print(updated_rules)
        self.assertTrue(updated_rules[test_idx].equals(correct_generalized_rule))
        self.assertTrue(bracid.conf_matrix == correct_confusion_matrix)

    def test_add_one_best_rule_unique(self):
            """Tests that the best rule found by this function is unique and correspondingly updates relevant
            statistics if that's not the case"""
            df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                               "C": [3, 2, 1, .5, 3, 2],
                               "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
            bracid = BRACID()
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
            test_idx = -1
            classes = ["apple", "banana"]
            min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
            bracid.minority_class = "apple"
            # name=6 because this guy already exists in the rules and the new rule with name=0 becomes the same, so
            # it's removed
            correct_generalized_rule = pd.Series({"A": "low", "B": (1, 1), "C": (2.0, 3), "Class": "apple"}, name=6)
            rules = [
                pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                          name=1),
                pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                           "Class": "banana"}, name=2),
                pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                           "Class": "banana"}, name=3),
                pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                           "Class": "banana"}, name=4),
                pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                           "Class": "banana"}, name=5),
                pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2.0, upper=3),
                           "Class": "apple"}, name=6),   # same as best rule
                pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                          name=0)  # Current rule is always at the end of the list
            ]
            for rule in rules:
                rule_hash = bracid.compute_hashable_key(rule)
                bracid.unique_rules[rule_hash] = {rule.name}
            correct_generalized_rule_hash = bracid.compute_hashable_key(correct_generalized_rule)

            bracid.examples_covered_by_rule = {}
            bracid.all_rules = {0: rules[test_idx], 1: rules[0], 2: rules[1], 3: rules[2], 4: rules[3], 5: rules[4],
                                 6: rules[5]}
            bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8}
            bracid.seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}

            bracid.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
            # Note that 6: {8} is incorrect and was just added to test if the entries are merged correctly
            bracid.examples_covered_by_rule = {6: {8}}
            print("rule hashes", bracid.unique_rules)
            print(correct_generalized_rule_hash)
            bracid.closest_rule_per_example = {
                0: Data(rule_id=1, dist=0.010000000000000002),
                1: Data(rule_id=6, dist=0.0),
                2: Data(rule_id=5, dist=0.67015625),
                3: Data(rule_id=1, dist=0.038125),
                4: Data(rule_id=0, dist=0.015625),
                5: Data(rule_id=2, dist=0.67015625),
                8: Data(rule_id=6, dist=0)  # Fake entry
            }
            bracid.conf_matrix = ConfusionMatrix(TP= {0, 1}, FP= {3, 4}, TN= {2, 5}, FN= set())
            initial_f1 = 0.66666
            k = 3
            neighbors, dists, _ = bracid.find_nearest_examples(df, k, rules[test_idx], class_col_name, lookup, min_max,
                                                        classes, label_type=my_vars.SAME_LABEL_AS_RULE,
                                                        only_uncovered_neighbors=True)
            improved, updated_rules, f1 = bracid.add_one_best_rule(df, neighbors, rules[test_idx], rules, initial_f1,
                                                            class_col_name, lookup, min_max, classes)
            correct_closest_rule_per_example = {
                0: Data(rule_id=1, dist=0.010000000000000002),
                1: Data(rule_id=6, dist=0.0),
                2: Data(rule_id=5, dist=0.67015625),
                3: Data(rule_id=1, dist=0.038125),
                4: Data(rule_id=6, dist=0.015625),
                5: Data(rule_id=2, dist=0.67015625),
                8: Data(rule_id=6, dist=0)}
            self.assertTrue(improved is True)
            correct_f1 = 2 * 0.5 * 1 / 1.5
            self.assertTrue(abs(correct_f1 - f1) < my_vars.PRECISION)
            correct_confusion_matrix = ConfusionMatrix(TP= {0, 1}, FP= {3, 4}, TN= {2, 5}, FN= set())

            # Make sure confusion matrix, closest rule per example, and rule set were updated with the updated rule too
            for example_id in bracid.closest_rule_per_example:
                # 8 was only added to test something else, since it won't be in the result
                # if example_id != 8:
                    rule_id, dist = bracid.closest_rule_per_example[example_id]
                    self.assertTrue(rule_id == correct_closest_rule_per_example[example_id].rule_id and
                                    abs(dist - correct_closest_rule_per_example[example_id].dist) < 0.001)
            self.assertTrue(updated_rules[5].equals(correct_generalized_rule))
            self.assertTrue(bracid.conf_matrix == correct_confusion_matrix)
            # Duplicate rule was deleted so that the last rule now corresponds to the rule with id
            self.assertTrue(len(rules) - 1 == len(updated_rules) and updated_rules[-1].name == 6)
