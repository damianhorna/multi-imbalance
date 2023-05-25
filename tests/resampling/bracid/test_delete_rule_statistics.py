from unittest import TestCase
from collections import Counter

import pandas as pd

# from scripts.utils import delete_rule_statistics, Bounds, compute_hashable_key, Data
from multi_imbalance.resampling.bracid.bracid import BRACID, Bounds, Data
import multi_imbalance.resampling.bracid.vars as my_vars
from tests.resampling.bracid.classes_ import _0, _1


class TestDeleteRuleStatistics(TestCase):
    """Tests delete_rule_statistics() from utils.py"""

    def test_delete_rule_statistics_unique_hash(self):
        """Deletes a rule with a unique hash"""
        bracid = BRACID()
        extra_rule = pd.Series({"B": Bounds(lower=0.1, upper=1), "C": Bounds(lower=1, upper=2),
                                "Class": _0}, name=4)
        rules = [
            extra_rule,
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0},
                      name=1),
        ]
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _0, _0, _0, _0]})
        class_col_name = "Class"
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 0.1, "max": 1}, "C": {"min": 1, "max": 3}})
        bracid.minority_class = _0

        for rule in rules:
            hash_val = bracid.compute_hashable_key(rule)
            bracid.unique_rules.setdefault(hash_val, set()).add(rule.name)
            bracid.all_rules[rule.name] = rule
        # Some random values
        bracid.seed_example_rule = {0: {1, 5}, 10: {0}, 4: {7}}
        bracid.seed_rule_example = {5: 0, 1: 0, 0: 10, 7: 4}
        bracid.closest_examples_per_rule = {0: {0, 3}, 1: {4}, 4: {8}}
        bracid.closest_rule_per_example = {0: Data(rule_id=0, dist=3), 3: Data(rule_id=0, dist=2),
                                            4: Data(rule_id=1, dist=0.13), 5: Data(rule_id=76, dist=3)}
        bracid.examples_covered_by_rule = {0: {43, 12}, 1: {7}, 2: {3}}
        final_rules = {}
        # Delete entries for rules with IDs 0 and 1 from all statistics
        rule1 = rules.pop()
        bracid.delete_rule_statistics(df, rule1, rules, final_rules, class_col_name, min_max, classes)
        rule2 = rules.pop()
        bracid.delete_rule_statistics(df, rule2, rules, final_rules,class_col_name, min_max, classes)

        correct_seed_example_rule = {4: {7}}
        correct_seed_rule_example = {5: 0, 7: 4}
        correct_unique_rules = {bracid.compute_hashable_key(extra_rule): {4}}
        correct_all_rules = {4: extra_rule}
        # extra_rule now also covers the 3 examples to which the 2 deleted rules were closest
        correct_closest_examples_per_rule = {4: {8, 0, 3, 4}}
        correct_closest_rule_per_example = {5: Data(rule_id=76, dist=3), 4: Data(rule_id=4, dist=0.25),
                                            0: Data(rule_id=4, dist=0.25), 3: Data(rule_id=4, dist=0.371141975308642)}
        correct_covered_by_rule = {2: {3}}
        self.assertEqual(bracid.seed_rule_example, correct_seed_rule_example)
        self.assertEqual(bracid.seed_example_rule, correct_seed_example_rule)
        self.assertEqual(bracid.unique_rules, correct_unique_rules)
        self.assertEqual(bracid.all_rules, correct_all_rules)
        self.assertEqual(bracid.closest_examples_per_rule, correct_closest_examples_per_rule)
        self.assertEqual(bracid.closest_rule_per_example, correct_closest_rule_per_example)
        self.assertEqual(bracid.examples_covered_by_rule, correct_covered_by_rule)

    def test_delete_rule_statistics_collision(self):
        """Deletes a rule that shares its hash with other rules"""
        bracid = BRACID()
        extra_rule = pd.Series({"B": Bounds(lower=0.1, upper=1), "C": Bounds(lower=1, upper=2),
                                "Class": _0}, name=4)
        rules = [
            extra_rule,
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=1),  # Duplicate
        ]
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _0, _0, _0, _0]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 1,
                        'low': 2,
                        my_vars.CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        _0: 1
                                    }),
                                'low':
                                    Counter({
                                        _0: 2
                                    })
                            }
                    }
            }
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 0.1, "max": 1}, "C": {"min": 1, "max": 3}})
        bracid.minority_class = _0
        bracid.unique_rules = {}
        bracid.all_rules = {}
        for rule in rules:
            hash_val = bracid.compute_hashable_key(rule)
            bracid.unique_rules.setdefault(hash_val, set()).add(rule.name)
            bracid.all_rules[rule.name] = rule
        print("hashes", bracid.unique_rules)
        # Some random values
        bracid.seed_example_rule = {0: {1, 5}, 10: {0}, 4: {7}}
        bracid.seed_rule_example = {5: 0, 1: 0, 0: 10, 7: 4}
        bracid.closest_examples_per_rule = {0: {0, 3}, 1: {4}, 4: {8}}
        bracid.closest_rule_per_example = {0: Data(rule_id=0, dist=3), 3: Data(rule_id=0, dist=2),
                                            4: Data(rule_id=1, dist=0.13), 5: Data(rule_id=76, dist=3)}
        bracid.examples_covered_by_rule = {0: {43, 12}, 1: {7}, 2: {3}}
        final_rules = {}
        # Delete entries for rules with IDs 0 and 1 from all statistics
        rule1 = rules.pop()
        bracid.delete_rule_statistics(df, rule1, rules, final_rules, class_col_name, min_max, classes)
        rule2 = rules.pop()
        bracid.delete_rule_statistics(df, rule2, rules, final_rules, class_col_name, min_max, classes)

        correct_seed_example_rule = {4: {7}}
        correct_seed_rule_example = {5: 0, 7: 4}
        correct_unique_rules = {bracid.compute_hashable_key(extra_rule): {4}}
        correct_all_rules = {4: extra_rule}
        # extra_rule now also covers the 3 examples to which the 2 deleted rules were closest
        correct_closest_examples_per_rule = {4: {8, 0, 3, 4}}
        correct_closest_rule_per_example = {5: Data(rule_id=76, dist=3), 4: Data(rule_id=4, dist=0.25),
                                            0: Data(rule_id=4, dist=0.25), 3: Data(rule_id=4, dist=0.371141975308642)}
        correct_covered_by_rule = {2: {3}}
        self.assertEqual(bracid.seed_rule_example, correct_seed_rule_example)
        self.assertEqual(bracid.seed_example_rule, correct_seed_example_rule)
        self.assertEqual(bracid.unique_rules, correct_unique_rules)
        self.assertEqual(bracid.all_rules, correct_all_rules)
        self.assertEqual(bracid.closest_examples_per_rule, correct_closest_examples_per_rule)
        self.assertEqual(bracid.closest_rule_per_example, correct_closest_rule_per_example)
        self.assertEqual(bracid.examples_covered_by_rule, correct_covered_by_rule)
