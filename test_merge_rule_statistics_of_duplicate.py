from unittest import TestCase

import pandas as pd

from scripts.bracid import BRACID, Bounds, Data
import scripts.vars as my_vars
from unit_tests.classes_ import _0, _1


class TestMergeRuleStatisticsOfDuplicate(TestCase):
    """Tests merge_rule_statistics_of_duplicate from utils.py"""

    def test_merge_rule_statistics_of_duplicate(self):
        """Checks that the statistics are updated correctly if a duplicate rule is generated during the generalization
        step in bracid()"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=1),  # Duplicate
        ]

        orig_idx = 0
        dupl_idx = 1
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

        # Delete entries of the rule with ID 1 as the one with ID 0 already exists
        bracid.merge_rule_statistics_of_duplicate(rules[orig_idx], rules[dupl_idx])

        # Read: example with ID 0 is seed for the rule with ID 5....
        correct_seed_example_rule = {0: {5}, 10: {0}, 4: {7}}
        # Read: rule with ID 5 has as seed example the one with ID 0...
        correct_seed_rule_example = {5: 0, 0: 10, 7: 4}
        correct_unique_rules = {bracid.compute_hashable_key(rules[orig_idx]): {0}}
        correct_all_rules = {0: rules[orig_idx]}
        # extra_rule now also covers the 3 examples to which the 2 deleted rules were closest
        correct_closest_examples_per_rule = {0: {0, 3, 4}, 4: {8}}
        correct_closest_rule_per_example = {0: Data(rule_id=0, dist=3), 3: Data(rule_id=0, dist=2),
                                            4: Data(rule_id=0, dist=0.13), 5: Data(rule_id=76, dist=3)}
        correct_covered_by_rule = {2: {3}, 0: {43, 12, 7}}
        self.assertEqual(bracid.seed_rule_example, correct_seed_rule_example)
        self.assertEqual(bracid.seed_example_rule, correct_seed_example_rule)
        self.assertEqual(bracid.unique_rules, correct_unique_rules)
        self.assertEqual(bracid.all_rules, correct_all_rules)
        self.assertEqual(bracid.closest_examples_per_rule, correct_closest_examples_per_rule)
        self.assertEqual(bracid.closest_rule_per_example, correct_closest_rule_per_example)
        self.assertEqual(bracid.examples_covered_by_rule, correct_covered_by_rule)
