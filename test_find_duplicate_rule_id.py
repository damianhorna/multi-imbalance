from unittest import TestCase

import pandas as pd

from scripts.bracid import find_duplicate_rule_id, Bounds, compute_hashable_key
import scripts.vars as my_vars
from unit_tests.classes_ import _0, _1


class TestFindDuplicateRuleId(TestCase):
    """Tests find_duplicate_rule_id() from utils.py"""

    def test_find_duplicate_rule_id(self):
        """Tests that a duplicate rule is detected properly"""
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                       "Class": _1}, name=7),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                       "Class": _1}, name=12)  # Duplicate
        ]
        duplicate_idx = 1
        bracid.unique_rules = {compute_hashable_key(rules[0]): {7}}
        bracid.all_rules = {7: rules[0]}
        duplicate_hash = compute_hashable_key(rules[duplicate_idx])
        duplicate_id = find_duplicate_rule_id(rules[duplicate_idx], duplicate_hash)
        print("duplicate ID:", duplicate_id)
        self.assertTrue(duplicate_id == rules[0].name)
