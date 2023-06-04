from unittest import TestCase

import pandas as pd

from multi_imbalance.classifiers.bracid.bracid import BRACID, Bounds, compute_hashable_key
from tests.classifiers.bracid.classes_ import _1


class TestFindDuplicateRuleId(TestCase):
    """Tests find_duplicate_rule_id() from utils.py"""

    def test_find_duplicate_rule_id(self):
        """Tests that a duplicate rule is detected properly"""
        bracid = BRACID(k=-1, minority_class=-1)
        rules = [
            pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0), "Class": _1}, name=7),
            pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0), "Class": _1}, name=12),  # Duplicate
        ]
        duplicate_idx = 1
        bracid.unique_rules = {compute_hashable_key(rules[0]): {7}}
        bracid.all_rules = {7: rules[0]}
        duplicate_hash = compute_hashable_key(rules[duplicate_idx])
        duplicate_id = bracid.find_duplicate_rule_id(rules[duplicate_idx], duplicate_hash)
        print("duplicate ID:", duplicate_id)
        self.assertTrue(duplicate_id == rules[0].name)
