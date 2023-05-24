from unittest import TestCase
from collections import Counter

import pandas as pd

from multi_imbalance.resampling.bracid.bracid import BRACID, Bounds
import multi_imbalance.resampling.bracid.vars as my_vars
from tests.resampling.bracid.classes_ import _0, _1


class TestIsDuplicate(TestCase):
    """Tests is_duplicate() from utils.py"""

    def test_is_duplicate_true(self):
        """Tests if the duplicate rule is detected"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=2), "C": Bounds(lower=1, upper=3), "Class": _0},
                      name=0),
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": _0}, name=1)
        ]
        new_rule = pd.Series({"A": "high", "B": Bounds(lower=1, upper=2), "C": Bounds(lower=1, upper=3),
                              "Class": _0}, name=2)
        bracid.all_rules = {0: rules[0], 1: rules[1]}
        rule_id = bracid.is_duplicate(new_rule, existing_rule_ids=[0, 1])
        self.assertEqual(rule_id, 0)

    def test_is_duplicate_false(self):
        """Tests if no duplicate rule is detected"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=2), "C": Bounds(lower=1, upper=3), "Class": _0},
                      name=0),
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": _0}, name=1)
        ]
        new_rule = pd.Series({"A": "high", "B": Bounds(lower=1, upper=3), "C": Bounds(lower=1, upper=3),
                              "Class": _0}, name=2)
        bracid.all_rules = {0: rules[0], 1: rules[1]}
        rule_id = bracid.is_duplicate(new_rule, existing_rule_ids=[0, 1])
        self.assertTrue(rule_id == -1)