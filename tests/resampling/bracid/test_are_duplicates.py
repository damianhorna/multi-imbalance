from unittest import TestCase

import pandas as pd

# from scripts.utils import _are_duplicates, Bounds
from multi_imbalance.resampling.bracid.bracid import BRACID, Bounds
from tests.resampling.bracid.classes_ import _0, _1

class TestAreDuplicates(TestCase):
    """Tests _are_duplicates() from utils.py"""

    def test_are_duplicates_bounds(self):
        """Tests that no duplicate rules are detected if they are different in a lower or upper boundary value"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1), "Class": _0},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=0.8, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": _0}, name=2)
            ]
        duplicate = bracid._are_duplicates(rules[0], rules[1])
        self.assertFalse(duplicate)

    def test_are_duplicates_true(self):
        """Tests that two rules are detected as duplicates if only the rule ID is different in both rules"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1), "Class": _0},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": _0}, name=2)
            ]
        duplicate = bracid._are_duplicates(rules[0], rules[1])
        self.assertTrue(duplicate)

    def test_are_duplicates_length(self):
        """Tests that two rules of different lengths can never be duplicates"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1), "Class": _0},
                      name=1),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": _0}, name=2)
            ]
        duplicate = bracid._are_duplicates(rules[0], rules[1])
        self.assertFalse(duplicate)