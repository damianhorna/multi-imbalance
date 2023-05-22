from unittest import TestCase

import pandas as pd

# from scripts.utils import _are_duplicates, Bounds
from scripts.bracid import BRACID, Bounds


class TestAreDuplicates(TestCase):
    """Tests _are_duplicates() from utils.py"""

    def test_are_duplicates_bounds(self):
        """Tests that no duplicate rules are detected if they are different in a lower or upper boundary value"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=0.8, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": "apple"}, name=2)
            ]
        duplicate = bracid._are_duplicates(rules[0], rules[1])
        self.assertTrue(duplicate is False)

    def test_are_duplicates_nominal(self):
        """Tests that no duplicate rules are detected if they are different in a nominal feature"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2)
        ]
        duplicate = bracid._are_duplicates(rules[0], rules[1])
        self.assertTrue(duplicate is False)

    def test_are_duplicates_true(self):
        """Tests that two rules are detected as duplicates if only the rule ID is different in both rules"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": "apple"}, name=2)
            ]
        duplicate = bracid._are_duplicates(rules[0], rules[1])
        self.assertTrue(duplicate is True)

    def test_are_duplicates_length(self):
        """Tests that two rules of different lengths can never be duplicates"""
        bracid = BRACID()
        rules = [
            pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1), "Class": "apple"},
                      name=1),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": "apple"}, name=2)
            ]
        duplicate = bracid._are_duplicates(rules[0], rules[1])
        self.assertTrue(duplicate is False)
