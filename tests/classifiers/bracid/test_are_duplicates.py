from unittest import TestCase

import pandas as pd

from multi_imbalance.classifiers.bracid.bracid import Bounds, _are_duplicates
from tests.classifiers.bracid.classes_ import _0


class TestAreDuplicates(TestCase):
    """Tests _are_duplicates() from utils.py"""

    def test_are_duplicates_bounds(self):
        """Tests that no duplicate rules are detected if they are different in a lower or upper boundary value"""
        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1), "Class": _0},
                      name=1),
            pd.Series({"B": Bounds(lower=0.8, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": _0}, name=2)
            ]
        duplicate = _are_duplicates(rules[0], rules[1])
        self.assertFalse(duplicate)

    def test_are_duplicates_true(self):
        """Tests that two rules are detected as duplicates if only the rule ID is different in both rules"""
        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1), "Class": _0},
                      name=1),
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1),
                       "Class": _0}, name=2)
            ]
        duplicate = _are_duplicates(rules[0], rules[1])
        self.assertTrue(duplicate)

    def test_are_duplicates_length(self):
        """Tests that two rules of different lengths can never be duplicates"""
        rules = [
            pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=1, upper=1), "Class": _0},
                      name=1),
            pd.Series({"B": Bounds(lower=1, upper=1), "Class": _0}, name=2)
            ]
        duplicate = _are_duplicates(rules[0], rules[1])
        self.assertFalse(duplicate)
