from unittest import TestCase
from collections import Counter

import pandas as pd

from multi_imbalance.resampling.bracid.bracid import BRACID, Bounds
from multi_imbalance.resampling.bracid import vars as my_vars
from tests.resampling.bracid.classes_ import _0, _1
import pytest

class TestBracid(TestCase):
    """Tests bracid() from utils.py"""

    def test_bracid_stops(self):
        """Tests that the method stops"""
        bracid = BRACID()
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
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
                                        _1: 2
                                    }),
                                'low':
                                    Counter({
                                        _1: 2,
                                        _0: 2
                                    })
                            }
                    }
            }
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        minority_label = _1
        k = 3
        correct_rules = {
            0: pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=3.0),
                          "Class": _0}, name=0),
            2: pd.Series({"B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=0.5, upper=2.5),
                          "Class": _1}, name=2),
            3: pd.Series({"B": Bounds(lower=0.75, upper=1.5), "C": Bounds(lower=0.5, upper=2.5),
                          "Class": _1}, name=3),
            4: pd.Series({"B": Bounds(lower=0.5, upper=0.75), "C": Bounds(lower=2.5, upper=3.0),
                          "Class": _1}, name=4),
            5: pd.Series({"B": Bounds(lower=0.75, upper=0.875), "C": Bounds(lower=2.0, upper=2.5),
                          "Class": _1}, name=5),
            6: pd.Series({"B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5), "Class": _1}, name=6)
        }
        rules = bracid.bracid(df, k, class_col_name, min_max, classes, minority_label)
        for key, rule in rules.items():
            with self.subTest(f'rule_{key}'):
                expected = correct_rules[key]
                pd.testing.assert_series_equal(expected, rule)
