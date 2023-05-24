from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import bracid, Bounds
import scripts.vars as my_vars


class TestBracid(TestCase):
    """Tests bracid() from utils.py"""

    def test_bracid_stops(self):
        """Tests that the method stops"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
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
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        minority_label = "banana"
        k = 3
        correct_rules = {
            0: pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=3.0),
                          "Class": "apple"}, name=0),
            2: pd.Series({"B": Bounds(lower=1.25, upper=4.0), "C": Bounds(lower=0.5, upper=1.5),
                          "Class": "banana"}, name=2),
            3: pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": "banana"}, name=3),
            4: pd.Series({"B": Bounds(lower=0.5, upper=0.875), "C": Bounds(lower=2.0, upper=3.0),
                          "Class": "banana"}, name=4),
            5: pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5),
                          "Class": "banana"}, name=5),
        }
        rules = bracid(df, k, class_col_name, lookup, min_max, classes, minority_label)
        all_rules_are_equal = True
        for r in rules:
            if not rules[r].equals(correct_rules[r]):
                all_rules_are_equal = False
                break
        self.assertTrue(all_rules_are_equal)
