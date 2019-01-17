from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import extend_rule, Bounds
import scripts.vars as my_vars


class TestExtendRule(TestCase):
    """Tests extend_rule() from utils.py"""

    def test_extend_rule_mixed(self):
        """Test that a rule containing nominal and numeric features is extended correctly"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3.1, 3.2],
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
        my_vars.minority_class = "apple"
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1), "Class": "banana"},
                      name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5)
        ]

        k = 3
        # Reset from previous test to make sure they don't affect the outcomes of this test
        my_vars.closest_examples_per_rule = {}
        my_vars.closest_rule_per_example = {}
        my_vars.examples_covered_by_rule = {}
        extended_rule = extend_rule(df, k, rules[0], class_col_name, lookup, min_max, classes)
        correct_rule = pd.Series({"A": "low", "B": (0.875, 1.25), "C": (1.75, 3.05), "Class": "apple"}, name=0)
        print(extended_rule)
        self.assertTrue(extended_rule.equals(correct_rule))

    def test_extend_rule_no_change(self):
        """Test that a rule containing nominal and numeric features isn't extended due to no neighbors"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 1, 1, 1, 1],
                           "C": [3, 2, 3, 3, 3, 3],
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
        my_vars.minority_class = "apple"
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1), "Class": "banana"},
                      name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"},  name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3), "Class": "banana"},
                      name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5)
        ]
        my_vars.closest_examples_per_rule = {}
        my_vars.closest_rule_per_example = {}
        k = 3
        extended_rule = extend_rule(df, k, rules[0], class_col_name, lookup, min_max, classes)
        correct_rule = pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0)
        self.assertTrue(extended_rule.equals(correct_rule))
