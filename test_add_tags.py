from unittest import TestCase
from collections import Counter

import pandas as pd
import numpy as np

from scripts.vars import CONDITIONAL, TAG, BORDERLINE, SAFE, NOISY
from scripts.utils import add_tags, Bounds
import scripts.vars as my_vars


class TestAddTags(TestCase):
    """Tests add_tags() from utils.py"""

    def test_add_tags_safe_borderline(self):
        """Add tags when using nominal and numeric features assigning borderline and safe as tags"""
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
                        CONDITIONAL:
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
        correct = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "apple", "banana", "banana", "banana", "banana"],
                                TAG: [BORDERLINE, BORDERLINE, SAFE, BORDERLINE, BORDERLINE, BORDERLINE]
                                })
        my_vars.closest_rule_per_example = {}
        my_vars.closest_examples_per_rule = {}
        my_vars.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        my_vars.seed_example_rule = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        my_vars.examples_covered_by_rule = {}
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 3
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5)
        ]
        my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        tagged = add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        self.assertTrue(tagged.equals(correct))

    def test_add_tags_noisy_safe(self):
        """Add tags when using nominal and numeric features and assigning noisy and safe as tags"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "banana", "banana", "banana", "banana", "banana"]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 2,
                        'low': 4,
                        CONDITIONAL:
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
        correct = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "banana", "banana", "banana", "banana", "banana"],
                                TAG: [NOISY, BORDERLINE, SAFE, SAFE, SAFE, SAFE]
                                })
        classes = ["apple", "banana"]
        my_vars.closest_rule_per_example = {}
        my_vars.closest_examples_per_rule = {}
        my_vars.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        my_vars.seed_example_rule = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        my_vars.examples_covered_by_rule = {}
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 3
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5)
        ]
        my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        tagged = add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        self.assertTrue(tagged.equals(correct))

    def test_add_tags_nan(self):
        """Add tags when using nominal and numeric features when all examples contain at least one NaN value"""
        df = pd.DataFrame({"A": [np.NaN, np.NaN, "high", np.NaN, "low", np.NaN],
                           "B": [np.NaN, 1, np.NaN, 1.5, np.NaN, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        my_vars.examples_covered_by_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 1,
                        'low': 1,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        'banana': 1
                                    }),
                                'low':
                                    Counter({
                                        'banana': 1
                                    })
                            }
                    }
            }
        correct = pd.DataFrame({"A": [np.NaN, np.NaN, "high", np.NaN, "low", np.NaN],
                                "B": [np.NaN, 1, np.NaN, 1.5, np.NaN, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "apple", "banana", "banana", "banana", "banana"],
                                TAG: [BORDERLINE, NOISY, SAFE, SAFE, SAFE, SAFE]
                                })
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5)
        ]
        k = 3
        my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        my_vars.closest_rule_per_example = {}
        my_vars.closest_examples_per_rule = {}
        my_vars.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        my_vars.seed_example_rule = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        my_vars.examples_covered_by_rule = {}
        tagged = add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        self.assertTrue(tagged.equals(correct))

    def test_add_tags_all_tags(self):
        """Add tags when using nominal and numeric features and assigning noisy, borderline and safe as tags"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        class_col_name = "Class"
        my_vars.examples_covered_by_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        lookup = \
            {
                "A":
                    {
                        'high': 2,
                        'low': 4,
                        CONDITIONAL:
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
        correct = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "apple", "banana", "banana", "banana", "banana"],
                                TAG: [BORDERLINE, BORDERLINE, SAFE, NOISY, NOISY, BORDERLINE]
                                })
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 2
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": "banana"}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": "banana"}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5)
        ]
        my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        my_vars.closest_rule_per_example = {}
        my_vars.closest_examples_per_rule = {}
        my_vars.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        my_vars.seed_example_rule = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        my_vars.examples_covered_by_rule = {}
        tagged = add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        self.assertTrue(tagged.equals(correct))
