from unittest import TestCase
from collections import Counter

import pandas as pd
import numpy as np

from scripts.vars import CONDITIONAL, TAG, BORDERLINE, SAFE, NOISY
from scripts.bracid import BRACID, Bounds
import scripts.vars as my_vars
from unit_tests.classes_ import _0, _1


class TestAddTags(TestCase):
    """Tests add_tags() from utils.py"""

    def test_add_tags_safe_borderline(self):
        """Add tags when using nominal and numeric features assigning borderline and safe as tags"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        bracid = BRACID()
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
        correct = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": [_0, _0, _1, _1, _1, _1],
                                TAG: [BORDERLINE, BORDERLINE, SAFE, BORDERLINE, BORDERLINE, BORDERLINE]
                                })
        bracid.closest_rule_per_example = {}
        bracid.closest_examples_per_rule = {}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        bracid.examples_covered_by_rule = {}
        classes = [_0, _1]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 3
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": _1}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": _1}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": _1}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": _1}, name=5)
        ]
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        tagged = bracid.add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        pd.testing.assert_frame_equal(tagged, correct)

    def test_add_tags_noisy_safe(self):
        """Add tags when using nominal and numeric features and assigning noisy and safe as tags"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _1, _1, _1, _1, _1]})
        class_col_name = "Class"
        bracid = BRACID()
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
        correct = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": [_0, _1, _1, _1, _1, _1],
                                TAG: [NOISY, BORDERLINE, SAFE, SAFE, SAFE, SAFE]
                                })
        classes = [_0, _1]
        bracid.closest_rule_per_example = {}
        bracid.closest_examples_per_rule = {}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        bracid.examples_covered_by_rule = {}
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 3
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": _1}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": _1}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": _1}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": _1}, name=5)
        ]
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        tagged = bracid.add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        pd.testing.assert_frame_equal(tagged, correct)

    def test_add_tags_nan(self):
        """Add tags when using nominal and numeric features when all examples contain at least one NaN value"""
        df = pd.DataFrame({"A": [np.NaN, np.NaN, "high", np.NaN, "low", np.NaN],
                           "B": [np.NaN, 1, np.NaN, 1.5, np.NaN, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        bracid.examples_covered_by_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        bracid = BRACID()
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
                                        _1: 1
                                    }),
                                'low':
                                    Counter({
                                        _1: 1
                                    })
                            }
                    }
            }
        correct = pd.DataFrame({"A": [np.NaN, np.NaN, "high", np.NaN, "low", np.NaN],
                                "B": [np.NaN, 1, np.NaN, 1.5, np.NaN, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": [_0, _0, _1, _1, _1, _1],
                                TAG: [BORDERLINE, NOISY, SAFE, SAFE, SAFE, SAFE]
                                })
        classes = [_0, _1]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": _1}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": _1}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": _1}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": _1}, name=5)
        ]
        k = 3
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.closest_rule_per_example = {}
        bracid.closest_examples_per_rule = {}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        bracid.examples_covered_by_rule = {}
        tagged = bracid.add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        # Due to floating point precision, use approximate comparison
        pd.testing.assert_frame_equal(tagged, correct)

    def test_add_tags_all_tags(self):
        """Add tags when using nominal and numeric features and assigning noisy, borderline and safe as tags"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        bracid.examples_covered_by_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
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
        correct = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": [_0, _0, _1, _1, _1, _1],
                                TAG: [BORDERLINE, BORDERLINE, SAFE, NOISY, NOISY, BORDERLINE]
                                })
        classes = [_0, _1]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 2
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": _0},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": _0},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                       "Class": _1}, name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": _1}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                       "Class": _1}, name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": _1}, name=5)
        ]
        bracid.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        bracid.closest_rule_per_example = {}
        bracid.closest_examples_per_rule = {}
        bracid.seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        bracid.seed_example_rule = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        bracid.examples_covered_by_rule = {}
        tagged = bracid.add_tags(df, k, rules, class_col_name, lookup, min_max, classes)
        pd.testing.assert_frame_equal(tagged, correct)
