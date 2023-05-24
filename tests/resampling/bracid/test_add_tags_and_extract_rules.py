from unittest import TestCase
from collections import Counter

import pandas as pd

from multi_imbalance.resampling.bracid.vars import CONDITIONAL, TAG
# from scripts.utils import add_tags_and_extract_rules, Bounds
from multi_imbalance.resampling.bracid.bracid import BRACID, Bounds, ExampleClass
import multi_imbalance.resampling.bracid.vars as my_vars
from tests.resampling.bracid.classes_ import _0, _1


class TestAddTagsAndExtractRules(TestCase):
    """Tests add_tags_and_extract_rules() from utils.py"""

    def test_add_tags_all_tags(self):
        """Add tags when using nominal and numeric features and assigning noisy, borderline and safe as tags"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        bracid = BRACID()
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
        bracid.latest_rule_id = 5
        correct = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": [_0, _0, _1, _1, _1, _1],
                                TAG: [ExampleClass.BORDERLINE, ExampleClass.BORDERLINE, ExampleClass.SAFE, ExampleClass.NOISY, ExampleClass.NOISY, ExampleClass.BORDERLINE]
                                })
        classes = [_0, _1]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 2
        correct_rules = [
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
        bracid.all_rules = {}
        bracid.closest_rule_per_example = {}
        bracid.closest_examples_per_rule = {}
        bracid.seed_rule_example = {}
        bracid.seed_example_rule = {}

        correct_latest_id = 5
        correct_seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        correct_seed_example_rule = {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}}
        correct_all_rules = {0: correct_rules[0], 1: correct_rules[1], 2: correct_rules[2], 3: correct_rules[3],
                             4: correct_rules[4], 5: correct_rules[5]}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        bracid.examples_covered_by_rule = {}
        tagged, rules = bracid.add_tags_and_extract_rules(df, k, class_col_name, min_max, classes)
        pd.testing.assert_frame_equal(tagged, correct)
        self.assertEqual(bracid.seed_example_rule, correct_seed_example_rule)
        self.assertEqual(bracid.seed_rule_example, correct_seed_rule_example)
        self.assertTrue(bracid.all_rules.keys() == correct_all_rules.keys())
        self.assertEqual(bracid.latest_rule_id, correct_latest_id)
        for idx, rule in enumerate(rules):
            pd.testing.assert_series_equal(rule, correct_all_rules[idx])
