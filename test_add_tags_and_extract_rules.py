from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.vars import CONDITIONAL, TAG, BORDERLINE, SAFE, NOISY
from scripts.utils import add_tags_and_extract_rules, Bounds
import scripts.vars as my_vars


class TestAddTagsAndExtractRules(TestCase):
    """Tests add_tags_and_extract_rules() from utils.py"""

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
        my_vars.latest_rule_id = 0
        correct = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "apple", "banana", "banana", "banana", "banana"],
                                TAG: [BORDERLINE, BORDERLINE, SAFE, NOISY, NOISY, BORDERLINE]
                                })
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"C": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        k = 2
        correct_rules = [
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
        my_vars.all_rules = {}
        my_vars.closest_rule_per_example = {}
        my_vars.closest_examples_per_rule = {}
        my_vars.seed_rule_example = {}
        my_vars.seed_example_rule = {}

        correct_latest_id = 5
        correct_seed_rule_example = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        correct_seed_example_rule = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        correct_all_rules = {0: correct_rules[0], 1: correct_rules[1], 2: correct_rules[2], 3: correct_rules[3],
                             4: correct_rules[4], 5: correct_rules[5]}
        # Note: examples_covered_by_rule implicitly includes the seeds of all rules
        my_vars.examples_covered_by_rule = {}
        tagged, rules = add_tags_and_extract_rules(df, k, class_col_name, lookup, min_max, classes)
        self.assertTrue(tagged.equals(correct))
        self.assertTrue(my_vars.seed_example_rule == correct_seed_example_rule)
        self.assertTrue(my_vars.seed_rule_example == correct_seed_rule_example)
        self.assertTrue(my_vars.all_rules.keys() == correct_all_rules.keys())
        self.assertTrue(my_vars.latest_rule_id == correct_latest_id)
        for idx, rule in enumerate(rules):
            self.assertTrue(rule.equals(correct_all_rules[idx]))
