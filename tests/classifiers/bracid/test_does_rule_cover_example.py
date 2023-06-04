from unittest import TestCase

import pandas as pd

from multi_imbalance.classifiers.bracid.bracid import does_rule_cover_example
from tests.classifiers.bracid.classes_ import _0, _1


class TestDoesRuleCoverExample(TestCase):
    """Tests does_rule_cover_example() from utils.py"""

    def test_does_rule_cover_example_single(self):
        """Tests if a rule covers an example in both numeric and nominal features"""
        dataset = pd.DataFrame(
            {"A": [1.1, 2, 1.1, 1.1, 1.1, 1.1], "B": [1, 1, 2, 1, 1, 1], "C": [2, 2, 2, 3, 2, 2], "Class": [_0, _0, _0, _0, _0, _1]}
        )
        rules = [pd.Series({"A": (1.1, 1.1), "B": (1, 1), "C": (2, 2), "Class": _0})]
        rule = rules[0]
        dataset["is_covered"] = dataset.loc[:, :].apply(does_rule_cover_example, axis=1, args=(rule,))
        df = dataset.loc[dataset["is_covered"]]
        self.assertEqual(len(df.index), 2)
        self.assertEqual(df.index[0], 0)
        self.assertEqual(df.index[1], 4)
