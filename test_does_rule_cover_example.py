from unittest import TestCase

import pandas as pd

# from scripts.utils import does_rule_cover_example
from scripts.bracid import BRACID


class TestDoesRuleCoverExample(TestCase):
    """Tests does_rule_cover_example() from utils.py"""

    def test_does_rule_cover_example_single(self):
        """Tests if a rule covers an example in both numeric and nominal features"""
        bracid = BRACID()
        dataset = pd.DataFrame({"A": [1.1, 2, 1.1, 1.1, 1.1, 1.1], "B": [1, 1, 2, 1, 1, 1], "C": [2, 2, 2, 3, 2, 2],
                                "D": ["x", "x", "x", "x", "y", "x"], "Class": ["A", "A", "A", "A", "A", "B"]})
        rules = [pd.Series({"A": (1.1, 1.1), "B": (1, 1), "C": (2, 2), "D": "x", "Class": "A"})]
        # This commented code does the same as the code below
        # is_covered_correct = [True, False, False, False, False, False]
        # for i, _ in dataset.iterrows():
        #     example = dataset.iloc[i]
        #     rule = rules[0]
        #     print("example\n{}\nrule:\n{}".format(example, rule))
        #     is_covered = does_rule_cover_example(example, rule, dataset.dtypes)
        #     print("result:", is_covered)
        #     print("expected:", is_covered_correct[i])
        #     self.assertTrue(is_covered_correct[i] == is_covered)
        rule = rules[0]
        dataset["is_covered"] = dataset.loc[:, :].apply(bracid.does_rule_cover_example, axis=1, args=(rule, dataset.dtypes))
        df = dataset.loc[dataset["is_covered"] == True]
        correct_row = 0
        self.assertTrue(df.index[0] == correct_row)
