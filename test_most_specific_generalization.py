from unittest import TestCase

import pandas as pd

from scripts.bracid import BRACID


class TestMostSpecificGeneralization(TestCase):
    """Tests most_specific_generalization() from utils.py"""

    def test_most_specific_generalization_no_change(self):
        """No generalization for nominal and numeric features, i.e. rule remains the same"""
        bracid = BRACID()
        dataset = pd.DataFrame({"A": [1.1], "B": [1], "C": [2], "D": ["x"], "Class": ["A"]})
        class_col_name = "Class"
        rules = [pd.Series({"A": (1.1, 1.1), "B": (1, 1), "C": (2, 2), "D": "x", "Class": "A"})]
        correct = rules[0].copy()
        for i, _ in dataset.iterrows():
            nearest_example = dataset.iloc[i]
            # print("example", type(nearest_example), nearest_example, type(nearest_example["A"]), nearest_example["A"])
            rule = rules[i]
            rule = bracid.most_specific_generalization(nearest_example, rule, class_col_name, dataset.dtypes)
            self.assertTrue(rule.equals(correct))

    def test_most_specific_generalization_drop_nominal(self):
        """Generalization for nominal -> feature is dropped"""
        bracid = BRACID()
        dataset = pd.DataFrame({"A": [1.1], "B": [1], "C": [2], "D": ["x"], "Class": ["A"]})
        class_col_name = "Class"
        rules = [pd.Series({"A": (1.1, 1.1), "B": (1, 1), "C": (2, 2), "D": "y", "Class": "A"})]
        correct = pd.Series({"A": (1.1, 1.1), "B": (1, 1), "C": (2, 2), "Class": "A"})
        for i, _ in dataset.iterrows():
            nearest_example = dataset.iloc[i]
            rule = rules[i]
            rule = bracid.most_specific_generalization(nearest_example, rule, class_col_name, dataset.dtypes)
            self.assertTrue(rule.equals(correct))

    def test_most_specific_generalization_change_lower(self):
        """Generalization for numeric feature -> lower bound is updated"""
        bracid = BRACID()
        lower_bound = 0
        dataset = pd.DataFrame({"A": [lower_bound], "B": [1], "C": [2], "D": ["x"], "Class": ["A"]})
        class_col_name = "Class"
        rules = [pd.Series({"A": (0.1, 1), "B": (1, 1), "C": (2, 2), "D": "x", "Class": "A"})]
        correct = pd.Series({"A": (lower_bound, 1), "B": (1, 1), "C": (2, 2), "D": "x", "Class": "A"})
        for i, _ in dataset.iterrows():
            nearest_example = dataset.iloc[i]
            rule = rules[i]
            rule = bracid.most_specific_generalization(nearest_example, rule, class_col_name, dataset.dtypes)
            self.assertTrue(rule.equals(correct))

    def test_most_specific_generalization_change_upper(self):
        """Generalization for numeric feature -> upper bound is updated"""
        bracid = BRACID()
        upper_bound = 2.0
        dataset = pd.DataFrame({"A": [upper_bound], "B": [1], "C": [2], "D": ["x"], "Class": ["A"]})
        class_col_name = "Class"
        rules = [pd.Series({"A": (0.1, 1), "B": (1, 1), "C": (2, 2), "D": "x", "Class": "A"})]
        correct = pd.Series({"A": (0.1, upper_bound), "B": (1, 1), "C": (2, 2), "D": "x", "Class": "A"})
        for i, _ in dataset.iterrows():
            nearest_example = dataset.iloc[i]
            rule = rules[i]
            rule = bracid.most_specific_generalization(nearest_example, rule, class_col_name, dataset.dtypes)
            self.assertTrue(rule.equals(correct))

    def test_most_specific_generalization_change_multiple(self):
        """Generalization for using all cases at the same time for numeric and nominal features"""
        bracid = BRACID()
        lower_bound = 0.5
        upper_bound = 2.0
        dataset = pd.DataFrame({"A": [upper_bound], "B": [lower_bound], "C": [lower_bound], "D": ["x"], "Class": ["A"]})
        class_col_name = "Class"
        rules = [pd.Series({"A": (0.1, 1), "B": (1, 1), "C": (2, 2), "D": "y", "Class": "A"})]
        correct = pd.Series({"A": (0.1, upper_bound), "B": (lower_bound, 1), "C": (lower_bound, 2), "Class": "A"})
        for i, _ in dataset.iterrows():
            nearest_example = dataset.iloc[i]
            rule = rules[i]
            rule = bracid.most_specific_generalization(nearest_example, rule, class_col_name, dataset.dtypes)
            self.assertTrue(rule.equals(correct))

    def test_most_specific_generalization_multiple_rules(self):
        """Generalization for using all cases at the same time for numeric and nominal features with multiple rules"""
        bracid = BRACID()
        lower_bound = 0.5
        upper_bound = 2.0
        dataset = pd.DataFrame({"A": [upper_bound, lower_bound], "B": [lower_bound, upper_bound],
                                "C": [lower_bound, upper_bound], "D": ["x", "y"], "Class": ["A", "A"]})
        class_col_name = "Class"
        rules = [pd.Series({"A": (0.1, 1), "B": (1, 1), "C": (2, 2), "D": "y", "Class": "A"}),
                 pd.Series({"A": (0.1, 1), "B": (1, 1), "C": (2, 2), "D": "y", "Class": "A"})]
        correct = [pd.Series({"A": (0.1, upper_bound), "B": (lower_bound, 1), "C": (lower_bound, 2), "Class": "A"}),
                   pd.Series({"A": (0.1, 1), "B": (1, upper_bound), "C": (2, 2), "D": "y", "Class": "A"})]
        for i, _ in dataset.iterrows():
            nearest_example = dataset.iloc[i]
            rule = rules[i]
            updated_rule = bracid.most_specific_generalization(nearest_example, rule, class_col_name, dataset.dtypes)
            self.assertTrue(updated_rule.equals(correct[i]))
