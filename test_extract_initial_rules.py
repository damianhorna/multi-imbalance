from unittest import TestCase

import pandas as pd

from scripts.utils import extract_initial_rules


class TestExtractInitialRules(TestCase):
    """Tests test_extract_initial_rules() from utils"""

    def test_extract_initial_rules_numeric(self):
        """Test that rules are extracted correctly with a single numeric features"""
        df = pd.DataFrame({"A": [1.0, 2, 3], "Class": ["A", "B", "C"]})
        class_col_name = "Class"
        rules = extract_initial_rules(df, class_col_name)
        correct = pd.DataFrame({"A": [(1.0, 1.0), (2, 2), (3, 3)], "Class": ["A", "B", "C"]})
        self.assertTrue(df.shape == (3, 2) and rules.shape == (3, 2))
        self.assertTrue(rules.equals(correct))

    def test_extract_initial_rules_nominal(self):
        """Test that rules are extracted correctly with a single nominal features"""
        df = pd.DataFrame({"A": ["a", "b", "c"], "Class": ["A", "B", "C"]})
        class_col_name = "Class"
        rules = extract_initial_rules(df, class_col_name)
        correct = pd.DataFrame({"A": ["a", "b", "c"], "Class": ["A", "B", "C"]})
        self.assertTrue(df.shape == (3, 2) and rules.shape == (3, 2))
        self.assertTrue(rules.equals(correct))

    def test_extract_initial_rules_single_feature_mixed(self):
        """
        Test that rules are extracted correctly with a single numeric and nominal feature
        """
        df = pd.DataFrame({"A": [1.0, 2, 3], "B": ["a", "b", "c"], "Class": ["A", "B", "C"]})
        class_col_name = "Class"
        rules = extract_initial_rules(df, class_col_name)
        correct = pd.DataFrame({"A": [(1.0, 1.0), (2, 2), (3, 3)], "B": ["a", "b", "c"], "Class": ["A", "B", "C"]})
        self.assertTrue(df.shape == (3, 3) and rules.shape == (3, 3))
        self.assertTrue(rules.equals(correct))

    def test_extract_initial_rules_multiple_features_mixed(self):
        """
        Test that rules are extracted correctly with different numeric and nominal features
        """
        df = pd.DataFrame({"A": [1.0, 2, 3], "B": ["a", "b", "c"], "C": [5, -1, 3], "D": ["t", "t", "e"],
                           "Class": ["A", "B", "C"]})
        class_col_name = "Class"
        rules = extract_initial_rules(df, class_col_name)
        correct = pd.DataFrame({"A": [(1.0, 1.0), (2, 2), (3, 3)], "B": ["a", "b", "c"],
                                "C": [(5, 5), (-1, -1), (3, 3)], "D": ["t", "t", "e"], "Class": ["A", "B", "C"]})
        self.assertTrue(df.shape == (3, 5) and rules.shape == (3, 5))
        self.assertTrue(rules.equals(correct))
