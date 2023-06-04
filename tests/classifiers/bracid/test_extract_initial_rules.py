from unittest import TestCase

import pandas as pd

from multi_imbalance.classifiers.bracid.bracid import extract_initial_rules
from tests.classifiers.bracid.classes_ import _0, _1, _2


class TestExtractInitialRules(TestCase):
    """Tests test_extract_initial_rules() from utils"""

    def test_extract_initial_rules_numeric(self):
        """Test that rules are extracted correctly with a single numeric features"""
        df = pd.DataFrame({"A": [1.0, 2, 3], "Class": [_0, _1, _2]})
        class_col_name = "Class"
        rules = extract_initial_rules(df, class_col_name)
        correct = pd.DataFrame({"A": [(1.0, 1.0), (2, 2), (3, 3)], "Class": [_0, _1, _2]})
        self.assertEqual(df.shape, (3, 2))
        self.assertEqual(rules.shape, (3, 2))
        pd.testing.assert_frame_equal(rules, correct)

    def test_extract_initial_rules_multiple_features(self):
        """
        Test that rules are extracted correctly with different numeric and nominal features
        """
        df = pd.DataFrame({"A": [1.0, 2, 3], "C": [5, -1, 3], "Class": [_0, _1, _2]})
        class_col_name = "Class"
        rules = extract_initial_rules(df, class_col_name)
        correct = pd.DataFrame({"A": [(1.0, 1.0), (2, 2), (3, 3)], "C": [(5, 5), (-1, -1), (3, 3)], "Class": [_0, _1, _2]})

        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(rules.shape, (3, 3))
        pd.testing.assert_frame_equal(rules, correct)
