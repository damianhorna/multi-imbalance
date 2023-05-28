from unittest import TestCase

import numpy as np
import pandas as pd

from multi_imbalance.classifiers.bracid.bracid import BRACID, Bounds
import multi_imbalance.classifiers.bracid.vars as my_vars
from tests.classifiers.bracid.classes_ import _0, _1, _2


class TestBracid(TestCase):
    """Tests bracid() from utils.py"""

    def test_bracid_stops(self):
        """Tests that the method stops"""
        k = 3
        minority_label = _1
        bracid = BRACID(k=k, minority_class=minority_label)
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        correct_rules = {
            0: pd.Series({"B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=3.0),
                          "Class": _0}, name=0),
            2: pd.Series({"B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=0.5, upper=2.5),
                          "Class": _1}, name=2),
            3: pd.Series({"B": Bounds(lower=0.75, upper=1.5), "C": Bounds(lower=0.5, upper=2.5),
                          "Class": _1}, name=3),
            4: pd.Series({"B": Bounds(lower=0.5, upper=0.75), "C": Bounds(lower=2.5, upper=3.0),
                          "Class": _1}, name=4),
            5: pd.Series({"B": Bounds(lower=0.75, upper=0.875), "C": Bounds(lower=2.0, upper=2.5),
                          "Class": _1}, name=5),
            6: pd.Series({"B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5), "Class": _1}, name=6)
        }
        rules = bracid.bracid(df, k, class_col_name, min_max, classes, minority_label)
        for key, rule in rules.items():
            with self.subTest(f'rule_{key}'):
                expected = correct_rules[key]
                pd.testing.assert_series_equal(expected, rule.drop(my_vars.HASH, errors='ignore'))

    def test_bracid_fit_predict(self):
        """Tests that the bracid.fit() stops"""
        k = 3
        with self.subTest("binary_task"):
            minority_label = _1
            bracid = BRACID(k=k, minority_class=minority_label)
            X = np.array([
                [1, 1, 4, 1.5, 0.5, 0.75],
                [3, 2, 1, .5, 3, 2],
            ]).T
            y = [_0, _0, _1, _1, _1, _1]
            bracid.fit(X, y)
            prediction = bracid.predict(X)
            predict_proba = bracid.predict_proba(X)
            self.assertTrue(bracid._is_binary_classification)
            self.assertEqual(np.asarray(y).shape, prediction.shape)
            np.testing.assert_array_less(predict_proba, 1.0)
            np.testing.assert_array_less(0.0, predict_proba)

        with self.subTest("multiclass_task"):
            bracid = BRACID(k=k, minority_class=None)
            y = [_0, _0, _1, _1, _1, _2]
            bracid.fit(X, y)
            prediction = bracid.predict(X)
            predict_proba = bracid.predict_proba(X)
            self.assertFalse(bracid._is_binary_classification)
            self.assertEqual(np.asarray(y).shape, prediction.shape)
            np.testing.assert_array_less(predict_proba, 1.0 + 1e-5)
            np.testing.assert_array_less(0.0 - 1e-5, predict_proba)