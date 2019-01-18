from unittest import TestCase

import pandas as pd

from scripts.utils import predict, Bounds, Support, Predictions
import scripts.vars as my_vars


class TestPredict(TestCase):
    """Tests predict() from utils.py"""

    def test_predict(self):
        test_set = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                 "C": [3, 2, 1, .5, 3, 2],
                                 "Class": ["", "", "", "", "", ""]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        labels = ["apple", "banana"]
        class_col_name = "Class"
        my_vars.minority_class = labels[0]
        rules = {
            2: pd.Series({"B": Bounds(lower=1.25, upper=4.0), "C": Bounds(lower=0.5, upper=1.5),
                          "Class": "banana"}, name=2),
            6: pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": "banana"}, name=6),
            5: pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5),
                          "Class": "banana"}, name=5),
            0: pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": "apple"}, name=0),
        }
        model = {2: Support(minority=0.75, majority=0.25), 6: Support(minority=0.2, majority=0.8),
                 5: Support(minority=1.0, majority=0.0), 0: Support(minority=0, majority=1)}
        correct_predictions = {2: Predictions(label='apple', confidence=0.875),
                               3: Predictions(label='banana', confidence=0.6833333333333332),
                               0: Predictions(label='banana', confidence=0.9),
                               1: Predictions(label='banana', confidence=0.9),
                               4: Predictions(label='banana', confidence=0.9),
                               5: Predictions(label='apple', confidence=1.0)}

        preds = predict(model, test_set, rules, labels, class_col_name)
        self.assertTrue(correct_predictions == preds)
