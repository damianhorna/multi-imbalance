from unittest import TestCase
import pandas as pd

from scripts.utils import compute_f1_for_predictions, Predictions
import scripts.vars as my_vars


class TestComputeF1ForPredictions(TestCase):
    """Tests test_compute_f1_for_predictions() from utils.py"""

    def test_compute_f1_for_predictions(self):
        predictions = {2: Predictions(label='apple', confidence=0.875),
                       3: Predictions(label='banana', confidence=0.6833333333333332),
                       0: Predictions(label='banana', confidence=0.9),
                       1: Predictions(label='banana', confidence=0.9),
                       4: Predictions(label='banana', confidence=0.9),
                       5: Predictions(label='apple', confidence=1.0)}
        dataset = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        class_col_name = "Class"
        minority_class = "banana"
        f1, conf_matrix = compute_f1_for_predictions(dataset, predictions, class_col_name, minority_class)
        correct_conf_matrix = {my_vars.TP: {3, 4}, my_vars.FP: {0, 1},  my_vars.FN: {2, 5}, my_vars.TN: set()}
        self.assertTrue(conf_matrix == correct_conf_matrix)
        self.assertTrue(f1 == 0.5)
