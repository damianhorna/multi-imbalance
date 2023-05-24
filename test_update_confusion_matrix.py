from unittest import TestCase

import pandas as pd

import scripts.vars as my_vars
from scripts.utils import update_confusion_matrix


class TestUpdateConfusionMatrix(TestCase):
    """Tests update_confusion_matrix() from utils.py"""

    def test_update_confusion_matrix(self):
        """Tests that TP, FN, FP, TN are updated correctly"""
        my_vars.conf_matrix = {my_vars.TP: {0}, my_vars.FP: {2}, my_vars.TN: {1}, my_vars.FN: set()}
        positive_class = "apple"
        class_col_name = "Class"
        examples = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=3),
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "banana"}, name=4),
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=5),
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "banana"}, name=6),
        ]
        rules = [
            pd.Series({"A": "low", "B": (1, 1), "C": (3, 3), "Class": "apple"}, name=0),
            pd.Series({"A": "low", "B": (1, 1), "C": (2, 2), "Class": "banana"}, name=1),
        ]
        my_vars.conf_matrix = update_confusion_matrix(examples[0], rules[0], positive_class, class_col_name,
                                                      my_vars.conf_matrix)  # TP
        my_vars.conf_matrix = update_confusion_matrix(examples[1], rules[0], positive_class, class_col_name,
                                                      my_vars.conf_matrix)  # FP
        my_vars.conf_matrix = update_confusion_matrix(examples[2], rules[1], positive_class, class_col_name,
                                                      my_vars.conf_matrix)  # FN
        my_vars.conf_matrix = update_confusion_matrix(examples[3], rules[1], positive_class, class_col_name,
                                                      my_vars.conf_matrix)  # TN
        correct = {
            my_vars.TP: {0, 3},
            my_vars.TN: {1, 6},
            my_vars.FN: {5},
            my_vars.FP: {2, 4},
        }
        self.assertTrue(correct == my_vars.conf_matrix)
