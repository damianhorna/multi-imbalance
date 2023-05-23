from unittest import TestCase

import pandas as pd

import scripts.vars as my_vars
from scripts.bracid import BRACID, ConfusionMatrix


class TestUpdateConfusionMatrix(TestCase):
    """Tests update_confusion_matrix() from utils.py"""

    def test_update_confusion_matrix(self):
        """Tests that TP, FN, FP, TN are updated correctly"""
        bracid = BRACID()
        bracid.conf_matrix = ConfusionMatrix(TP= {0}, FP= {2}, TN= {1}, FN= set())
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
        bracid.conf_matrix = bracid.update_confusion_matrix(examples[0], rules[0], positive_class, class_col_name,
                                                      bracid.conf_matrix)  # TP
        bracid.conf_matrix = bracid.update_confusion_matrix(examples[1], rules[0], positive_class, class_col_name,
                                                      bracid.conf_matrix)  # FP
        bracid.conf_matrix = bracid.update_confusion_matrix(examples[2], rules[1], positive_class, class_col_name,
                                                      bracid.conf_matrix)  # FN
        bracid.conf_matrix = bracid.update_confusion_matrix(examples[3], rules[1], positive_class, class_col_name,
                                                      bracid.conf_matrix)  # TN
        correct = ConfusionMatrix(
            TP= {0, 3},
            TN= {1, 6},
            FN= {5},
            FP= {2, 4},
        )
        self.assertEqual(correct, bracid.conf_matrix)
