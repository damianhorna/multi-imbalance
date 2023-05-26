from unittest import TestCase

import pandas as pd

from tests.classifiers.bracid.classes_ import _0, _1
from multi_imbalance.classifiers.bracid.bracid import ConfusionMatrix, update_confusion_matrix


class TestUpdateConfusionMatrix(TestCase):
    """Tests update_confusion_matrix() from utils.py"""

    def test_update_confusion_matrix(self):
        """Tests that TP, FN, FP, TN are updated correctly"""
        conf_matrix = ConfusionMatrix(TP={0}, FP={2}, TN={1}, FN=set())
        positive_class = _0
        class_col_name = "Class"
        examples = [
            pd.Series({"B": (1, 1), "C": (3, 3), "Class": _0}, name=3),
            pd.Series({"B": (1, 1), "C": (3, 3), "Class": _1}, name=4),
            pd.Series({"B": (1, 1), "C": (3, 3), "Class": _0}, name=5),
            pd.Series({"B": (1, 1), "C": (3, 3), "Class": _1}, name=6),
        ]
        rules = [
            pd.Series({"B": (1, 1), "C": (3, 3), "Class": _0}, name=0),
            pd.Series({"B": (1, 1), "C": (2, 2), "Class": _1}, name=1),
        ]
        conf_matrix = update_confusion_matrix(examples[0], rules[0],
                                              positive_class, class_col_name,
                                              conf_matrix)  # TP
        conf_matrix = update_confusion_matrix(examples[1], rules[0],
                                              positive_class, class_col_name,
                                              conf_matrix)  # FP
        conf_matrix = update_confusion_matrix(examples[2], rules[1],
                                              positive_class, class_col_name,
                                              conf_matrix)  # FN
        conf_matrix = update_confusion_matrix(examples[3], rules[1],
                                              positive_class, class_col_name,
                                              conf_matrix)  # TN
        correct = ConfusionMatrix(
            TP={0, 3},
            TN={1, 6},
            FN={5},
            FP={2, 4},
        )
        self.assertEqual(correct, conf_matrix)
