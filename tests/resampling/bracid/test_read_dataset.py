from unittest import TestCase
import os
from collections import Counter

from multi_imbalance.resampling.bracid.bracid import BRACID


class TestReadDataset(TestCase):
    def test_read_numeric_dataset(self):
        # Get the absolute path to the parent directory of /scripts/
        bracid = BRACID()
        base_dir = os.path.abspath(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), os.pardir))
        src = os.path.join(base_dir, "datasets", "iris_test.csv")
        positive = "Iris-setosa"
        dataset, _, _ = bracid.read_dataset(src, positive)
        correct_column_names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm",
                                "class"]
        self.assertTrue(dataset.columns.tolist() == correct_column_names)
        self.assertTrue(dataset.shape == (12, 5))
        self.assertTrue(lookup == {})
