from unittest import TestCase
import os
from collections import Counter

from scripts.utils import read_dataset


class TestReadDataset(TestCase):
    def test_read_numeric_dataset(self):
        # Get the absolute path to the parent directory of /scripts/
        base_dir = os.path.abspath(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), os.pardir))
        src = os.path.join(base_dir, "datasets", "iris_test.csv")
        positive = "Iris-setosa"
        dataset, lookup, _, _ = read_dataset(src, positive)
        correct_column_names = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm",
                                "class"]
        self.assertTrue(dataset.columns.tolist() == correct_column_names)
        self.assertTrue(dataset.shape == (12, 5))
        self.assertTrue(lookup == {})

    def test_read_nominal_dataset(self):
        # Get the absolute path to the parent directory of /scripts/
        base_dir = os.path.abspath(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), os.pardir))
        src = os.path.join(base_dir, "datasets", "nominal_test.csv")
        # Automatically name the columns
        positive = "Iris-setosa"
        dataset, lookup, _, _ = read_dataset(src, positive, header=False)
        correct = \
            {0: {
                     'd': 2,
                     'e': 2,
                     'Conditional': {
                         'd': Counter({
                             'Iris-setosa': 2
                         }),
                         'e': Counter({
                             'Iris-versicolor': 1,
                             'Iris-virginica': 1
                         })}},
                1: {
                    'a': 2,
                    'b': 2,
                    'Conditional': {
                        'a': Counter({
                            'Iris-setosa': 2
                        }),
                        'b': Counter({
                            'Iris-versicolor': 1,
                            'Iris-virginica': 1
                        })
                    }
                },
                2: {
                    'y': 1,
                    'h': 1,
                    'j': 1,
                    'k': 1,
                    'Conditional': {
                        'y': Counter({
                            'Iris-setosa': 1
                        }),
                        'h': Counter({
                            'Iris-setosa': 1
                        }),
                        'j': Counter({
                            'Iris-versicolor': 1
                        }),
                        'k': Counter({
                            'Iris-virginica': 1
                        })
                    }
                }
             }
        self.assertTrue(dataset.columns.tolist() == [0, 1, 2, 3, 4])
        self.assertTrue(dataset.shape == (4, 5))
        self.assertTrue(lookup == correct)
