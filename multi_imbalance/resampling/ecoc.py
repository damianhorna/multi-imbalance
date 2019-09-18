import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from sklearn.tree import DecisionTreeClassifier


class ECOC(BaseEstimator):
    def __init__(self, classifier='CART', distance='bit_distance', penalty=1.0, oversample=False, encoding='dense'):
        self.classifier = classifier
        self.distance = distance
        self.penalty = penalty
        self.oversample = oversample
        self.encoding = encoding

        self._code_matrix = None

    def fit(self, X, y):
        self._code_matrix = self._gen_code_matrix(X, y)

        return self

    def predict(self, data):
        pass

    def _decode(self):
        pass

    def _gen_code_matrix(self, X, y):
        allowed_encodings = ('dense', 'sparse', 'complete', 'OVA', 'OVO')

        if self.encoding == 'dense':
            return self._encode_dense(y)
        elif self.encoding == 'sparse':
            return self._encode_sparse(y)
        elif self.encoding == 'complete':
            return self._encode_complete(y)
        elif self.encoding == 'OVO':
            pass
        elif self.encoding == 'OVA':
            pass
        else:
            raise ValueError("Unknown matrix generation encoding: %s, expected to be one of %s."
                             % (self.encoding, allowed_encodings))

    def _encode_dense(self, y, random_state=0, number_of_code_generations=10000):
        number_of_classes = len(y)
        number_of_columns = np.ceil(10 * np.log2(number_of_classes))

        code_matrix = np.ones((number_of_classes, number_of_columns))

        random_state = check_random_state(random_state)

        # TODO repeat generation of matrix given number of times and choose the one with largest minimum distance between rows
        for row in range(0, number_of_classes):
            for col in range(0, number_of_columns):
                if random_state.randint(0, 2) == 1:
                    code_matrix[row, col] = -1

        return code_matrix

    def _encode_sparse(self, y, random_state=0, number_of_code_generations=10000):
        number_of_classes = len(y)
        number_of_columns = np.ceil(15 * np.log2(number_of_classes))

        code_matrix = np.ones((number_of_classes, number_of_columns))

        random_state = check_random_state(random_state)

        # TODO repeat generation of matrix given number of times and choose the one with largest minimum distance between rows
        # TODO check if there are no rows with only 0s
        for row in range(0, number_of_classes):
            for col in range(0, number_of_columns):
                rand = random_state.randint(0, 4)
                if rand < 2:
                    code_matrix[row, col] = 0
                elif rand == 3:
                    code_matrix[row, col] = -1

        return code_matrix

    def _encode_complete(self, X, y):
        pass

    def _oversample(self, X, y):
        pass

    def _hamming_distance(self, v1, v2):
        return np.count_nonzero(v1 != v2)
