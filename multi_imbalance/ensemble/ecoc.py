import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from sklearn.tree import DecisionTreeClassifier


class ECOC(BaseEstimator):
    def __init__(self, classifier='CART', distance='hamming', penalty=1.0, oversample=False, encoding='dense'):
        self.classifier = classifier
        self.distance = distance
        self.penalty = penalty
        self.oversample = oversample
        self.encoding = encoding

        self._code_matrix = None
        self._binary_classifiers = []
        self._labels = None

    def fit(self, X, y):
        self._labels = np.unique(y)
        self._gen_code_matrix()
        self._binary_classifiers = [DecisionTreeClassifier() for _ in range(self._code_matrix.shape[1])]
        self._learn_binary_classifiers(X, y)
        return self

    def predict(self, X):
        output_codes = np.zeros(X.shape[0], self._code_matrix.shape[1])
        for classifier_idx, classifier in enumerate(self._binary_classifiers):
            output_codes[:, classifier_idx] = classifier.predict(X)

        predicted = np.zeros(X.shape[0])
        for row_idx, encoded_row in enumerate(output_codes):
            predicted[row_idx] = self._get_closest_class(encoded_row)

    def _learn_binary_classifiers(self, X, y):
        for classifier_idx, classifier in enumerate(self._binary_classifiers):
            excluded_classes_indices = [idx for idx in range(len(y)) if y[idx] == 0]
            X_filtered = np.delete(X, excluded_classes_indices, 0)
            y_filtered = np.delete(y, excluded_classes_indices)
            binary_labels = [self._code_matrix[self._labels.index(clazz)][classifier_idx] for clazz in y_filtered]
            classifier.fit(X_filtered, binary_labels)

    def _gen_code_matrix(self):
        allowed_encodings = ('dense', 'sparse', 'complete', 'OVA', 'OVO')

        if self.encoding == 'dense':
            self._code_matrix = self._encode_dense(self._labels)
        elif self.encoding == 'sparse':
            self._code_matrix = self._encode_sparse(self._labels)
        elif self.encoding == 'complete':
            self._code_matrix = self._encode_complete(self._labels)
        elif self.encoding == 'OVO':
            self._code_matrix = self._encode_ovo(self._labels)
        elif self.encoding == 'OVA':
            self._code_matrix = self._encode_ova(self._labels)
        else:
            raise ValueError("Unknown matrix generation encoding: %s, expected to be one of %s."
                             % (self.encoding, allowed_encodings))

    def _encode_dense(self, number_of_classes, random_state=0, number_of_code_generations=1000):
        number_of_columns = int(np.ceil(10 * np.log2(number_of_classes)))
        code_matrix = np.ones((number_of_classes, number_of_columns))
        random_state = check_random_state(random_state)

        max_min_dist = 0
        for i in range(number_of_code_generations):
            tmp_code_matrix = np.ones((number_of_classes, number_of_columns))
            min_dist = float('inf')

            for row in range(0, number_of_classes):
                for col in range(0, number_of_columns):
                    if random_state.randint(0, 2) == 1:
                        tmp_code_matrix[row, col] = -1

                for compared_row in range(0, row):
                    dist = self._hamming_distance(tmp_code_matrix[compared_row], tmp_code_matrix[row])
                    if dist < min_dist:
                        min_dist = dist

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                code_matrix = tmp_code_matrix
        return code_matrix

    def _encode_sparse(self, number_of_classes, random_state=0, number_of_code_generations=1000):
        number_of_columns = int(np.ceil(15 * np.log2(number_of_classes)))
        code_matrix = np.ones((number_of_classes, number_of_columns))
        random_state = check_random_state(random_state)

        max_min_dist = 0
        for i in range(number_of_code_generations):
            tmp_code_matrix = np.ones((number_of_classes, number_of_columns))
            min_dist = float('inf')

            for row in range(0, number_of_classes):
                for col in range(0, number_of_columns):
                    rand = random_state.randint(0, 4)
                    if rand < 2:
                        tmp_code_matrix[row, col] = 0
                    elif rand == 3:
                        tmp_code_matrix[row, col] = -1

                if np.count_nonzero(tmp_code_matrix[row]) == 0:
                    break

                for compared_row in range(0, row):
                    dist = self._hamming_distance(tmp_code_matrix[compared_row], tmp_code_matrix[row])
                    if dist < min_dist:
                        min_dist = dist

                if self._has_matrix_all_zeros_column(tmp_code_matrix):
                    continue

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                code_matrix = tmp_code_matrix

        return code_matrix

    def _encode_ova(self, number_of_classes):
        matrix = np.identity(number_of_classes)
        matrix[matrix == 0] = -1
        return matrix

    def _encode_ovo(self, number_of_classes):
        number_of_columns = int(number_of_classes * (number_of_classes - 1) / 2)
        matrix = np.zeros((number_of_classes, number_of_columns), dtype=int)
        indices_map = self._map_indices_to_class_pairs(number_of_classes)
        for row in range(number_of_classes):
            for col in range(number_of_columns):
                if indices_map[col][0] == row:
                    matrix[row, col] = 1
                elif indices_map[col][1] == row:
                    matrix[row, col] = -1
        return matrix

    def _map_indices_to_class_pairs(self, number_of_classes):
        indices_map = dict()
        idx = 0
        for i in range(number_of_classes):
            for j in range(i + 1, number_of_classes):
                indices_map[idx] = (i, j)
                idx += 1
        return indices_map

    def _encode_complete(self, X, y):
        pass

    def _oversample(self, X, y):
        pass

    def _hamming_distance(self, v1, v2):
        return np.count_nonzero(v1 != v2)

    def _has_matrix_all_zeros_column(self, matrix):
        return (~matrix.any(axis=0)).any()

    def _get_closest_class(self, row):
        return self._labels[
            np.argmin([self._hamming_distance(row, encoded_class) for encoded_class in self._code_matrix])]
