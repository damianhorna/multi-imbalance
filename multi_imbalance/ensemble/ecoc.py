import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from sklearn.tree import DecisionTreeClassifier


class ECOC(BaseEstimator):
    """
    ECOC (Error Correcting Output Codes) is ensemble method for multi-class classification problems.
    Each class is encoded with unique binary or ternary code (where 0 means that class is excluded from training set
    of dichotomy). Then in the learning phase each dichotomy is learned. In the decoding phase the class which is
    closest to test instance is chosen.
    """

    def __init__(self, classifier='CART', distance='hamming', penalty=1.0, oversample=None, encoding='dense'):
        """
        Parameters
        ----------
        classifier: binary classifier used by dichotomies. Possible classifiers:
        * 'CART': Decision Tree Classifier,
        * 'NB': Naive Bayes Classifier,

        distance: binary classifier. Possible classifiers:
        * 'hamming': Hamming's distance

        oversample: method for oversampling imbalanced data. Possible methods:
        * None : no oversampling applied,
        * 'random' : random oversampling - random chosen instances of minority classes are duplicated
        * 'SMOTE' : Synthetic Minority Oversampling Technique

        encoding : algorithm for encoding classes. Possible encodings:
        * 'dense'
        * 'sparse'
        * 'OVO'
        * 'OVA'
        * 'complete'

        """
        self.classifier = classifier
        self.distance = distance
        self.penalty = penalty
        self.oversample = oversample
        self.encoding = encoding

        self._code_matrix = None
        self._binary_classifiers = []
        self._labels = None

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: two dimensional numpy array (number of samples x number of features) with float numbers
        y: one dimensional numpy array with labels for rows in X
        Returns
        -------
        self: object
        """
        X, y = self._oversample(X, y)
        self._labels = np.unique(y)
        self._gen_code_matrix()
        self._binary_classifiers = [DecisionTreeClassifier() for _ in range(self._code_matrix.shape[1])]
        self._learn_binary_classifiers(X, y)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X: two dimensional numpy array (number of samples x number of features) with float numbers
        Returns
        -------
        y : numpy array, shape = [number of samples]
            Predicted target values for X.
        """
        output_codes = np.zeros((X.shape[0], self._code_matrix.shape[1]))
        for classifier_idx, classifier in enumerate(self._binary_classifiers):
            output_codes[:, classifier_idx] = classifier.predict(X)

        predicted = np.zeros(X.shape[0])
        for row_idx, encoded_row in enumerate(output_codes):
            predicted[row_idx] = self._get_closest_class(encoded_row)

        return predicted

    def _learn_binary_classifiers(self, X, y):
        for classifier_idx, classifier in enumerate(self._binary_classifiers):
            excluded_classes_indices = [idx for idx in range(len(y)) if y[idx] == 0]
            X_filtered = np.delete(X, excluded_classes_indices, 0)
            y_filtered = np.delete(y, excluded_classes_indices)
            binary_labels = [self._code_matrix[self._labels.tolist().index(clazz)][classifier_idx] for clazz in
                             y_filtered]
            classifier.fit(X_filtered, binary_labels)

    def _gen_code_matrix(self):
        allowed_encodings = ('dense', 'sparse', 'complete', 'OVA', 'OVO')

        if self.encoding == 'dense':
            self._code_matrix = self._encode_dense(self._labels.shape[0])
        elif self.encoding == 'sparse':
            self._code_matrix = self._encode_sparse(self._labels.shape[0])
        elif self.encoding == 'complete':
            self._code_matrix = self._encode_complete(self._labels.shape[0])
        elif self.encoding == 'OVO':
            self._code_matrix = self._encode_ovo(self._labels.shape[0])
        elif self.encoding == 'OVA':
            self._code_matrix = self._encode_ova(self._labels.shape[0])
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

    def _hamming_distance(self, v1, v2):
        return np.count_nonzero(v1 != v2)

    def _has_matrix_all_zeros_column(self, matrix):
        return (~matrix.any(axis=0)).any()

    def _get_closest_class(self, row):
        return self._labels[
            np.argmin([self._hamming_distance(row, encoded_class) for encoded_class in self._code_matrix])]

    def _oversample(self, X, y):
        allowed_oversampling = [None, 'random', 'SMOTE']
        if self.oversample not in allowed_oversampling:
            raise ValueError("Unknown matrix generation encoding: %s, expected to be one of %s."
                             % (self.encoding, allowed_oversampling))
        elif self.oversample is None:
            return X, y
        elif self.oversample == 'random':
            return self._random_oversample(X, y)
        elif self.oversample == 'SMOTE':
            return self._smote_oversample(X, y)

    def _random_oversample(self, X, y, random_state=0):
        random_state = check_random_state(random_state)

        values, counts = np.unique(y, return_counts=True)
        max_cardinality = np.max(counts)
        instances_to_be_added = max_cardinality - counts

        for clazz, missing_examples in zip(values, instances_to_be_added):
            y_clazz_indices = np.where(y == clazz)
            X_clazz = X[y_clazz_indices]
            for _ in range(missing_examples):
                rand_idx = random_state.randint(0, X_clazz.shape[0])
                y = np.append(y, clazz)
                X = np.vstack([X, X_clazz[rand_idx]])

        return X, y

    def _smote_oversample(self, X, y):
        pass