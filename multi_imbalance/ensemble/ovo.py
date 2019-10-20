import numpy as np

from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


class OVO(BaseEstimator):
    """

    OVO (One vs One) is an ensemble method that makes predictions for multi-class problems. OVO decomposes problem
    into m(m-1)/2 binary problems, where m is number of classes. Each of binary classifiers distinguishes between two
    classes. In the learning phase each classifier is learned only with instances from particular two classes. In
    prediction phase each classifier decides between these two classes. Results are aggregated and final output is
    derived depending on chosen aggregation model.

    """

    def __init__(self, voting_strategy='max', binary_classifier='CART'):
        """
        Parameters
        ----------
        voting_strategy: aggregation model for deriving final output. Possible strategies:

        * 'max': class with largest number of votes is chosen,

        binary_classifier: binary classifier. Possible classifiers:

        *'CART': Decision Tree Classifier,
        """
        self.voting_strategy = voting_strategy
        self.binary_classifier = binary_classifier
        self._binary_classifiers = []
        self._indices_map = None
        self._labels = np.array([])

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
        self._labels = np.unique(y)
        num_of_classes = len(self._labels)
        num_of_binary_classifiers = int(num_of_classes * (num_of_classes - 1) / 2)
        self._binary_classifiers = [self._get_classifier() for _ in range(num_of_binary_classifiers)]
        self._indices_map = self._map_indices_to_class_pairs(num_of_classes)
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
        num_of_classes = len(self._labels)
        predicted = list()
        for instance in X:
            binary_outputs_matrix = np.zeros((num_of_classes, num_of_classes))
            for classifier_idx, classifier in enumerate(self._binary_classifiers):
                binary_outputs_matrix[self._indices_map[classifier_idx][0]][
                    self._indices_map[classifier_idx][1]] = classifier.predict([instance])
            predicted.append(self._perform_voting(binary_outputs_matrix))

        return np.array(predicted)

    def _learn_binary_classifiers(self, X, y):
        for classifier_idx, classifier in enumerate(self._binary_classifiers):
            first_class, second_class = self._labels[self._indices_map[classifier_idx][0]], \
                                        self._labels[self._indices_map[classifier_idx][1]]
            filtered_indices = [idx for idx in range(len(y)) if y[idx] in (first_class, second_class)]
            X_filtered, y_filtered = X[filtered_indices], y[filtered_indices]
            classifier.fit(X_filtered, y_filtered)

    def _map_indices_to_class_pairs(self, number_of_classes):
        indices_map = dict()
        idx = 0
        for i in range(number_of_classes):
            for j in range(i + 1, number_of_classes):
                indices_map[idx] = (i, j)
                idx += 1
        return indices_map

    def _get_classifier(self):
        allowed_classifiers = ('CART', 'NB')
        if self.binary_classifier not in allowed_classifiers:
            raise ValueError("Unknown binary classifier: %s, expected to be one of %s."
                             % (self.binary_classifier, allowed_classifiers))
        elif self.binary_classifier == 'CART':
            decision_tree_classifier = DecisionTreeClassifier()  # by default pruning is disabled
            return decision_tree_classifier
        elif self.binary_classifier == 'NB':
            gnb = GaussianNB()
            return gnb

    def _perform_voting(self, binary_outputs_matrix):
        allowed_voting_strategies = ('max',)
        if self.voting_strategy not in allowed_voting_strategies:
            raise ValueError("Unknown voting strategy: %s, expected to be one of %s."
                             % (self.voting_strategy, allowed_voting_strategies))
        elif self.voting_strategy == 'max':
            return self._perform_max_voting(binary_outputs_matrix)

    def _perform_max_voting(self, binary_outputs_matrix):
        scores = np.zeros(len(self._labels))
        for clf_1 in range(len(binary_outputs_matrix)):
            for clf_2 in range(clf_1 + 1, len(binary_outputs_matrix)):
                scores[self._labels.tolist().index(binary_outputs_matrix[clf_1][clf_2])] += 1
        return self._labels[np.argmax(scores)]
