import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from multi_imbalance.resampling.GlobalCS import GlobalCS


class OVO(BaseEstimator):
    """

    OVO (One vs One) is an ensemble method that makes predictions for multi-class problems. OVO decomposes problem
    into m(m-1)/2 binary problems, where m is number of classes. Each of binary classifiers distinguishes between two
    classes. In the learning phase each classifier is learned only with instances from particular two classes. In
    prediction phase each classifier decides between these two classes. Results are aggregated and final output is
    derived depending on chosen aggregation model.

    """

    _allowed_classifiers = ['CART', 'NB', 'KNN']
    _allowed_voting_strategies = ['max']
    _allowed_oversampling = [None, 'globalCS', 'SMOTE']
    _allowed_oversample_between = ['all', 'maj-min']

    def __init__(self, voting_strategy='max', binary_classifier='CART', n_neighbors=5, oversample_binary=None,
                 oversample_between='all'):
        """
        Parameters
        ----------
        voting_strategy: aggregation model for deriving final output. Possible strategies:

        * 'max': class with largest number of votes is chosen,

        binary_classifier: binary classifier. Possible classifiers:

        * 'CART': Decision Tree Classifier,
        * 'KNN': K-Nearest Neighbors
        * 'NB' : Naive Bayes

        n_neighbors: number of nearest neighbors in KNN, works only if binary_classifier=='KNN'

        oversample_between: types of classes between which oversampling should be applied. Possible values:
        * 'all' - oversampling between each pair of classes
        * 'maj-min' - oversampling only between majority ad minority classes

        """
        self.voting_strategy = voting_strategy
        self.binary_classifier = binary_classifier
        self.n_neighbors = n_neighbors
        self.oversample_binary = oversample_binary
        self.oversample_between = oversample_between
        self._binary_classifiers = []
        self._labels = np.array([])
        self._minority_classes = list()

    def fit(self, X, y, minority_classes=None):
        """
        Parameters
        ----------
        X: two dimensional numpy array (number of samples x number of features) with float numbers
        y: one dimensional numpy array with labels for rows in X
        minority_classes: list of classes considered to be minority

        Returns
        -------
        self: object
        """
        if minority_classes is None:
            minority_classes = list()

        self._labels = np.unique(y)
        self._minority_classes = minority_classes
        num_of_classes = len(self._labels)
        self._binary_classifiers = [[self._get_classifier() for _ in range(n)] for n in
                                    range(0, num_of_classes)]
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
            binary_outputs_matrix = self._construct_binary_outputs_matrix(instance, num_of_classes)
            predicted.append(self._perform_voting(binary_outputs_matrix))

        return np.array(predicted)

    def _construct_binary_outputs_matrix(self, instance, num_of_classes):
        binary_outputs_matrix = np.zeros((num_of_classes, num_of_classes))
        for class_idx1 in range(len(self._labels)):
            for class_idx2 in range(class_idx1):
                binary_outputs_matrix[class_idx1][class_idx2] = self._binary_classifiers[class_idx1][class_idx2] \
                    .predict([instance])
        return binary_outputs_matrix

    def _learn_binary_classifiers(self, X, y):
        for row in range(len(self._labels)):
            for col in range(row):
                first_class, second_class = self._labels[row], self._labels[col]
                filtered_indices = [idx for idx in range(len(y)) if y[idx] in (first_class, second_class)]
                X_filtered, y_filtered = X[filtered_indices], y[filtered_indices]
                if self.should_perform_oversampling(first_class, second_class):
                    X_filtered, y_filtered = self._oversample(X_filtered, y_filtered, strategy=self.oversample_binary)
                self._binary_classifiers[row][col].fit(X_filtered, y_filtered)

    def _get_classifier(self):
        if self.binary_classifier not in OVO._allowed_classifiers:
            raise ValueError("Unknown binary classifier: %s, expected to be one of %s."
                             % (self.binary_classifier, OVO._allowed_classifiers))
        elif self.binary_classifier == 'CART':
            decision_tree_classifier = DecisionTreeClassifier()
            return decision_tree_classifier
        elif self.binary_classifier == 'NB':
            gnb = GaussianNB()
            return gnb
        elif self.binary_classifier == 'KNN':
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            return knn

    def _perform_voting(self, binary_outputs_matrix):
        if self.voting_strategy not in OVO._allowed_voting_strategies:
            raise ValueError("Unknown voting strategy: %s, expected to be one of %s."
                             % (self.voting_strategy, OVO._allowed_voting_strategies))
        elif self.voting_strategy == 'max':
            return self._perform_max_voting(binary_outputs_matrix)

    def _perform_max_voting(self, binary_outputs_matrix):
        scores = np.zeros(len(self._labels))
        for clf_1 in range(len(binary_outputs_matrix)):
            for clf_2 in range(clf_1):
                scores[self._labels.tolist().index(binary_outputs_matrix[clf_1][clf_2])] += 1
        return self._labels[np.argmax(scores)]

    def _oversample(self, X, y, strategy=None):
        if strategy not in OVO._allowed_oversampling:
            raise ValueError("Unknown matrix generation encoding: %s, expected to be one of %s."
                             % (strategy, OVO._allowed_oversampling))
        elif strategy is None:
            return X, y
        elif strategy == 'globalCS':
            gcs = GlobalCS()
            return gcs.fit_transform(X, y)
        elif strategy == 'SMOTE':
            return self._smote_oversample_if_possible_random_otherwise(X, y)

    def _smote_oversample_if_possible_random_otherwise(self, X, y):
        if min(np.unique(y, return_counts=True)[1]) < 2:
            return GlobalCS().fit_transform(X, y)

        n_neighbors = 3 if min(np.unique(y, return_counts=True)[1]) > 3 else 1
        smote = SMOTE(k_neighbors=n_neighbors)
        smote.fit(X, y)
        return smote.fit_resample(X, y)

    def should_perform_oversampling(self, first_class, second_class):
        if self.oversample_between not in OVO._allowed_oversample_between:
            raise ValueError("Unknown strategy for oversampling: %s, expected to be one of %s."
                             % (self.oversample_between, OVO._allowed_oversample_between))
        elif self.oversample_between == 'all':
            return True
        elif self.oversample_between == 'maj-min':
            return (first_class in self._minority_classes and second_class not in self._minority_classes) or \
                   (second_class in self._minority_classes and first_class not in self._minority_classes)
