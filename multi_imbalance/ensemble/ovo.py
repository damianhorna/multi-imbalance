from copy import deepcopy

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from multi_imbalance.resampling.global_cs import GlobalCS
from multi_imbalance.resampling.soup import SOUP


class OVO:
    """

    OVO (One vs One) is an ensemble method that makes predictions for multi-class problems. OVO decomposes problem
    into m(m-1)/2 binary problems, where m is number of classes. Each of binary classifiers distinguishes between two
    classes. In the learning phase each classifier is learned only with instances from particular two classes. In
    prediction phase each classifier decides between these two classes. Results are aggregated and final output is
    derived depending on chosen aggregation model.

    """

    _allowed_classifiers = ['tree', 'NB', 'KNN']
    _allowed_preprocessing = [None, 'globalCS', 'SMOTE', 'SOUP']
    _allowed_preprocessing_between = ['all', 'maj-min']

    def __init__(self, binary_classifier='tree', n_neighbors=3, preprocessing='SOUP', preprocessing_between='all'):
        """
        Parameters
        ----------
        binary_classifier: binary classifier. Possible classifiers:

        * 'tree': Decision Tree Classifier,
        * 'KNN': K-Nearest Neighbors
        * 'NB' : Naive Bayes
        * An instance of a class that implements ClassifierMixin

        n_neighbors: number of nearest neighbors in KNN, works only if binary_classifier=='KNN'

        preprocessing: method for preprocessing of pairs of classes in the learning phase of ensemble.
        Possible values:
        * None: no preprocessing applied
        * 'globalCS': oversampling with globalCS algorithm
        * 'SMOTE': oversampling with SMOTE algorithm
        * 'SOUP': oversampling and undersampling with SOUP algorithm
        * An instance of a class that implements TransformerMixin

        preprocessing_between: types of classes between which resampling should be applied. Possible values:
        * 'all' - oversampling between each pair of classes
        * 'maj-min' - oversampling only between majority ad minority classes

        """
        self.binary_classifier = binary_classifier
        self.n_neighbors = n_neighbors
        self.preprocessing = preprocessing
        self.oversample_between = preprocessing_between
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
            predicted.append(self._perform_max_voting(binary_outputs_matrix))

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
                    X_filtered, y_filtered = self._oversample(X_filtered, y_filtered)
                self._binary_classifiers[row][col].fit(X_filtered, y_filtered)

    def _get_classifier(self):
        if isinstance(self.binary_classifier, str):
            if self.binary_classifier not in OVO._allowed_classifiers:
                raise ValueError(
                    "Unknown binary classifier: %s, expected to be one of %s."
                    % (self.binary_classifier, OVO._allowed_classifiers))
            elif self.binary_classifier == 'tree':
                decision_tree_classifier = DecisionTreeClassifier(random_state=42)
                return decision_tree_classifier
            elif self.binary_classifier == 'NB':
                gnb = GaussianNB()
                return gnb
            elif self.binary_classifier == 'KNN':
                knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                return knn
        else:
            if not hasattr(self.binary_classifier, 'fit') or not hasattr(self.binary_classifier, 'predict'):
                raise ValueError("Your classifier must implement fit and predict methods")
            return deepcopy(self.binary_classifier)

    def _perform_max_voting(self, binary_outputs_matrix):
        scores = np.zeros(len(self._labels))
        for clf_1 in range(len(binary_outputs_matrix)):
            for clf_2 in range(clf_1):
                scores[self._labels.tolist().index(binary_outputs_matrix[clf_1][clf_2])] += 1
        return self._labels[np.argmax(scores)]

    def _oversample(self, X, y):
        if self.preprocessing is None:
            return X, y

        if isinstance(self.preprocessing, str):
            if self.preprocessing not in OVO._allowed_preprocessing:
                raise ValueError("Unknown preprocessing: %s, expected to be one of %s."
                                 % (self.preprocessing, OVO._allowed_preprocessing))
            elif np.unique(y).size == 1:
                return X, y
            elif self.preprocessing == 'globalCS':
                gcs = GlobalCS()
                return gcs.fit_transform(X, y)
            elif self.preprocessing == 'SMOTE':
                return self._smote_oversample(X, y)
            elif self.preprocessing == 'SOUP':
                soup = SOUP()
                return soup.fit_transform(X, y)
        else:
            if not hasattr(self.preprocessing, 'fit_transform'):
                raise ValueError("Your resampler must implement fit_transform method")
            return self.preprocessing.fit_transform(X, y)

    def _smote_oversample(self, X, y):
        n_neighbors = min(3, min(np.unique(y, return_counts=True)[1]) - 1)
        if n_neighbors == 0:
            raise ValueError(
                'In order to use SMOTE preprocessing, the training set should contain at least 2 examples from each class')
        smote = SMOTE(k_neighbors=n_neighbors, random_state=42)
        return smote.fit_resample(X, y)

    def should_perform_oversampling(self, first_class, second_class):
        if self.oversample_between not in OVO._allowed_preprocessing_between:
            raise ValueError("Unknown strategy for oversampling: %s, expected to be one of %s."
                             % (self.oversample_between, OVO._allowed_preprocessing_between))
        elif self.oversample_between == 'all':
            return True
        elif self.oversample_between == 'maj-min':
            return (first_class in self._minority_classes and second_class not in self._minority_classes) or \
                   (second_class in self._minority_classes and first_class not in self._minority_classes)
