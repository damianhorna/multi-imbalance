from collections import Counter
from copy import deepcopy
from math import sqrt

from scipy.stats import multinomial
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.utils import resample
from sklearn.utils.random import sample_without_replacement
import numpy as np


class MRBBagging(object):
    """
    Multi-class Roughly Balanced Bagging (MRBBagging) is a generalization of MRBBagging for adapting to multiple
    minority classes.

    Reference:
    M. Lango, J. Stefanowski:
    Multi-class and feature selection extensions of RoughlyBalanced Bagging for imbalanced data.
    J. Intell Inf Syst (2018) 50: 97

    Methods:
    ----------
    fit(x, y)
        Build a MRBBagging ensemble of estimators from the training data.

    predict(data)
        Predict classes for examples in data.
    """

    def __init__(self, k, learning_algorithm, undersampling=True, feature_selection=False,
                 random_fs=False, half_features=True, random_state=None):
        """
        Parameters
        ----------
        :param k: number of classifiers (multiplied by 3 when choosing feature selection)
        :param learning_algorithm: classifier to be used
        :param undersampling: (optional) boolean value to determine if undersampling or oversampling should be performed
        :param feature_selection: (optional) boolean value to determine if feature selection should be performed
        :param random_fs: (optional) boolean value to determine if feature selection should be all random (if False, chi^2, F test
        and random feature selection are performed)
        :param half_features: (optional) boolean value to determine if the number of features to be selected should be 50%
        (if False, it is set to the square root of the base number of features)
        :param random_state: (optional) the seed of the pseudo random number generator
        """
        assert learning_algorithm is not None, "Learning algorithm cannot be None"
        assert k > 0, "Number of classifiers must be > 0"
        self.classifiers, self.classes, self.classifier_classes = dict(), dict(), dict()
        self.feature_selection_methods = dict()
        self.k = k
        self.learning_algorithm = learning_algorithm
        self.undersampling = undersampling
        self.feature_selection = feature_selection
        self.all_random = random_fs
        self.half_features = half_features
        self.random_state = random_state

    def _group_data(self, x, y):
        classes = set(y)
        self.classes = {key: value for (key, value) in enumerate(classes)}
        data = [[x[i], y[i]] for i in range(len(x))]
        grouped_data = dict()
        for cl in classes:
            assert cl is not None, "Missing class name"
            grouped_data[cl] = list(filter(lambda d: d[1] == cl, data))
        return classes, grouped_data

    def _resample(self, n, prob, classes, grouped_data):
        samples_no = multinomial.rvs(n=n, p=prob, random_state=self.random_state)
        subset_x, subset_y = [], []
        for no, j in enumerate(classes):
            data = grouped_data[j]
            resample_class = resample(data, replace=True, n_samples=samples_no[no], random_state=self.random_state)
            for sample in resample_class:
                subset_x.append(sample[0])
                subset_y.append(sample[1])
        return np.array(subset_x), np.array(subset_y)

    def fit(self, x, y):
        """
        Parameters
        ----------
        :param x:
            Two dimensional numpy array (number of samples x number of features) with float numbers.
        :param y:
            One dimensional numpy array with labels for rows in X.
        """
        assert len(x) == len(y), "Not enough labels"

        classes, grouped_data = self._group_data(x, y)
        prob = [1 / len(classes)] * len(classes)
        self._set_classes_dict(classes)

        n = len(x)
        if self.undersampling:
            q = Counter(y)
            n = min(q.values()) * len(q)

        la_list = []

        if self.feature_selection:
            for i in range(3 * self.k):
                la_list.append(deepcopy(self.learning_algorithm))
            self._train_with_feature_selection(la_list, n, prob, classes, grouped_data)

        else:
            for i in range(self.k):
                la_list.append(deepcopy(self.learning_algorithm))
            self._train(la_list, n, prob, classes, grouped_data)

        return self

    def _train(self, la_list, n, prob, classes, grouped_data):
        for i in range(len(la_list)):
            subset_x, subset_y = self._resample(n, prob, classes, grouped_data)

            subset_x = np.array(subset_x).astype(np.float)
            subset_y = np.array(subset_y).astype(np.float)

            self.classifiers[i] = la_list[i].fit(subset_x, subset_y)

    def _find_random_features(self, labels_no, features_no, subset_x):
        random_features_idx = sample_without_replacement(labels_no, features_no)
        random_features = self._get_features_array(subset_x, random_features_idx)
        return random_features, random_features_idx

    def _get_features_array(self, subset_x, random_features_idx):
        random_features = np.array(subset_x[:, random_features_idx[0]])
        for f in range(1, len(random_features_idx)):
            random_features = np.vstack((random_features, subset_x[:, random_features_idx[f]]))
        if random_features.ndim == 1:
            return random_features[:, np.newaxis]
        return random_features.T

    def _get_kbest_classifier(self, test, features_no, subset_x, subset_y):
        kBest_estimator = SelectKBest(test, k=features_no)
        subset = kBest_estimator.fit_transform(subset_x, subset_y)
        return subset, kBest_estimator

    def _train_with_feature_selection(self, la_list, n, prob, classes, grouped_data):
        for i in range(0, len(la_list), 3):
            subset_x, subset_y = self._resample(n, prob, classes, grouped_data)
            labels_no = len(subset_x[0])
            if self.half_features:
                features_no = int(labels_no / 2)
            else:
                features_no = int(sqrt(labels_no))

            subset_x = np.array(subset_x).astype(np.float)
            subset_y = np.array(subset_y).astype(np.float)

            if self.all_random:
                subset1, subset1_idx = self._find_random_features(labels_no, features_no, subset_x)
                subset2, subset2_idx = self._find_random_features(labels_no, features_no, subset_x)
                subset3, subset3_idx = self._find_random_features(labels_no, features_no, subset_x)

            else:
                subset1, subset1_idx = self._get_kbest_classifier(chi2, features_no, subset_x, subset_y)
                subset2, subset2_idx = self._get_kbest_classifier(f_classif, features_no, subset_x, subset_y)
                subset3, subset3_idx = self._find_random_features(labels_no, features_no, subset_x)

            self.feature_selection_methods[i] = subset1_idx
            self.feature_selection_methods[i + 1] = subset2_idx
            self.feature_selection_methods[i + 2] = subset3_idx

            self.classifiers[i] = la_list[i].fit(subset1, subset_y)
            self.classifiers[i + 1] = la_list[i + 1].fit(subset2, subset_y)
            self.classifiers[i + 2] = la_list[i + 2].fit(subset3, subset_y)

    def _set_classes_dict(self, classes):
        self.classifier_classes = dict(enumerate(classes))

    def _select_data(self, classifier_id, data):
        if self.feature_selection:
            if self.all_random:
                new_data = self._get_features_array(data, self.feature_selection_methods[classifier_id])
            else:
                if (classifier_id % 3) - 2 == 0:
                    new_data = self._get_features_array(data, self.feature_selection_methods[classifier_id])
                else:
                    new_data = self.feature_selection_methods[classifier_id].transform(data)
            return new_data
        return data

    def _count_votes(self, data):
        voting_matrix = np.zeros((len(data), len(self.classes)))
        for classifier_id in range(len(self.classifiers)):
            new_data = self._select_data(classifier_id, data)
            classes = self.classifiers[classifier_id].predict(new_data)
            probabilities = self.classifiers[classifier_id].predict_proba(new_data)
            for i, cl in enumerate(classes):
                idx = list(self.classifier_classes.keys())[list(self.classifier_classes.values()).index(int(cl))]
                voting_matrix[i][idx] += max(probabilities[i])
        return voting_matrix

    def _select_classes(self, data):
        voting_matrix = self._count_votes(data)
        selected_classes_ids = voting_matrix.argmax(axis=1)
        selected_classes = []
        for class_id in selected_classes_ids:
            selected_classes.append(self.classifier_classes[class_id])
        return selected_classes

    def predict(self, data):
        """
        Parameters
        ----------
        :param data:
            Two dimensional numpy array (number of samples x number of features) with float numbers.
        """
        return self._select_classes(data)
