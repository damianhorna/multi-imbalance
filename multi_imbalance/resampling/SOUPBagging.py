from copy import deepcopy

import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from multi_imbalance.datasets import load_datasets
from multi_imbalance.resampling.SOUP import SOUP


class SOUPBagging(object):
    def __init__(self, classifier=None, n_classifiers=30, seed=0):
        self.classifiers = list()
        self.n_classifiers = n_classifiers
        self.classes = None
        self.random_state = seed
        for _ in range(n_classifiers):
            if classifier is not None:
                self.classifiers.append(deepcopy(classifier))
            else:
                self.classifiers.append(KNeighborsClassifier())

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        for clf in self.classifiers:
            x_sampled, y_sampled = resample(X_train, y_train, stratify=y_train, random_state=self.random_state)
            x_resampled, y_resampled = SOUP().fit_transform(x_sampled, y_sampled)
            clf.fit(x_resampled, y_resampled)

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        n_classes = self.classes.shape[0]

        results = np.zeros(shape=(self.n_classifiers, n_samples, n_classes))

        for i, clf in enumerate(self.classifiers):
            results[i] = clf.predict_proba(X_test)

        weights_sum = np.sum(results, axis=0)
        y_result = np.argmax(weights_sum, axis=1)
        return y_result
