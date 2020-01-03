import multiprocessing
from copy import deepcopy

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from multi_imbalance.resampling.SOUP import SOUP


def fit_clf(args):
    return SOUPBagging.fit_classifier(args)


class SOUPBagging(BaggingClassifier):
    def __init__(self, classifier=None, n_classifiers=5):
        super().__init__()
        self.classifiers = list()
        self.num_core = multiprocessing.cpu_count()
        self.n_classifiers = n_classifiers
        self.classes = None
        for _ in range(n_classifiers):
            if classifier is not None:
                self.classifiers.append(deepcopy(classifier))
            else:
                self.classifiers.append(KNeighborsClassifier())

    @staticmethod
    def fit_classifier(args):
        clf, X, y = args
        x_sampled, y_sampled = resample(X, y, stratify=y)
        x_resampled, y_resampled = SOUP().fit_transform(x_sampled, y_sampled)
        clf.fit(x_resampled, y_resampled)
        return clf

    def fit(self, X, y):
        """

        :param X: {array-like, sparse matrix} of shape = [n_samples, n_features] The training input samples.
        :param y: array-like, shape = [n_samples]. The target values (class labels).
        :return: self object
        """
        self.classes = np.unique(y)

        pool = multiprocessing.Pool(self.num_core)
        self.classifiers = pool.map(fit_clf, [(clf, X, y) for clf in self.classifiers])
        pool.close()
        pool.join()

    def predict(self, X, strategy: str = 'average', maj_int_min: dict = None):
        """
        Predict class for X. The predicted class of an input sample is computed as the class with the highest
        sum of predicted probability.

        :param X: {array-like, sparse matrix} of shape = [n_samples, n_features]. The training input samples.
        :param strategy:
            'average' - takes max from average values in prediction
            'optimistic' - takes always best value of probability
            'pessimistic' - takes always the worst value of probability
            'mixed' - for minority classes takes optimistic strategy, and pessimistic for others. It requires maj_int_min
        :param maj_int_min: dict. It keeps indices of minority classes under 'min' key.
        :return: y : array of shape = [n_samples]. The predicted classes.
        """
        weights_sum = self.predict_proba(X)
        if strategy == 'average':
            p = np.sum(weights_sum, axis=0)
        elif strategy == 'optimistic':
            p = np.max(weights_sum, axis=0)
        elif strategy == 'pessimistic':
            p = np.min(weights_sum, axis=0)
        elif strategy == 'mixed':
            n_samples = X.shape[0]
            n_classes = self.classes.shape[0]
            p = np.zeros(shape=(n_samples, n_classes)) - 1

            for i in range(n_classes):
                two_dim_class_vector = weights_sum[:, :, i]  # [:,:,1] -> [classifiers x samples]
                if i in maj_int_min['min']:
                    squeeze_with_strategy = np.max(two_dim_class_vector, axis=0)
                else:
                    squeeze_with_strategy = np.min(two_dim_class_vector, axis=0)  # [1, n_samples, 1] -> [n_samples]
                p[:, i] = squeeze_with_strategy
            assert -1 not in p
        else:
            raise KeyError(f'Incorrect strategy param: ${strategy}')

        y_result = np.argmax(p, axis=1)
        return y_result

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        :param X:{array-like, sparse matrix} of shape = [n_samples, n_features]. The training input samples.
        :return: array of shape = [n_classifiers, n_samples, n_classes]. The class probabilities of the input samples.
        """
        n_samples = X.shape[0]
        n_classes = self.classes.shape[0]

        results = np.zeros(shape=(self.n_classifiers, n_samples, n_classes))

        for i, clf in enumerate(self.classifiers):
            results[i] = clf.predict_proba(X)

        return results
