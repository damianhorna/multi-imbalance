import multiprocessing
from collections import Counter
from copy import deepcopy

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from multi_imbalance.resampling.soup import SOUP
from multi_imbalance.utils.array_util import setdiff


def fit_clf(args):
    return SOUPBagging.fit_classifier(args)


class SOUPBagging(BaggingClassifier):
    def __init__(self, classifier=None, maj_int_min=None, n_classifiers=5):
        super().__init__()
        self.classifiers, self.clf_weights = list(), list()
        self.maj_int_min = maj_int_min
        self.num_core = multiprocessing.cpu_count()
        self.n_classifiers = n_classifiers
        self.classes = None
        for _ in range(n_classifiers):
            self.clf_weights.append(1)
            if classifier is not None:
                self.classifiers.append(deepcopy(classifier))
            else:
                self.classifiers.append(KNeighborsClassifier())

    @staticmethod
    def fit_classifier(args):
        clf, X, y, resampled, maj_int_min = args
        x_sampled, y_sampled = resampled

        out_of_bag = setdiff(np.hstack((X, y[:, np.newaxis])), np.hstack((x_sampled, y_sampled[:, np.newaxis])))
        x_out, y_out = out_of_bag[:, :-1], out_of_bag[:, -1].astype(int)

        x_resampled, y_resampled = SOUP(maj_int_min=maj_int_min).fit_transform(x_sampled, y_sampled)
        clf.fit(x_resampled, y_resampled)

        result = clf.predict_proba(x_out)
        class_sum_prob = np.sum(result, axis=0) + 0.001
        class_quantities = Counter(y_out)
        expected_sum_prob = np.array([class_quantities[i] for i in range(len(Counter(y)))])
        try:
            global_weights = expected_sum_prob / class_sum_prob
        except Exception:
            global_weights = np.ones(shape=len(Counter(y)))
            print(f'Exc {Counter(y)} {Counter(y_out)} {result.shape} {expected_sum_prob.shape} {class_sum_prob.shape}')
        return clf, global_weights

    def fit(self, X, y):
        """

        :param X: {array-like, sparse matrix} of shape = [n_samples, n_features] The training input samples.
        :param y: array-like, shape = [n_samples]. The target values (class labels).
        :return: self object
        """
        self.classes = np.unique(y)

        pool = multiprocessing.Pool(self.num_core)
        results = pool.map(fit_clf, [(clf, X, y, resample(X, y, stratify=y, random_state=i), self.maj_int_min)
                                     for i, clf in enumerate(self.classifiers)])
        pool.close()
        pool.join()
        for i, (clf, weights) in enumerate(results):
            self.classifiers[i] = clf
            self.clf_weights[i] = weights

        self.clf_weights = np.array(self.clf_weights)

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
        elif strategy == 'global':
            for i, weight in enumerate(self.clf_weights):
                weights_sum[i] *= weight
            p = np.sum(weights_sum, axis=0)
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
