
from scipy.stats import multinomial
from sklearn.utils import resample
import numpy as np


class MRBBagging(object):
    def __init__(self):
        self.classifiers, self.classes, self.classifier_classes = dict(), dict(), dict()

    def _group_data(self, x, y):
        classes = set(y)
        self.classes = {key: value for (key, value) in enumerate(classes)}
        data = [[x[i], y[i]] for i in range(len(x))]
        grouped_data = dict()
        for cl in classes:
            assert cl is not None, "Missing class name"
            grouped_data[cl] = list(filter(lambda d: d[1] == cl, data))
        return classes, grouped_data

    def fit(self, x, y, n, k, learning_algorithm):
        assert len(x) == len(y), "Not enough labels"
        assert learning_algorithm is not None, "Invalid learning algorithm"
        classes, grouped_data = self._group_data(x, y)
        prob = [1 / len(classes)] * len(classes)
        for i in range(k):
            samples_no = multinomial.rvs(n=n, p=prob)
            subset_x, subset_y = [], []
            for no, j in enumerate(classes):
                data = grouped_data[j]
                resample_class = resample(data, replace=True, n_samples=samples_no[no])
                for sample in resample_class:
                    subset_x.append(sample[0])
                    subset_y.append(sample[1])
            self.classifiers[i] = learning_algorithm[i].fit(subset_x, subset_y)
        return self

    def _set_classes_dict(self):
        self.classifier_classes = {key: value for (key, value) in enumerate(self.classifiers[0].classes_)}

    def _count_votes(self, data):
        voting_matrix = np.zeros((len(data), len(self.classes)))
        self._set_classes_dict()
        for c in range(len(self.classifiers)):
            classes = self.classifiers[c].predict(data)
            probabilities = self.classifiers[c].predict_proba(data)
            for i, cl in enumerate(classes):
                idx = list(self.classifier_classes.keys())[list(self.classifier_classes.values()).index(cl)]
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
        return self._select_classes(data)
