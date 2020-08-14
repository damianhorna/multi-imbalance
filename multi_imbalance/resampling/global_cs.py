from collections import Counter

import numpy as np
import sklearn
from imblearn.base import BaseSampler


class GlobalCS(BaseSampler):
    """
    Global CS is an algorithm that equalizes number of samples in each class. It duplicates all samples equally
    for each class to achieve majority class size
    """

    def __init__(self, shuffle: bool = True):
        super().__init__()
        self._sampling_type = 'over-sampling'
        self.shuffle = shuffle
        self.quantities, self.max_quantity, self.X, self.y = [None] * 4

    def _fit_resample(self, X, y):
        """
        :param X:
            two dimensional numpy array (number of samples x number of features) with float numbers
        :param y:
            one dimensional numpy array with labels for rows in X
        :return:
            Resampled X (max class quantity * number of unique classes), y (number of rows in X) as numpy array
        """
        assert len(X.shape) == 2, 'X should have 2 dimension'
        assert X.shape[0] == y.shape[0], 'Number of labels must be equal to number of samples'

        self.quantities = Counter(y)
        self.max_quantity = int(np.max(list(self.quantities.values())))
        self.X = X
        self.y = y

        result_X, result_y = list(), list()

        for class_name, class_quantity in self.quantities.items():
            temp_X, temp_y = self._equal_oversample(self.X, self.y, class_name)

            result_X.extend(temp_X)
            result_y.extend(temp_y)

        if self.shuffle:
            result_X, result_y = sklearn.utils.shuffle(result_X, result_y)

        return np.array(result_X), np.array(result_y)

    def _equal_oversample(self, X, y, class_name):
        indices_in_class = [i for i, class_label in enumerate(y) if class_label == class_name]
        desired_quantity = self.max_quantity - len(indices_in_class)

        oversampled_X, oversampled_y = list(X[indices_in_class]), list(y[indices_in_class])

        for i in range(desired_quantity):
            sample_index_to_duplicate: int = i % self.quantities[class_name]
            sample_id: int = indices_in_class[sample_index_to_duplicate]
            oversampled_X.append(X[sample_id])
            oversampled_y.append(y[sample_id])

        return oversampled_X, oversampled_y
