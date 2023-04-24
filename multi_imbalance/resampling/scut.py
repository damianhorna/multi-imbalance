from copy import deepcopy
from typing import List, Tuple
from collections import Counter

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from imblearn.base import BaseSampler


class SCUT(BaseSampler):
    def __init__(self, k: int = 3):
        super().__init__()
        self._sampling_type = "clean-sampling"

        self.k = k

        self.m = None

    def _undersample(
        self, X: np.ndarray, y: np.ndarray, classes_for_undersample: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_undersample = list()
        y_undersample = list()
        for class_label in classes_for_undersample:
            X_subset = X[y == class_label]
            gmm_mixture = GaussianMixture(n_components=self.k)
            gmm_mixture.fit(X_subset)
            generated_samples, _ = gmm_mixture.sample(self.m)
            X_undersample.extend(generated_samples)
            y_undersample.extend([class_label] * self.m)

        return np.array(X_undersample), np.array(y_undersample)

    def _oversample(
        self, X: np.ndarray, y: np.ndarray, classes_for_oversample: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_oversample = list()
        y_oversample = list()
        for class_label in classes_for_oversample:
            y_binared = deepcopy(y)
            y_binared[y != class_label] = 0 if class_label > 0 else 1
            sm = SMOTE(sampling_strategy={class_label: self.m})
            X_smote, y_smote = sm.fit_resample(X, y_binared)
            X_oversample.extend(X_smote[y_smote == class_label])
            y_oversample.extend(y_smote[y_smote == class_label])

        return np.array(X_oversample), np.array(y_oversample)

    def _fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        quantities = Counter(y)
        self.m = np.mean(list(quantities.values()), dtype=int)
        classes_for_undersample = list(
            filter(lambda key: quantities[key] > self.m, quantities.keys())
        )
        X_undersample, y_undersample = self._undersample(X, y, classes_for_undersample)

        classes_for_oversample = list(
            filter(lambda key: quantities[key] < self.m, quantities.keys())
        )
        X_oversample, y_oversample = self._oversample(X, y, classes_for_oversample)

        stay_classes = list(
            filter(lambda key: quantities[key] == self.m, quantities.keys())
        )
        not_changing_indixes = y.searchsorted(stay_classes)
        X_not_changing, y_not_changing = (
            X[not_changing_indixes],
            y[not_changing_indixes],
        )
        X_resampled = np.vstack((X_undersample, X_oversample, X_not_changing))
        y_resampled = np.hstack((y_undersample, y_oversample, y_not_changing))
        return shuffle(X_resampled, y_resampled)
