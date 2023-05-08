from collections import Counter
from typing import Tuple

import numpy as np
from imblearn.base import BaseSampler


class DefaultSampler(BaseSampler):

    def __init__(self, sampler: BaseSampler):
        if sampler._sampling_type not in ["over-sampling", "under-sampling"]:
            raise ValueError("Base sampler must be of over-sampling or under-sampling type")
        self._sampling_type = sampler._sampling_type
        self.sampler = sampler
        super().__init__()

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        classes = Counter(y).keys()
        if self.sampler._sampling_type == "over-sampling":
            resampled_X = []
            resampled_y = []

            for current_class in classes:
                current_y = np.full(y.shape, 0)
                current_y[y == current_class] = 1
                res_X, res_y = self.sampler.fit_resample(X, current_y)
                minority_count = (res_y == 1).sum()
                resampled_X.append(res_X[res_y == 1])
                resampled_y.append(np.full(minority_count, current_class))

            return np.vstack(resampled_X), np.hstack(resampled_y)

        elif self.sampler._sampling_type == "under-sampling":
            X_copy, y_copy = X.copy(), y.copy()
            sorted_class_counts = sorted(list(Counter(y).items()), key=lambda x: x[1], reverse=True)
            classes = [class_count[0] for class_count in sorted_class_counts]

            for i in range(1, len(sorted_class_counts)):
                index = np.in1d(y_copy, classes[:i + 1])
                safe_X, safe_y = X_copy[~index], y_copy[~index]
                current_X, current_y = X_copy[index], y_copy[index]
                res_X, res_y = self.sampler.fit_resample(current_X, current_y)
                X_copy, y_copy = np.concatenate([res_X, safe_X]), np.concatenate([res_y, safe_y])

            return X_copy, y_copy


class StaticSampler(BaseSampler):

    def __init__(self, sampler: BaseSampler):
        super().__init__()
        self.sampler = sampler
        self._sampling_type = sampler._sampling_type

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cnt = Counter(y)
        min_class = min(cnt, key=cnt.get)
        X_original, y_original = X.copy(), y.copy()
        X_resampled, y_resampled = X.copy(), y.copy()

        M = len(list(cnt.keys()))
        for _ in range(M):
            self.sampler.sampling_strategy = {min_class: cnt[min_class] * 2}
            X_smote, y_smote = self.sampler.fit_resample(X_original, y_original)
            idx = cnt[min_class]
            X_added_examples = X_smote[y_smote == min_class][idx:, :]
            X_resampled = np.vstack([X_resampled, X_added_examples])
            y_resampled = np.hstack([y_resampled, y_smote[y_smote == min_class][idx:]])
            cnt = Counter(y_resampled)
            min_class = min(cnt, key=cnt.get)

        return X_resampled, y_resampled


class SamplerFactory:
    @staticmethod
    def create_sampler(sampler: BaseSampler, mode: str) -> BaseSampler:
        if mode == "default":
            return DefaultSampler(sampler)
        elif mode == "static":
            return StaticSampler(sampler)
        else:
            raise ValueError("Invalid resampler type")


class ImblearnWrapper(BaseSampler):
    """
    A wrapper for imblearn binary sampling methods. It is used to perform multi-class resampling by performing binary
    resampling for each class.
    """

    def __init__(self, sampler: BaseSampler, mode: str = "default") -> None:
        if sampler._sampling_type not in ["over-sampling", "under-sampling"]:
            raise ValueError("Base sampler must be of over-sampling or under-sampling type")
        self.sampler = SamplerFactory.create_sampler(sampler, mode)
        self.mode = mode
        self._sampling_type = sampler._sampling_type
        super().__init__()

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X:
            two-dimensional numpy array (number of samples x number of features) with float numbers
        :param y:
            one-dimensional numpy array with labels for rows in X
        :return:
            resampled X, resampled y
        """
        return self.sampler.fit_resample(X, y)
