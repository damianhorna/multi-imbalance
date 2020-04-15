from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE


class StaticSMOTE:
    """
    Static SMOTE implementation:

    Reference:
    Fernández-Navarro, F., Hervás-Martínez, C., Gutiérrez, P.A.: A dynamic over-sampling
    procedure based on sensitivity for multi-class problems. Pattern Recognit. 44, 1821–1833
    (2011)
    """

    def fit_transform(self, X, y):
        cnt = Counter(y)
        min_class = min(cnt, key=cnt.get)
        X_original, y_original = X.copy(), y.copy()
        X_resampled, y_resampled = X.copy(), y.copy()

        M = len(list(cnt.keys()))
        for _ in range(M):
            sm = SMOTE(sampling_strategy={min_class: cnt[min_class] * 2})
            X_smote, y_smote = sm.fit_resample(X_original, y_original)
            X_added_examples = X_smote[y_smote == min_class][cnt[min_class]:, :]
            X_resampled = np.vstack([X_resampled, X_added_examples])
            y_resampled = np.hstack([y_resampled, y_smote[y_smote == min_class][cnt[min_class]:]])
            cnt = Counter(y_resampled)
            min_class = min(cnt, key=cnt.get)

        return X_resampled, y_resampled
