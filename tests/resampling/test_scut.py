from collections import Counter
import numpy as np

from multi_imbalance.resampling.scut import SCUT


def test_scut_resampling(X_ecoc, y_ecoc):
    scut = SCUT()
    quantities = Counter(y_ecoc)
    expected_size = np.mean(list(quantities.values()), dtype=int)

    X_resampled, y_resampled = scut.fit_resample(X_ecoc, y_ecoc)

    resampled_quantities = Counter(y_resampled)
    assert all([count == expected_size for count in resampled_quantities.values()])
