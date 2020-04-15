from multi_imbalance.resampling.static_smote import StaticSMOTE
from collections import Counter

from multi_imbalance.utils.data import load_arff_datasets


def test_static_smote():
    datasets = load_arff_datasets()
    X_ecoli, y_ecoli = datasets['new_ecoli'].data, datasets['new_ecoli'].target
    ssm = StaticSMOTE()
    X_resampled, y_resampled = ssm.fit_transform(X_ecoli, y_ecoli)
    cnt = Counter(y_resampled)
    assert cnt[0] == 145
    assert cnt[1] == 77
    assert cnt[2] == 148
    assert cnt[3] == 100
    assert cnt[4] == 104

