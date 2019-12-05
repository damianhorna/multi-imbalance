import pytest
import numpy as np

import multi_imbalance.ensemble.ecoc as ecoc

X = np.array([
    [1.8938566839198983, 0.7347724642028586, 1.5817290619305417],
    [1.6893330472771877, 1.3729481360429043, 0.1779576959347715],
    [1.1103882804642866, 0.2684931500114267, 0.24565717871603532],
    [0.9635120154904986, 0.44338438370111577, 1.6559238383999697],
    [0.6525827502237067, 0.8978087724631425, 1.5056794207545134],
    [0.8232009732859464, 0.5270243940630088, 1.434695372722657],
    [0.519304726338536, 0.4635228434262648, 0.014170648565480004],
    [1.3938520002157688, 1.524670776643407, 0.9011189423913637],
    [0.09993454781831534, 0.5991188594563008, 0.6462181194010983],
    [1.5300019511124079, 0.08177359763506553, 1.7642527715349894],
    [1.1770688242955876, 0.9604049547799067, 0.6989025594835503],
    [1.5143651712498534, 1.4914673103908214, 1.3377704178955587],
    [1.1299009013495136, 0.700540900007983, 1.071829181951729],
    [1.530652133805449, 0.2992536048983532, 1.957731948975865],
    [1.6236761570974148, 0.5919033806975751, 1.6334065904199757],
    [0.9365056250644108, 1.526475631725099, 1.420298571686271],
    [0.9063995770780813, 1.0248369545634513, 1.36911505163145],
    [0.3861789635773656, 0.5758917834278445, 0.910187724154228],
    [0.7165380621896438, 1.494299618627891, 0.521854931610239],
    [1.6764775993219296, 0.15364096535456317, 1.371925603935502]
])

y = np.array([2, 0, 2, 3, 0, 3, 1, 0, 2, 0, 2, 3, 1, 2, 1, 3, 0, 3, 2, 0])


def test_random_oversampling():
    ecoc_clf = ecoc.ECOC(oversample_binary='globalCS')
    X_oversampled, y_oversampled = ecoc_clf._oversample(X, y)

    assert len(X_oversampled) == len(y_oversampled)
    assert len(set(np.unique(y_oversampled, return_counts=True)[1])) == 1
    assert set(y_oversampled).issubset(set(y))


def test_no_oversampling():
    ecoc_clf = ecoc.ECOC(oversample_binary=None)
    X_oversampled, y_oversampled = ecoc_clf._oversample(X, y)

    assert X.shape == X_oversampled.shape
    assert y.shape == y_oversampled.shape


@pytest.mark.parametrize("encoding_strategy", ['dense', 'sparse', 'OVO', 'OVA', 'complete'])
@pytest.mark.parametrize("oversampling", [None, 'globalCS', 'SMOTE'])
def test_encoding(encoding_strategy, oversampling):
    ecoc_clf = ecoc.ECOC(encoding=encoding_strategy, oversample_binary=oversampling)
    ecoc_clf.fit(X, y)
    matrix = ecoc_clf._code_matrix

    number_of_classes = len(np.unique(y))

    assert matrix.shape[0] == number_of_classes
    assert len(np.unique(matrix, axis=0)) == number_of_classes
    assert bool((~matrix.any(axis=0)).any()) is False


def test_hamming_distance():
    v1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1])
    v2 = np.array([-1, 1, -1, 1, -1, 0, 1, 0, 1, 0, -1, -1, -1, -1])
    distance = ecoc.ECOC()._hamming_distance(v1, v2)

    assert distance == 5
