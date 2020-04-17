import pytest
import multi_imbalance.ensemble.ovo as ovo
import numpy as np

X = np.array([
    [-0.5813674466943386, -0.37091887120486655, -0.4465813355321204],
    [-0.630844420005455, 0.2871060228285258, 0.25613448374582437],
    [-0.6714353752038125, -0.3537255703996809, -0.9687281557330454],
    [0.48996953214789785, -0.09697447345439447, 0.5495667841083927],
    [-0.8821146485100975, -0.7739441502933209, 0.34906417794620515],
    [-0.6652132165510964, -0.8488383805882527, -0.030511639438375093],
    [0.7846621478367604, -0.9231479370667406, 0.7262231362586529],
    [0.7860907554630845, -0.33615224298146584, 0.6928271619140047],
    [0.7630774674537872, -0.7753044382704197, -0.7570971821030896],
    [-0.5843764899573332, -0.524996569658353, 0.9675951634125524],
    [-33.5843764899573332, -0.303030303030303, 5.9292929292929290],
    [22.5843764899573332, -2.020202022020202, 6.3213211113213211],
    [11.5843764899573332, -22.110101010011010, 7.3213123131232111],
    [2.5843764899573332, -1.010123211231232, 1.9675951634125524],
    [-0.5843764899573332, 1.321321312112321, 2.3213123123123222],
    [-0.2321312313211321, -412.321321312112321, 6.1010101010100101],
    [-0.3921809321038213, -2.321321312112321, 3.1010101010100101],
    [1.5843764899573332, -0.4324234243242342, 32.1010101010100101],
    [2.5843764899573332, -0.321321312112321, 53.1010101010100101],
    [42.5843764899573332, 66.321321312112321, 3242.1010101010100101],
    [1.5843764899573332, 44.321321312112321, 2.999909909090909090],
    [-2.5843764899573332, 12.321321312112321, 2112.342423],
    [-5.5843764899573332, 3.321321312112321, 2212.1010101010100101],
    [-8.5843764899573332, 1.321321312112321, 1222.1010101010100101],
    [-2.3213123213211232, -2.321321312112321, 992.1010101010100101],
])

y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1])


def test_fit_predict():
    clf = ovo.OVO(preprocessing=None)
    clf.fit(X[:-1], y[:-1])
    predicted = clf.predict(X[-1].reshape(1, -1))
    assert predicted in (1, 2, 3)


@pytest.mark.parametrize("classifier", ['tree', 'NB', 'KNN'])
def test_binary_classifiers(classifier):
    clf = ovo.OVO(binary_classifier=classifier, preprocessing=None)
    clf.fit(X[:-1], y[:-1])
    classifiers = clf._binary_classifiers

    assert len(classifiers) == 3
    assert classifiers[1][0].predict(X[-1].reshape(1, -1)) in (1, 2)
    assert classifiers[2][0].predict(X[-1].reshape(1, -1)) in (1, 3)
    assert classifiers[2][1].predict(X[-1].reshape(1, -1)) in (2, 3)


def test_max_voting():
    labels = np.array([4, 3, 6, 5, 7])
    binary_outputs = np.array([[0, 0, 0, 0, 0],
                               [3, 0, 0, 0, 0],
                               [4, 3, 0, 0, 0],
                               [4, 5, 6, 0, 0],
                               [7, 7, 7, 5, 0]])

    clf = ovo.OVO()
    clf._labels = labels
    voting_winner = clf._perform_max_voting(binary_outputs)
    assert voting_winner == 7


def test_with_own_classifier():
    class DummyClassifier:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.ones(len(X))

    dummy_clf = DummyClassifier()
    ovo_clf = ovo.OVO(binary_classifier=dummy_clf, preprocessing=None)
    ovo_clf.fit(X, y)
    predicted = ovo_clf.predict(np.array([[1.0, 2.0], [4.0, 5.5], [6.7, 8.8]]))
    assert np.all(predicted == 1)


def test_with_own_preprocessing():
    class DummyResampler:
        def fit_transform(self, X, y):
            return np.concatenate((X, X), axis=0), np.concatenate((y, y), axis=None)

    dummy_resampler = DummyResampler()
    ovo_clf = ovo.OVO(preprocessing=dummy_resampler)
    X_oversampled, y_oversampled = ovo_clf._oversample(X, y)
    assert len(X_oversampled) == 2 * len(X)
    assert len(y_oversampled) == 2 * len(y)


@pytest.mark.parametrize("preprocessing_btwn", ['all', 'maj-min'])
@pytest.mark.parametrize("classifier", ['tree', 'NB', 'KNN'])
@pytest.mark.parametrize("preprocessing", [None, 'globalCS', 'SMOTE', 'SOUP'])
def test_predefined_classifiers_and_preprocessings_without_errors(classifier, preprocessing, preprocessing_btwn):
    ovo_clf = ovo.OVO(binary_classifier=classifier, preprocessing=preprocessing,
                      preprocessing_between=preprocessing_btwn)
    ovo_clf.fit(X, y)
    predicted = ovo_clf.predict(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]))
    assert len(predicted) == 3


def test_unknown_preprocessing():
    ovo_clf = ovo.OVO(preprocessing='DUMMY_OVERSAMPLING')
    with pytest.raises(ValueError) as e:
        ovo_clf.fit(X, y)
    assert 'DUMMY_OVERSAMPLING' in str(e.value)


def test_own_preprocessing_without_fit_transform():
    class DummyOversampler:
        def foo(self, X, y):
            pass

        def bar(self, X):
            return np.zeros(len(X))

    dummy_oversampler = DummyOversampler()
    ovo_clf = ovo.OVO(preprocessing=dummy_oversampler)
    with pytest.raises(ValueError) as e:
        ovo_clf.fit(X, y)
    assert 'fit_transform' in str(e.value)


def test_unknown_preprocessing_between_strategy_raises_exception():
    ovo_clf = ovo.OVO(preprocessing_between='min-intermediate')
    with pytest.raises(ValueError) as e:
        ovo_clf.fit(X, y)
    assert 'min-intermediate' in str(e.value)
