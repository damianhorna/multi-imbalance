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
    [-0.5843764899573332, -0.524996569658353, 0.9675951634125524]
])

y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3])


def test_fit_predict():
    clf = ovo.OVO()
    clf.fit(X[:-1], y[:-1])
    predicted = clf.predict(X[-1].reshape(1, -1))
    assert predicted in (1, 2, 3)


def test_binary_classifiers():
    clf = ovo.OVO()
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
