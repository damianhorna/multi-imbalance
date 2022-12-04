import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import multi_imbalance.ensemble.ecoc as ecoc


def test_random_oversampling(X_ecoc, y_ecoc):
    ecoc_clf = ecoc.ECOC(preprocessing="globalCS")
    X_oversampled, y_oversampled = ecoc_clf._oversample(X_ecoc, y_ecoc)

    assert len(X_oversampled) == len(y_oversampled)
    assert len(set(np.unique(y_oversampled, return_counts=True)[1])) == 1
    assert set(y_oversampled).issubset(set(y_ecoc))


def test_no_oversampling(X_ecoc, y_ecoc):
    ecoc_clf = ecoc.ECOC(preprocessing=None)
    X_oversampled, y_oversampled = ecoc_clf._oversample(X_ecoc, y_ecoc)

    assert X_ecoc.shape == X_oversampled.shape
    assert y_ecoc.shape == y_oversampled.shape


@pytest.mark.parametrize("encoding_strategy", ["dense", "sparse", "OVO", "OVA", "complete"])
@pytest.mark.parametrize(
    "oversampling, minority_classes",
    [(None, None), ("globalCS", None), ("SMOTE", None), ("SOUP", [0, 2])],
)
def test_encoding(encoding_strategy, oversampling, minority_classes, X_ecoc, y_ecoc):
    ecoc_clf = ecoc.ECOC(encoding=encoding_strategy, preprocessing=oversampling)
    ecoc_clf.fit(X_ecoc, y_ecoc, minority_classes=minority_classes)
    matrix = ecoc_clf._code_matrix

    number_of_classes = len(np.unique(y_ecoc))

    assert matrix.shape[0] == number_of_classes
    assert len(np.unique(matrix, axis=0)) == number_of_classes
    assert bool((~matrix.any(axis=0)).any()) is False


@pytest.mark.parametrize("encoding_strategy", ["dense", "sparse"])
def test_dense_and_sparse_with_not_cached_matrices(encoding_strategy, X_ecoc, y_ecoc):
    X1 = np.concatenate((X_ecoc, 2 * X_ecoc, 3 * X_ecoc, 4 * X_ecoc, 5 * X_ecoc), axis=0)
    y1 = np.concatenate((y_ecoc + 4, y_ecoc + 8, y_ecoc + 12, y_ecoc + 16, y_ecoc + 20))

    ecoc_clf = ecoc.ECOC(encoding=encoding_strategy)
    ecoc_clf.fit(X1, y1)
    matrix = ecoc_clf._code_matrix

    number_of_classes = len(np.unique(y1))

    assert matrix.shape[0] == number_of_classes
    assert len(np.unique(matrix, axis=0)) == number_of_classes
    assert bool((~matrix.any(axis=0)).any()) is False


def test_hamming_distance():
    v1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1])
    v2 = np.array([-1, 1, -1, 1, -1, 0, 1, 0, 1, 0, -1, -1, -1, -1])
    distance = ecoc.ECOC()._hamming_distance(v1, v2)

    assert distance == 5


def test_with_own_classifier(X_ecoc, y_ecoc):
    class DummyClassifier:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    dummy_clf = DummyClassifier()
    ecoc_clf = ecoc.ECOC(binary_classifier=dummy_clf, preprocessing=None)
    ecoc_clf.fit(X_ecoc, y_ecoc)
    predicted = ecoc_clf.predict(np.array([[1.0, 2.0], [4.0, 5.5], [6.7, 8.8]]))
    assert np.all(predicted == 0)


def test_with_own_preprocessing(X_ecoc, y_ecoc):
    class DummyResampler:
        def fit_transform(self, X, y):
            return np.concatenate((X, X), axis=0), np.concatenate((y, y), axis=None)

    dummy_resampler = DummyResampler()
    ecoc_clf = ecoc.ECOC(preprocessing=dummy_resampler)
    X_oversampled, y_oversampled = ecoc_clf._oversample(X_ecoc, y_ecoc)
    assert len(X_oversampled) == 2 * len(X_ecoc)
    assert len(y_oversampled) == 2 * len(y_ecoc)


def test_unknown_classifier(X_ecoc, y_ecoc):
    ecoc_clf = ecoc.ECOC(binary_classifier="DUMMY_CLASSIFIER", preprocessing=None)
    with pytest.raises(ValueError) as e:
        ecoc_clf.fit(X_ecoc, y_ecoc)
    assert "DUMMY_CLASSIFIER" in str(e.value)


def test_unknown_encoding(X_ecoc, y_ecoc):
    ecoc_clf = ecoc.ECOC(encoding="dummy")
    with pytest.raises(ValueError) as e:
        ecoc_clf.fit(X_ecoc, y_ecoc)
    assert (
        e.value.args[0] == "Unknown matrix generation encoding: dummy, expected to be one of ['dense', 'sparse', 'complete', 'OVA', 'OVO']."
    )


def test_unknown_weighting_strategy(X_ecoc, y_ecoc):
    ecoc_clf = ecoc.ECOC(weights="dummy")
    with pytest.raises(ValueError) as e:
        ecoc_clf.fit(X_ecoc, y_ecoc)
    assert e.value.args[0] == "Unknown weighting strategy: dummy, expected to be one of [None, 'acc', 'avg_tpr_min']."


def test_own_classifier_without_predict_and_fit(X_ecoc, y_ecoc):
    class DummyClassifier:
        def foo(self, X, y):
            pass

        def bar(self, X):
            return np.zeros(len(X))

    dummy_clf = DummyClassifier()
    ecoc_clf = ecoc.ECOC(binary_classifier=dummy_clf, preprocessing=None)
    with pytest.raises(ValueError) as e:
        ecoc_clf.fit(X_ecoc, y_ecoc)
    assert "predict" in str(e.value)
    assert "fit" in str(e.value)


@pytest.mark.parametrize("classifier", ["tree", "NB", "KNN"])
@pytest.mark.parametrize("weights", [None, "acc", "avg_tpr_min"])
def test_predefined_classifiers_and_weighting_without_exceptions(classifier, weights, X_ecoc, y_ecoc):
    ecoc_clf = ecoc.ECOC(binary_classifier=classifier, weights=weights)
    ecoc_clf.fit(X_ecoc, y_ecoc)
    predicted = ecoc_clf.predict(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]))
    assert len(predicted) == 3


def test_unknown_preprocessing(X_ecoc, y_ecoc):
    ecoc_clf = ecoc.ECOC(preprocessing="DUMMY_OVERSAMPLING")
    with pytest.raises(ValueError) as e:
        ecoc_clf.fit(X_ecoc, y_ecoc)
    assert "DUMMY_OVERSAMPLING" in str(e.value)


def test_own_preprocessing_without_fit_transform(X_ecoc, y_ecoc):
    class DummyOversampler:
        def foo(self, X, y):
            pass

        def bar(self, X):
            return np.zeros(len(X))

    dummy_oversampler = DummyOversampler()
    ecoc_clf = ecoc.ECOC(preprocessing=dummy_oversampler)
    with pytest.raises(ValueError) as e:
        ecoc_clf.fit(X_ecoc, y_ecoc)
    assert "fit_transform" in str(e.value)


@pytest.mark.parametrize("encoding_strategy", ["dense", "sparse", "OVO", "OVA", "complete"])
@pytest.mark.parametrize("oversampling", [None, "globalCS", "SMOTE", "SOUP"])
def test_ecoc_with_sklearn_pipeline(encoding_strategy, oversampling, X_ecoc, y_ecoc):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ecoc", ecoc.ECOC(encoding=encoding_strategy, preprocessing=oversampling)),
        ]
    )
    pipeline.fit(X_ecoc, y_ecoc)
    y_hat = pipeline.predict(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]))
    assert len(y_hat) == 3
