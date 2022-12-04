import pytest
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd
from imblearn.metrics import geometric_mean_score
import numpy as np


from multi_imbalance.datasets.analysis import AnalysisPipeline, Config
from multi_imbalance.resampling.global_cs import GlobalCS
from multi_imbalance.resampling.soup import SOUP


def get_dummy_config():
    return {
        "datasets": [],
        "classifiers": {},
        "resample_methods": {},
        "metrics": {},
        "n_repeats": 2,
        "train_test_split_kwargs": {},
    }


@pytest.fixture
def dataset_file(tmp_path):
    filename = tmp_path / "dataset.csv"
    filename.touch()

    return str(filename)


@pytest.fixture
def output_file(tmp_path):
    filename = tmp_path / "output.csv"

    return str(filename)


@pytest.mark.parametrize(
    "config_dict",
    [
        {
            "datasets": ["path/to/data"],
            "classifiers": {
                "tree": [{}],
            },
            "resample_methods": {
                "globalCS": {},
            },
            "metrics": {lambda x, y: (x, y): {}},
            "n_repeats": 2,
            "train_test_split_kwargs": dict(test_size=0.2),
        }
    ],
)
def test_config_from_dict(config_dict):
    config = Config.from_dict(config_dict)

    assert config.datasets == config_dict["datasets"]
    assert config.classifiers == config_dict["classifiers"]
    assert config.resample_methods == config_dict["resample_methods"]
    assert config.metrics == config_dict["metrics"]
    assert config.n_repeats == config_dict["n_repeats"]
    assert config.train_test_split_kwargs == config_dict["train_test_split_kwargs"]


@pytest.mark.parametrize(
    "classifier, expected_name, expected_clf",
    [("tree", "tree", DecisionTreeClassifier), (LinearRegression, "linearregression", LinearRegression)],
)
def test_get_classifier(classifier, expected_name, expected_clf):
    config_dict = get_dummy_config()
    config_dict["classifiers"].update({classifier: [{}]})

    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)

    clf_name, clf = next(pipeline._get_classifier())
    assert clf_name == expected_name
    assert isinstance(clf, expected_clf)


@pytest.mark.parametrize(
    "resampler, expected_name, expected_resampler",
    [("globalCS", "globalCS", GlobalCS), (SOUP, "soup", SOUP)],
)
def test_get_resampler(resampler, expected_name, expected_resampler):
    config_dict = get_dummy_config()
    config_dict["resample_methods"].update({resampler: {}})

    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)

    resampler_name, resampler = next(pipeline._get_resampler())
    assert resampler_name == expected_name
    assert isinstance(resampler, expected_resampler)


@pytest.mark.parametrize("data, columns", [([[1, 2, 0]], ["X1", "X2", "y"]), ([[1, 2, 0], [4, 2, 1]], ["X1", "X2", "y"])])
def test_get_dataset(data, columns, tmp_path, dataset_file):
    config_dict = get_dummy_config()
    config_dict["datasets"].append(str(tmp_path))

    config = Config.from_dict(config_dict)
    expected_df = pd.DataFrame(data, columns=columns)
    expected_df.to_csv(dataset_file, index=False)

    pipeline = AnalysisPipeline(config)

    dataset_name, df = next(pipeline._get_dataset())
    assert dataset_name == "dataset"
    pd.testing.assert_frame_equal(df, expected_df)


def test_run_analysis(X_ecoc, y_ecoc, dataset_file, output_file):
    df = pd.DataFrame(X_ecoc, columns=["X1", "X2", "X3"])
    df["y"] = y_ecoc
    df.to_csv(dataset_file, index=False)
    config_dict = {
        "datasets": [dataset_file],
        "classifiers": dict(tree=[{"max_depth": 30}]),
        "resample_methods": dict(globalCS={"shuffle": True}),
        "metrics": {geometric_mean_score: {"correction": 0.005}, accuracy_score: {}},
        "n_repeats": 2,
        "train_test_split_kwargs": dict(test_size=0.2, random_state=42),
    }
    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)
    pipeline.run_analysis(output_file)

    result_df = pd.read_csv(output_file)
    assert (result_df["dataset_name"] == "dataset").all()
    assert (result_df["classifier"] == "tree").all()
    np.testing.assert_array_equal(result_df["metric_name"].unique(), ["geometric_mean_score", "accuracy_score"])
    np.testing.assert_array_almost_equal(result_df["metric_value"].unique(), [0.018803, 0.166667])


def test_get_dataset_wrong_path():
    config_dict = get_dummy_config()
    config_dict["datasets"].append("bad/path/to/file.ext")

    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)

    with pytest.raises(Exception) as ex:
        next(pipeline._get_dataset())

    assert ex.value.args[0] == "Wrong dataset path, should be csv file or dir with csv files"


@pytest.mark.parametrize(
    "wrong_resampler, expected_exception",
    [
        ("wrong", "Unknown resample method: wrong, expected to be one of ['globalCS', 'StaticSMOTE', 'SOUP', 'spider3', 'MDO']"),
        (lambda x: x, "Your resampler must implement fit_resample method"),
    ],
)
def test_get_resampler_wrong(wrong_resampler, expected_exception):
    config_dict = get_dummy_config()
    config_dict["resample_methods"].update({wrong_resampler: {}})

    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)

    with pytest.raises(ValueError) as ex:
        next(pipeline._get_resampler())

    assert ex.value.args[0] == expected_exception


@pytest.mark.parametrize(
    "wrong_clf, expected_exception",
    [
        ("wrong", "Unknown classifier: wrong, expected to be one of ['tree', 'NB', 'KNN']"),
        (lambda x: x, "Your classifier must implement fit and predict methods"),
    ],
)
def test_get_classifier_wrong(wrong_clf, expected_exception):
    config_dict = get_dummy_config()
    config_dict["classifiers"].update({wrong_clf: {}})

    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)

    with pytest.raises(ValueError) as ex:
        next(pipeline._get_classifier())

    assert ex.value.args[0] == expected_exception
