import json
from pathlib import Path
from click.testing import CliRunner
import pytest
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd
from imblearn.metrics import geometric_mean_score
from scikit_posthocs import posthoc_dunn

from multi_imbalance.datasets.analysis import AnalysisPipeline, Config, main
from multi_imbalance.resampling.global_cs import GlobalCS
from multi_imbalance.resampling.soup import SOUP


def get_dummy_config():
    return {
        "datasets": [],
        "classifiers": {},
        "resampling_methods": {},
        "metrics": {},
        "n_repeats": 2,
        "split_method": ["train_test", {}],
    }


@pytest.fixture
def dataset_file(tmp_path):
    filename = tmp_path / "dataset.csv"
    filename.touch()

    return filename


@pytest.fixture
def output_file(tmp_path):
    filename = tmp_path / "output.csv"

    return filename


@pytest.fixture
def output_csv(tmp_path):
    filename = tmp_path / "output.csv"
    filename.touch()
    columns = ["metric_name", "classifier", "dataset_name", "resampling_method", "metric_value", "no_repeat", "clf_params"]
    data = [
        ["geometric_mean_score", "decisiontreeclassifier", "glass", "globalcs", 0.5, 0, {}],
        ["geometric_mean_score", "decisiontreeclassifier", "glass", "globalcs", 0.5, 1, {}],
        ["geometric_mean_score", "decisiontreeclassifier", "glass", "globalcs", 1, 2, {}],
        ["geometric_mean_score", "decisiontreeclassifier", "glass", "globalcs", 1, 3, {}],
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    return filename


@pytest.fixture
def run_analysis_config(dataset_file):
    return {
        "datasets": [dataset_file],
        "classifiers": {DecisionTreeClassifier: [{"max_depth": 30}]},
        "resampling_methods": {GlobalCS: {"default": {"shuffle": True}}},
        "metrics": {geometric_mean_score: {"correction": 0.005}, accuracy_score: {}},
        "n_repeats": 2,
        "split_method": ["Kfold", dict(n_splits=2, shuffle=True)],
    }


@pytest.fixture
def run_analysis_config_json(dataset_file, tmp_path):
    config = {
        "datasets": [str(dataset_file)],
        "classifiers": {"sklearn.tree.DecisionTreeClassifier": [{"max_depth": 30}]},
        "resampling_methods": {"multi_imbalance.resampling.global_cs.GlobalCS": {"default": {"shuffle": True}}},
        "metrics": {"imblearn.metrics.geometric_mean_score": {"correction": 0.005}, "sklearn.metrics.accuracy_score": {}},
        "n_repeats": 2,
        "split_method": ["Kfold", dict(n_splits=2, shuffle=True)],
    }
    json_path = tmp_path / "config.json"
    with open(json_path, "w") as f:
        json.dump(config, f)

    return json_path


@pytest.fixture
def query_dict():
    return {
        "classifier": ["decisiontreeclassifier"],
        "metric_name": ["geometric_mean_score"],
        "dataset_name": ["dataset"],
        "resampling_method": ["globalcs"],
    }


@pytest.fixture
def query_dict_json(query_dict, tmp_path):
    json_path = tmp_path / "query.json"
    with open(json_path, "w") as f:
        json.dump(query_dict, f)

    return json_path


@pytest.fixture
def prepare_dataset_file(dataset_file, X_ecoc, y_ecoc):
    df = pd.DataFrame(X_ecoc, columns=["X1", "X2", "X3"])
    df["y"] = y_ecoc
    df.to_csv(dataset_file, index=False)


@pytest.fixture
def config_dict():
    return {
        "datasets": ["path/to/data"],
        "classifiers": {
            "tree": [{}],
        },
        "resampling_methods": {
            "globalCS": {},
        },
        "metrics": {lambda x, y: (x, y): {}},
        "n_repeats": 2,
        "split_method": ["train_test", {}],
    }


@pytest.fixture
def config_json(tmp_path):
    config_dict = {
        "datasets": [],
        "classifiers": {"sklearn.tree.DecisionTreeClassifier": [{}]},
        "resampling_methods": {"multi_imbalance.resampling.global_cs.GlobalCS": {"default": {}}},
        "metrics": {"imblearn.metrics.geometric_mean_score": {}},
        "n_repeats": 2,
        "split_method": ["train_test", {}],
    }
    json_path = tmp_path / "config.json"
    with open(json_path, "w") as f:
        json.dump(config_dict, f)

    return json_path


def test_config_from_dict(config_dict):
    config = Config.from_dict(config_dict)

    assert config.datasets == config_dict["datasets"]
    assert config.classifiers == config_dict["classifiers"]
    assert config.resampling_methods == config_dict["resampling_methods"]
    assert config.metrics == config_dict["metrics"]
    assert config.n_repeats == config_dict["n_repeats"]
    assert config.split_method == config_dict["split_method"]


def test_config_from_json(config_json):
    config = Config.from_json(config_json)
    for clf, clf_params in config.classifiers.items():
        for param in clf_params:
            assert isinstance(clf(**param), DecisionTreeClassifier)

    for resample, resample_params in config.resampling_methods.items():
        assert isinstance(resample(**resample_params["default"]), GlobalCS)


@pytest.mark.parametrize(
    "classifier, expected_name, expected_clf",
    [(DecisionTreeClassifier, "decisiontreeclassifier", DecisionTreeClassifier), (LinearRegression, "linearregression", LinearRegression)],
)
def test_get_classifier(classifier, expected_name, expected_clf):
    config_dict = get_dummy_config()
    config_dict["classifiers"].update({classifier: [{}]})

    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)

    clf_name, clf, params = next(pipeline._get_classifier())
    assert clf_name == expected_name
    assert isinstance(clf, expected_clf)
    assert params == {}


@pytest.mark.parametrize(
    "resampler, expected_name, expected_resampler",
    [(GlobalCS, "globalcs", GlobalCS), (SOUP, "soup", SOUP)],
)
def test_get_resampler(resampler, expected_name, expected_resampler):
    config_dict = get_dummy_config()
    config_dict["resampling_methods"].update({resampler: {"default": {}}})

    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)

    resampler_name, resampler = next(pipeline._get_resampler(train_without_resampling=False, dataset_name="dataset"))
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


@pytest.mark.parametrize(
    "kwargs, expected_output",
    [
        (
            {
                "dataset_name": "glass",
                "clf_name": "decisiontreeclassifier",
                "resampler_name": "globalcs",
                "metric_name": "geometric_mean_score",
                "n": 0,
                "clf_params": {},
            },
            True,
        ),
        (
            {
                "dataset_name": "dataset",
                "clf_name": "clf",
                "resampler_name": "resampler",
                "metric_name": "metric",
                "n": 0,
                "clf_params": {},
            },
            False,
        ),
    ],
)
def test_check_if_exist_in_csv(output_csv, kwargs, expected_output):
    config_dict = get_dummy_config()

    config = Config.from_dict(config_dict)
    pipeline = AnalysisPipeline(config)

    output = pipeline._check_if_exist_in_csv(path=output_csv, **kwargs)
    assert output == expected_output


@pytest.mark.parametrize("split_method", [["Kfold", dict(n_splits=4, shuffle=True)], ["train_test", {}]])
def test_run_analysis(prepare_dataset_file, output_file, run_analysis_config, split_method):
    run_analysis_config["split_method"] = split_method
    config = Config.from_dict(run_analysis_config)

    pipeline = AnalysisPipeline(config)
    pipeline.run_analysis(output_file, train_without_resampling=True)

    assert pipeline.dataset_names == ["dataset"]
    assert pipeline.clf_names == ["decisiontreeclassifier"]
    assert sorted(pipeline.resampling_methods) == ["Not defined", "globalcs"]
    assert sorted(pipeline.metric_names) == ["accuracy_score", "geometric_mean_score"]
    assert sorted(pipeline.column_names) == [
        "classifier",
        "clf_params",
        "dataset_name",
        "metric_name",
        "metric_value",
        "no_repeat",
        "resampling_method",
    ]


def test_run_analysis_cli(prepare_dataset_file, output_file, run_analysis_config_json):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            str(output_file),
            "--run-analysis",
            "--config-json",
            str(run_analysis_config_json),
        ],
    )
    assert result.exit_code == 0
    assert result.output == "Start\nRun analysis pipeline\nDone\n"

    assert Path(output_file).exists()


def test_generate_summary(prepare_dataset_file, query_dict, output_file, run_analysis_config, tmp_path):
    config = Config.from_dict(run_analysis_config)
    pipeline = AnalysisPipeline(config)
    pipeline.run_analysis(output_file, train_without_resampling=False)

    list_of_df = pipeline.generate_summary(query_dict, csv_path=output_file, save_to_csv=True)

    assert len(list_of_df) == 1
    df = list_of_df[0]
    assert df.shape[0] == 1
    assert (tmp_path / ("_".join([j for i in query_dict.values() for j in i]) + ".csv")).exists() is True


def test_generate_summary_cli(prepare_dataset_file, query_dict_json, output_file, run_analysis_config, tmp_path):
    config = Config.from_dict(run_analysis_config)
    pipeline = AnalysisPipeline(config)
    pipeline.run_analysis(output_file, train_without_resampling=False)

    aggr_func_path = tmp_path / "aggr_func.json"
    with open(aggr_func_path, "w") as f:
        json.dump(["numpy.median"], f)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [str(output_file), "--summary", "--query-json", str(query_dict_json), "--save-to-csv", "--aggregate-json", str(aggr_func_path)],
    )
    assert result.exit_code == 0
    assert result.output == "Start\nRun generate summary\nDone\n"

    with open(query_dict_json, "r") as f:
        query_dict = json.load(f)

    assert (tmp_path / ("_".join([j for i in query_dict.values() for j in i]) + ".csv")).exists() is True


def test_generate_posthoc_analysis(prepare_dataset_file, query_dict, output_file, run_analysis_config, tmp_path):
    config = Config.from_dict(run_analysis_config)
    pipeline = AnalysisPipeline(config)
    pipeline.run_analysis(output_file, train_without_resampling=False)
    query_dict.pop("resampling_method")
    list_of_df = pipeline.generate_posthoc_analysis(
        query_dict, csv_path=output_file, posthoc_func_list=[[posthoc_dunn, {}]], save_to_csv=True
    )

    assert len(list_of_df) == 1
    df = list_of_df[0]
    assert df.shape[0] == 1
    assert (tmp_path / ("_".join([posthoc_dunn.__name__, *[j for i in query_dict.values() for j in i]]) + ".csv")).exists() is True


def test_generate_posthoc_analysis_cli(prepare_dataset_file, query_dict, output_file, run_analysis_config, tmp_path):
    config = Config.from_dict(run_analysis_config)
    pipeline = AnalysisPipeline(config)
    pipeline.run_analysis(output_file, train_without_resampling=False)
    query_dict.pop("resampling_method")

    posthoc_func_path = tmp_path / "posthoc_func.json"
    with open(posthoc_func_path, "w") as f:
        json.dump({"scikit_posthocs.posthoc_dunn": {}}, f)

    query_dict_path = tmp_path / "query_dict.json"
    with open(query_dict_path, "w") as f:
        json.dump(query_dict, f)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            str(output_file),
            "--posthoc-analysis",
            "--posthoc-query-json",
            str(query_dict_path),
            "--save-to-csv",
            "--posthoc-func-json",
            str(posthoc_func_path),
        ],
    )
    assert result.exit_code == 0
    assert result.output == "Start\nRun generate posthoc analysis\nDone\n"

    assert (tmp_path / ("_".join([posthoc_dunn.__name__, *[j for i in query_dict.values() for j in i]]) + ".csv")).exists() is True


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
        (lambda x: x, "Your resampler must implement fit_resample method"),
        (GlobalCS, "Must define default params for all datasets or for the specific dataset"),
    ],
)
def test_get_resampler_wrong(wrong_resampler, expected_exception):
    config_dict = get_dummy_config()
    config_dict["resampling_methods"].update({wrong_resampler: {}})

    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)

    with pytest.raises(Exception) as ex:
        next(pipeline._get_resampler(train_without_resampling=False, dataset_name="dataset"))

    assert ex.value.args[0] == expected_exception


@pytest.mark.parametrize(
    "wrong_clf, expected_exception",
    [
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


def test_run_analysis_exception(X_ecoc, y_ecoc, dataset_file, output_file, caplog):
    class DummyClf:
        def __init__(*args, **kwargs):
            pass

        def fit(*args):
            raise Exception("Error during fit")

        def predict(*args):
            pass

    df = pd.DataFrame(X_ecoc, columns=["X1", "X2", "X3"])
    df["y"] = y_ecoc
    df.to_csv(dataset_file, index=False)
    config_dict = {
        "datasets": [dataset_file],
        "classifiers": {DummyClf: [{"max_depth": 30}]},
        "resampling_methods": {},
        "metrics": {geometric_mean_score: {"correction": 0.005}, accuracy_score: {}},
        "n_repeats": 2,
        "split_method": ["train_test", {}],
    }
    config = Config.from_dict(config_dict)

    pipeline = AnalysisPipeline(config)
    pipeline.run_analysis(output_file, train_without_resampling=True)

    for record in caplog.records:
        if record.levelname == "ERROR":
            assert (
                record.message
                == "Raised exception: 'Error during fit' for dataset_name='dataset', resampler_name='Not defined' and clf_name='dummyclf'\n"
                "Raised exception: 'Error during fit' for dataset_name='dataset', resampler_name='Not defined' and clf_name='dummyclf'"
            )
