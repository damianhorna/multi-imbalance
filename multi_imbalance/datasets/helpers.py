from dataclasses import dataclass
import importlib
import json
from typing import Callable, Dict, List, Tuple, Union
from sklearn.base import ClassifierMixin
from imblearn.base import BaseSampler


def import_from_string(cls_path: str) -> Union[BaseSampler, ClassifierMixin, Callable]:
    module_name, class_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@dataclass
class Config:
    """A class representing the configuration for an analysis pipeline.

    Attributes:
    ----------
        datasets: A list of dataset names to use in the analysis pipeline.
        classifiers: A dictionary mapping classifier objects to lists of dictionaries containing the hyperparameters to use for each classifier.
        resampling_methods: A dictionary mapping resampling objects to dictionaries of hyperparameters to use for each resampling method.
        metrics: A dictionary mapping metric functions to dictionaries of hyperparameters to use for each metric.
        n_repeats: The number of times to repeat the experiment for datasets.
        split_method: A dictionary mapping split method to dictionaries of additional parameters.
            There are two options to choose: `Kfold` (StratifiedKFold) and `train_test` (train_test_split).
    """

    datasets: List[str]
    classifiers: Dict[ClassifierMixin, List[Dict]]
    resampling_methods: Dict[BaseSampler, Dict[str, Dict]]
    metrics: Dict[Callable, Dict]
    n_repeats: int
    split_method: List[Tuple[str, Dict]]

    @classmethod
    def from_dict(cls, config: Dict) -> "Config":
        """Load configuration from a dict.

        :param: config:
            The dict containing the configuration.

        :return:
            A Config object representing the configuration from dict.
        """
        return cls(**config)

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """Load configuration from a JSON file.

        :param: json_path:
            The path to the JSON file containing the configuration to load.

        :return:
            A Config object representing the configuration from the JSON file.
        """
        with open(json_path, "r") as f:
            dict_config = json.load(f)

        tmp_dict = {}
        for clf_path, clf_params in dict_config["classifiers"].items():
            tmp_dict[import_from_string(clf_path)] = clf_params

        dict_config["classifiers"].clear()
        dict_config["classifiers"].update(tmp_dict)
        tmp_dict.clear()

        for resample_path, resample_params in dict_config["resampling_methods"].items():
            tmp_dict[import_from_string(resample_path)] = resample_params

        dict_config["resampling_methods"].clear()
        dict_config["resampling_methods"].update(tmp_dict)
        tmp_dict.clear()

        for metric_path, metric_params in dict_config["metrics"].items():
            tmp_dict[import_from_string(metric_path)] = metric_params

        dict_config["metrics"].clear()
        dict_config["metrics"].update(tmp_dict)
        tmp_dict.clear()

        return cls(**dict_config)


@dataclass
class Result:
    """
    Result class is used to store the results of a model evaluation in the analysis pipeline.

    Attributes:
    ----------
    metric_name (str): The name of the evaluation metric.
    classifier (str): The name of the classifier used.
    dataset_name (str): The name of the dataset used.
    resampling_method (str): The method used for resampling the data.
    metric_value (float): The value of the evaluation metric.
    no_repeat (int): The number of times the model was trained and evaluated.
    clf_params (Dict): The parameters used for the classifier."""

    metric_name: str
    classifier: str
    dataset_name: str
    resampling_method: str
    metric_value: float
    no_repeat: int
    clf_params: Dict
