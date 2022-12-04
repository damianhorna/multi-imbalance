from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.base import BaseSampler
from sklearn.pipeline import _name_estimators
import logging

from multi_imbalance.resampling.global_cs import GlobalCS
from multi_imbalance.resampling.soup import SOUP
from multi_imbalance.resampling.spider import SPIDER3
from multi_imbalance.resampling.mdo import MDO
from multi_imbalance.resampling.static_smote import StaticSMOTE

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s %(message)s", datefmt="%d.%m.%Y %H:%M:%S")


@dataclass
class Config:
    datasets: List[str]
    classifiers: Dict[Union[str, ClassifierMixin], List[Dict]]
    resample_methods: Dict[Union[str, BaseSampler], Dict]
    metrics: Dict[Callable, Dict]
    n_repeats: int
    train_test_split_kwargs: Dict

    @classmethod
    def from_dict(cls, config: Dict) -> "Config":
        return cls(**config)


@dataclass
class Result:
    dataset_name: str
    classifier: str
    resampling_method: str
    metric_name: str
    metric_value: float
    no_repeat: int
    kwargs: Dict


class AnalysisPipeline:
    _allowed_resampling = ["globalCS", "StaticSMOTE", "SOUP", "spider3", "MDO"]
    _allowed_classifiers = ["tree", "NB", "KNN"]

    def __init__(self, configs: List[Config]) -> None:
        self.__logger = logging.getLogger("AnalysisPipeline")
        self._configs = configs
        self.__resampling_methods = {"globalCS": GlobalCS, "StaticSMOTE": StaticSMOTE, "SOUP": SOUP, "spider3": SPIDER3, "MDO": MDO}
        self.__classifiers = {"tree": DecisionTreeClassifier, "NB": GaussianNB, "KNN": KNeighborsClassifier}

    def run_analysis(self, output_path: str, explode_clf_kwargs: bool, train_without_resampling: bool):
        self._output_path = Path(output_path)
        for config in self._configs:
            self._config = config
            self.__metrics = self._config.metrics
            self.__n_repeats = self._config.n_repeats
            self.__tts_kwargs = self._config.train_test_split_kwargs
            for clf_name, clf, clf_kwargs in self._get_classifier():
                for n in range(1, self.__n_repeats + 1):
                    for dataset_name, dataset in self._get_dataset():
                        for resampler_name, resampler in self._get_resampler():
                            tmp_clf = deepcopy(clf)
                            X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, **self.__tts_kwargs)

                            try:
                                X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
                                results = []
                                tmp_clf.fit(X_train_res, y_train_res)
                                y_pred = tmp_clf.predict(X_test)

                                for metric, kwargs in self.__metrics.items():
                                    results.append(
                                        Result(
                                            dataset_name,
                                            clf_name,
                                            resampler_name,
                                            metric.__name__,
                                            metric(y_test, y_pred, **kwargs),
                                            n,
                                            clf_kwargs,
                                        )
                                    )
                                df_results = pd.DataFrame(results)
                                if self._output_path.exists():
                                    df_results.to_csv(output_path, mode="a", index=False, header=False)
                                else:
                                    df_results.to_csv(output_path, index=False)
                            except Exception as e:
                                self.__logger.error(f"Raised exception '{e}' for {dataset_name=}, {resampler_name=} and {clf_name=}")

                        if train_without_resampling:
                            tmp_clf = deepcopy(clf)
                            results = []
                            X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, **self.__tts_kwargs)
                            tmp_clf.fit(X_train, y_train)
                            y_pred = tmp_clf.predict(X_test)

                            for metric, kwargs in self.__metrics.items():
                                results.append(
                                    Result(
                                        dataset_name,
                                        clf_name,
                                        "Not defined",
                                        metric.__name__,
                                        metric(y_test, y_pred, **kwargs),
                                        n,
                                        clf_kwargs,
                                    )
                                )
                            df_results = pd.DataFrame(results)
                            if self._output_path.exists():
                                df_results.to_csv(output_path, mode="a", index=False, header=False)
                            else:
                                df_results.to_csv(output_path, index=False)

            if explode_clf_kwargs:
                df_results = pd.read_csv(output_path)
                df_results = pd.concat(
                    [df_results.drop(columns="kwargs"), df_results["kwargs"].apply(lambda x: dict(eval(x))).apply(pd.Series)], axis=1
                )
                df_results.to_csv(output_path, index=False)

    def _get_dataset(self) -> Tuple[str, pd.DataFrame]:
        for dataset_path in self._config.datasets:
            path = Path(dataset_path)

            if path.is_file() and path.suffix == ".csv":
                yield path.stem, pd.read_csv(str(path))
            elif path.is_dir():
                dataset_dir = path
                for path in dataset_dir.glob("**/*.csv"):
                    yield path.stem, pd.read_csv(str(path))
            else:
                raise Exception("Wrong dataset path, should be csv file or dir with csv files")

    def _get_resampler(self) -> List[Tuple[str, BaseSampler]]:
        for resampler, kwargs in self._config.resample_methods.items():
            if isinstance(resampler, str):
                if resampler not in AnalysisPipeline._allowed_resampling:
                    raise ValueError(
                        "Unknown resample method: %s, expected to be one of %s" % (resampler, AnalysisPipeline._allowed_resampling)
                    )
                yield resampler, self.__resampling_methods[resampler](**kwargs)
            else:
                if not hasattr(resampler, "fit_resample"):
                    raise ValueError("Your resampler must implement fit_resample method")
                yield self._get_name(resampler(**kwargs))

    def _get_classifier(self) -> List[Tuple[str, ClassifierMixin, Dict]]:
        for classifier, kwargs_list in self._config.classifiers.items():
            if isinstance(classifier, str):
                if classifier not in AnalysisPipeline._allowed_classifiers:
                    raise ValueError(
                        "Unknown classifier: %s, expected to be one of %s" % (classifier, AnalysisPipeline._allowed_classifiers)
                    )
                for kwargs in kwargs_list:
                    yield classifier, self.__classifiers[classifier](**kwargs), kwargs
            else:
                if not hasattr(classifier, "fit") or not hasattr(classifier, "predict"):
                    raise ValueError("Your classifier must implement fit and predict methods")
                for kwargs in kwargs_list:
                    yield *self._get_name(classifier(**kwargs)), kwargs

    def _get_name(self, estimator: Union[ClassifierMixin, BaseSampler]) -> Tuple[str, Union[ClassifierMixin, BaseSampler]]:
        return _name_estimators([estimator])[0]
