from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from imblearn.base import BaseSampler
from sklearn.pipeline import _name_estimators
import logging
from itertools import product


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s %(message)s", datefmt="%d.%m.%Y %H:%M:%S")


@dataclass
class Config:
    datasets: List[str]
    classifiers: Dict[ClassifierMixin, List[Dict]]
    resample_methods: Dict[BaseSampler, Dict[str, Dict]]
    metrics: Dict[Callable, Dict]
    n_repeats: int
    stratifiedkfold_params: Dict

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
    clf_params: Dict


class AnalysisPipeline:
    def __init__(self, config: Config) -> None:
        self.__logger = logging.getLogger("AnalysisPipeline")
        self._config = config
        self.__metrics = self._config.metrics
        self.__n_repeats = self._config.n_repeats
        self.__stratifiedkfold_params = self._config.stratifiedkfold_params
        self.__iter = 0
        self.__chunksize = 10000

    def run_analysis(self, output_path: str, train_without_resampling: bool) -> None:
        self._output_path = Path(output_path)
        list_of_errors = []

        for clf_data, n, dataset_data in product(self._get_classifier(), range(self.__n_repeats), self._get_dataset()):
            for resampler_data in self._get_resampler(train_without_resampling, dataset_name=dataset_data[0]):
                self._prepare_result(clf_data, dataset_data, resampler_data, output_path, n, list_of_errors)
                self.__iter += 1

        self.__iter = 0
        if list_of_errors:
            self.__logger.error("\n".join(list_of_errors))

    def explode_clf_params(self, input_path: str, output_path: str) -> None:
        df_results = pd.read_csv(input_path)
        df_results = pd.concat(
            [df_results.drop(columns="clf_params"), df_results["clf_params"].apply(lambda x: dict(eval(x))).apply(pd.Series)], axis=1
        )
        df_results.to_csv(output_path, index=False)

    def generate_summary(self, query_dict: Dict[str, List[str]], aggregate_func: Union[List[Callable], None] = None) -> List[pd.DataFrame]:
        selected_columns = self.column_names
        selected_columns.remove("no_repeat")
        selected_columns.remove("metric_value")
        agg_func_list = ["mean", "std"] if aggregate_func is None else ["mean", "std", *aggregate_func]
        df_list = []
        for i in product(*query_dict.values()):
            tmp_dict = dict(zip(query_dict.keys(), i))
            query = " & ".join(map(lambda x: f"{x[0]}=='{x[1]}'", tmp_dict.items()))

            gen = pd.read_csv(str(self._output_path), chunksize=self.__chunksize)
            df = pd.concat([df.query(query) for df in gen])

            group_df = df[[*selected_columns, "metric_value"]].groupby(by=selected_columns[:-1]).agg({"metric_value": agg_func_list})
            df_list.append(group_df)

        return df_list

    @property
    def dataset_names(self) -> List[str]:
        gen = pd.read_csv(str(self._output_path), chunksize=self.__chunksize)
        return list(set(i for df in gen for i in df["dataset_name"].unique()))

    @property
    def metric_names(self) -> List[str]:
        gen = pd.read_csv(str(self._output_path), chunksize=self.__chunksize)
        return list(set(i for df in gen for i in df["metric_name"].unique()))

    @property
    def clf_names(self) -> List[str]:
        gen = pd.read_csv(str(self._output_path), chunksize=self.__chunksize)
        return list(set(i for df in gen for i in df["classifier"].unique()))

    @property
    def resampling_methods(self) -> List[str]:
        gen = pd.read_csv(str(self._output_path), chunksize=self.__chunksize)
        return list(set(i for df in gen for i in df["resampling_method"].unique()))

    @property
    def column_names(self) -> List[str]:
        gen = pd.read_csv(str(self._output_path), chunksize=self.__chunksize)
        return list(next(gen).columns)

    def _prepare_result(
        self,
        clf_data: Tuple[str, ClassifierMixin, Dict],
        dataset_data: Tuple[str, pd.DataFrame],
        resampler_data: Tuple[str, BaseSampler],
        output_path: str,
        n: int,
        list_of_errors: List[str],
    ) -> None:
        try:
            clf_name, clf, clf_params = clf_data
            dataset_name, dataset = dataset_data
            resampler_name, resampler = resampler_data

            tmp_clf = deepcopy(clf)

            X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

            skf = StratifiedKFold(**self.__stratifiedkfold_params, random_state=self.__iter)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                if resampler is not None:
                    X_train, y_train = resampler.fit_resample(X_train, y_train)

                results = []
                tmp_clf.fit(X_train, y_train)
                y_pred = tmp_clf.predict(X_test)

                for metric, params in self.__metrics.items():
                    results.append(
                        Result(
                            dataset_name,
                            clf_name,
                            resampler_name,
                            metric.__name__,
                            metric(y_test, y_pred, **params),
                            n,
                            clf_params,
                        )
                    )
                df_results = pd.DataFrame(results)

                if self._output_path.exists():
                    df_results.to_csv(output_path, mode="a", index=False, header=False)
                else:
                    df_results.to_csv(output_path, index=False)

        except Exception as e:
            list_of_errors.append(f"Raised exception '{e}' for {dataset_name=}, {resampler_name=} and {clf_name=}")

    def _get_dataset(self) -> Iterable[Tuple[str, pd.DataFrame]]:
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

    def _get_resampler(self, train_without_resampling: bool, dataset_name: str) -> Iterable[Tuple[str, Union[BaseSampler, None]]]:
        for resampler, params_dict in self._config.resample_methods.items():
            if not hasattr(resampler, "fit_resample"):
                raise ValueError("Your resampler must implement fit_resample method")
            if dataset_name and "all" not in params_dict:
                raise KeyError("Must define params for all datasets or for the specific dataset")

            params = params_dict.get(dataset_name, params_dict.get("all"))
            yield self._get_name(resampler(**params))
        if train_without_resampling:
            yield "Not defined", None

    def _get_classifier(self) -> Iterable[Tuple[str, ClassifierMixin, Dict]]:
        for classifier, params_list in self._config.classifiers.items():
            if not hasattr(classifier, "fit") or not hasattr(classifier, "predict"):
                raise ValueError("Your classifier must implement fit and predict methods")
            for params in params_list:
                yield *self._get_name(classifier(**params)), params

    def _get_name(self, estimator: Union[ClassifierMixin, BaseSampler]) -> Tuple[str, Union[ClassifierMixin, BaseSampler]]:
        return _name_estimators([estimator])[0]
