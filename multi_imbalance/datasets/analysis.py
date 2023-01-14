from copy import deepcopy
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import click
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.base import BaseSampler
from sklearn.pipeline import _name_estimators
import logging
from itertools import product
import numpy as np

from multi_imbalance.datasets.helpers import Result, Config, import_from_string

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
)


class AnalysisPipeline:
    """
    This is a class for an analysis pipeline.
    The __init__() method initializes the object, taking a Config object containing the pipeline configuration
    and an optional csv_path to the results CSV file.
    The run_analysis() method runs the analysis on a set of classifiers, datasets, and resamplers,
    saving the results to the specified output_path.
    The explode_clf_params() method takes a input_path to a CSV file and explodes the clf_params column
    into individual columns, saving the result to output_path.
    Finally, the generate_summary() method takes a query_dict specifying which results to include in the summary
    and an optional aggregate_func to apply to the metric_value column
    and returns a list of Pandas DataFrames containing the summary of the results.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the AnalysisPipeline object.

        :param: config:
            Config object containing the configuration for the pipeline.
        """
        __split_funcs = {"Kfold": self.__KFold_split, "train_test": self.__train_test_split}

        self.__logger = logging.getLogger("AnalysisPipeline")
        self._config = config
        self.__metrics = self._config.metrics
        self.__n_repeats = self._config.n_repeats
        self.__split_params = self._config.split_method[1]
        self.__split_func = __split_funcs[self._config.split_method[0]]
        self.__iter = 0
        self.__chunksize = 10000

    def run_analysis(self, output_path: str, train_without_resampling: bool) -> None:
        """
        This function runs a specified analysis on a set of classifiers, datasets, and resamplers.
        The results of the analysis are saved to the specified output path.

        :param output_path:
            str, the location where the results of the analysis will be saved as a CSV file
        :param train_without_resampling:
            bool, if `True`, the analysis will be run without using resampling on the training set
        """
        self._csv_path = Path(output_path)
        list_of_errors = []

        for clf_data, n, dataset_data in product(self._get_classifier(), range(self.__n_repeats), self._get_dataset()):
            for resampler_data in self._get_resampler(train_without_resampling, dataset_name=dataset_data[0]):
                self._prepare_result(clf_data, dataset_data, resampler_data, output_path, n, list_of_errors)
                self.__iter += 1

        self.__iter = 0
        if list_of_errors:
            self.__logger.error("\n".join(list_of_errors))

    @staticmethod
    def generate_summary(
        query_dict: Dict[str, List[str]],
        csv_path: str,
        save_to_csv: bool = False,
        save_path: Optional[str] = None,
        aggregate_func: Optional[List[Callable]] = None,
    ) -> List[pd.DataFrame]:
        """
        Generate summary of analysis results based on specified query parameters.

        This method generates a summary of the results of an analysis based on the specified query parameters. The `csv_path`
        should be the path to the CSV file containing the results of the analysis. By default, the mean and std functions
        are used for aggregation. If `save_to_csv` is `True`, the summary will be saved to a CSV file. The optional
        `aggregate_func` parameter allows specifying a list of functions that will be applied to the `metric_value` column
        of the results to generate the summary.

        :param query_dict:
            Dict[str, List[str]], a dictionary that specifies the values of
            different columns in the results to include in the summary
        :param csv_path:
            str, the path to the CSV file containing the results of the analysis
        :param save_to_csv:
            bool, optional, if `True`, the summary will be saved to a CSV file
        :param save_path:
            str, optional, the location where the summary csv files should be saved.
            If None summary csv files will be saved in the same location as csv_path
        :param aggregate_func:
            Optional[List[Callable]], optional, a list of functions that will be applied
            to the `metric_value` column of the results to generate the summary
        :return:
            List[pd.DataFrame], a list of Pandas DataFrames containing the summary of the results of the analysis
        """
        chunksize = 1000
        gen = pd.read_csv(csv_path, chunksize=chunksize)
        selected_columns = list(next(gen).columns)
        selected_columns.remove("no_repeat")
        selected_columns.remove("metric_value")
        agg_func_list = ["mean", "std"] if aggregate_func is None else ["mean", "std", *aggregate_func]
        df_list = []
        for i in product(*query_dict.values()):
            df = AnalysisPipeline._search_df_by_query(query_dict, combination=i, csv_path=csv_path, chunksize=chunksize)

            group_df = df[[*selected_columns, "metric_value"]].groupby(by=selected_columns).agg({"metric_value": agg_func_list})
            if save_to_csv:
                if save_path is None:
                    save_path = Path(csv_path).parent
                group_df.reset_index().to_csv(Path(save_path) / ("_".join(i) + ".csv"), index=False)
            df_list.append(group_df)

        return df_list

    @staticmethod
    def generate_posthoc_analysis(
        query_dict: Dict[str, List[str]],
        csv_path: str,
        posthoc_func_list: List[Tuple[Callable, Dict]],
        save_to_csv: bool = False,
        save_path: Optional[str] = None,
    ) -> List[pd.DataFrame]:
        """
        Generates a posthoc analysis of the results of the analysis based on the specified query parameters and posthoc functions.
        `csv_path` should be the path to the CSV file containing the results of the analysis.

        :param query_dict:
            Dict[str, List[str]], a dictionary that specifies the values of different
            columns in the results to include in the posthoc analysis
        :param csv_path:
            str, the path to the CSV file containing the results of the analysis
        :param posthoc_func_list:
            List[Tuple[Callable, Dict]], a list of tuples containing the posthoc functions and their parameters to be applied to the results
        :param save_to_csv:
            bool, optional, if `True`, the posthoc analysis will be saved to a CSV file
        :param save_path:
            str, optional, the location where the summary csv files should be saved.
            If None summary csv files will be saved in the same location as csv_path
        :return:
            List[pd.DataFrame], a list of Pandas DataFrames containing the posthoc analysis of the results
        """
        chunksize = 1000
        gen = pd.read_csv(csv_path, chunksize=chunksize)
        selected_columns = list(next(gen).columns)
        selected_columns.remove("no_repeat")
        selected_columns.remove("metric_value")

        df_list = []
        for i in product(*query_dict.values()):
            df = AnalysisPipeline._search_df_by_query(query_dict, combination=i, csv_path=csv_path, chunksize=chunksize)

            for posthoc_func, params in posthoc_func_list:
                posthoc_df = posthoc_func(df, "metric_value", "resampling_method", **params)
                df_name = posthoc_func.__name__ + "_" + "_".join(i)
                posthoc_df.columns.name = df_name
                if save_to_csv:
                    if save_path is None:
                        save_path = Path(csv_path).parent
                    df_path = save_path / (df_name + ".csv")
                    posthoc_df.to_csv(Path(df_path))
                df_list.append(posthoc_df)

        return df_list

    @property
    def dataset_names(self) -> List[str]:
        """Returns a list of unique dataset names in the CSV file"""
        gen = pd.read_csv(self._csv_path, chunksize=self.__chunksize)
        return list(set(i for df in gen for i in df["dataset_name"].unique()))

    @property
    def metric_names(self) -> List[str]:
        """Returns a list of unique metric names in the CSV file"""
        gen = pd.read_csv(self._csv_path, chunksize=self.__chunksize)
        return list(set(i for df in gen for i in df["metric_name"].unique()))

    @property
    def clf_names(self) -> List[str]:
        """Returns a list of unique classifier names in the CSV file"""
        gen = pd.read_csv(self._csv_path, chunksize=self.__chunksize)
        return list(set(i for df in gen for i in df["classifier"].unique()))

    @property
    def resampling_methods(self) -> List[str]:
        """Returns a list of unique resample method names in the CSV file"""
        gen = pd.read_csv(self._csv_path, chunksize=self.__chunksize)
        return list(set(i for df in gen for i in df["resampling_method"].unique()))

    @property
    def column_names(self) -> List[str]:
        """Returns a list of column names in the CSV file"""
        gen = pd.read_csv(self._csv_path, chunksize=self.__chunksize)
        return list(next(gen).columns)

    @staticmethod
    def _search_df_by_query(query_dict: Dict, combination: Tuple[str, ...], csv_path: str, chunksize: int = 10000) -> pd.DataFrame:
        query_dict = dict(zip(query_dict.keys(), combination))
        query = " & ".join(map(lambda x: f"{x[0]}=='{x[1]}'", query_dict.items()))

        gen = pd.read_csv(csv_path, chunksize=chunksize)
        return pd.concat([df.query(query) for df in gen])

    def _prepare_result(
        self,
        clf_data: Tuple[str, ClassifierMixin, Dict],
        dataset_data: Tuple[str, pd.DataFrame],
        resampler_data: Tuple[str, BaseSampler],
        output_path: str,
        n: int,
        list_of_errors: List[str],
    ) -> None:
        """
        This method prepares the results of running a classifier on a dataset which is resampling by resampler.
        It will compute the specified metrics for each repeat.

        :param clf_data:
            A tuple containing the name of the classifier, the classifier and a dictionary of classifier parameters
        :param dataset_data:
            A tuple containing the name of the dataset and the dataset
        :param resampler_data:
            A tuple containing the name of the resampler and the resampler
        :param output_path:
            str, the path to the output CSV file
        :param n:
            int, the current number of repeat
        :param list_of_errors:
            List[str], a list of error messages to append to if an exception is raised
        """
        clf_name, clf, clf_params = clf_data
        dataset_name, dataset = dataset_data
        resampler_name, resampler = resampler_data
        tmp_clf = deepcopy(clf)
        metric_to_check = {}
        for metric in self.__metrics.keys():
            if not self._check_if_exist_in_csv(output_path, dataset_name, clf_name, resampler_name, metric.__name__, n, clf_params):
                metric_to_check.update({metric: self.__metrics[metric]})

        if metric_to_check != {}:
            try:
                X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
                self.__split_func(
                    X,
                    y,
                    resampler,
                    tmp_clf,
                    metrics=metric_to_check,
                    dataset_name=dataset_name,
                    clf_name=clf_name,
                    resampler_name=resampler_name,
                    n=n,
                    clf_params=clf_params,
                    output_path=output_path,
                )

            except Exception as e:
                list_of_errors.append(f"Raised exception: '{e}' for {dataset_name=}, {resampler_name=} and {clf_name=}")

    def __KFold_split(self, X: np.ndarray, y: np.ndarray, resampler: BaseSampler, clf: ClassifierMixin, **kwargs) -> None:
        """
        This method performs a K-Fold split of the data and computes the specified metrics on the test set for each split.

        :param X:
            numpy.ndarray, the feature matrix
        :param y:
            numpy.ndarray, the target vector
        :param resampler:
            BaseSampler, the resampling method to use
        :param clf:
            ClassifierMixin, the classifier to be evaluated
        :param kwargs:
            Additional keyword arguments, including the list of metrics to be computed, the name of the dataset, classifier, and resampler,
            the current number of repeat, and the parameters of the classifier
        """
        skf = StratifiedKFold(**self.__split_params, random_state=self.__iter)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            if resampler is not None:
                X_train, y_train = resampler.fit_resample(X_train, y_train)

            results = []
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            self._processing_result(results=results, y_test=y_test, y_pred=y_pred, **kwargs)

    def __train_test_split(self, X: np.ndarray, y: np.ndarray, resampler: BaseSampler, clf: ClassifierMixin, **kwargs):
        """
        This method performs a train-test split of the data and computes the specified metrics on the test set.

        :param X:
            numpy.ndarray, the feature matrix
        :param y:
            numpy.ndarray, the target vector
        :param resampler:
            BaseSampler, the resampling method to use
        :param clf:
            ClassifierMixin, the classifier to be evaluated
        :param kwargs:
            Additional keyword arguments, including the list of metrics to be computed, the name of the dataset, classifier, and resampler,
            the current number of repeat, and the parameters of the classifier
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=self.__iter,
            **self.__split_params,
        )
        if resampler is not None:
            X_train, y_train = resampler.fit_resample(X_train, y_train)

        results = []
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self._processing_result(results=results, y_test=y_test, y_pred=y_pred, **kwargs)

    def _processing_result(
        self,
        results: List,
        metrics: Dict[Callable, Dict],
        dataset_name: str,
        clf_name: str,
        resampler_name: str,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        n: int,
        clf_params: Dict,
        output_path: str,
    ):
        """
        This method processes the results of applying a classifier to a dataset using a resampler.

        :param results:
            List, a list to append the computed results to
        :param metrics:
            Dict[Callable, Dict], a dictionary of metrics to be computed, along with any additional parameters for each metric
        :param dataset_name:
            str, the name of the dataset
        :param clf_name:
            str, the name of the classifier
        :param resampler_name:
            str, the name of the resampler
        :param y_test:
            numpy.ndarray, the target vector of the test set
        :param y_pred:
            numpy.ndarray, the predicted target vector of the test set
        :param n:
            int, the current number of repeat
        :param clf_params:
            Dict, the parameters of the classifier
        :param output_path:
            str, the path to the output CSV file

        """
        for metric, params in metrics.items():
            results.append(
                Result(
                    dataset_name=dataset_name,
                    classifier=clf_name,
                    resampling_method=resampler_name,
                    metric_name=metric.__name__,
                    metric_value=metric(y_test, y_pred, **params),
                    no_repeat=n,
                    clf_params=clf_params,
                )
            )
        df_results = pd.DataFrame(results)

        if self._csv_path.exists():
            df_results.to_csv(output_path, mode="a", index=False, header=False)
        else:
            df_results.to_csv(output_path, index=False)

    def _check_if_exist_in_csv(
        self,
        path: str,
        dataset_name: str,
        clf_name: str,
        resampler_name: str,
        metric_name: str,
        n: int,
        clf_params: str,
    ) -> bool:
        """
        Check if a combination of parameters already exists in a CSV file.

        :param path:
            str, The path to the CSV file
        :param dataset_name:
            str, The name of the dataset
        :param clf_name:
            str, The name of the classifier
        :param resampler_name:
            str, The name of the resampler
        :param metric_name:
            str, The name of the metric
        :param n:
            int, Current number of repeat
        :param clf_params:
            str, The classifier parameters
        :return:
            bool, True if the results already exist, False otherwise
        """
        query = (
            f"dataset_name=='{dataset_name}' & classifier=='{clf_name}' & resampling_method=='{resampler_name}'"
            f"""& metric_name=='{metric_name}' & clf_params=="{clf_params}" & no_repeat=={n}"""
        )
        if Path(path).exists():
            gen = pd.read_csv(path, chunksize=self.__chunksize)
            return any([not df.query(query).empty for df in gen])
        return False

    def _get_dataset(self) -> Iterable[Tuple[str, pd.DataFrame]]:
        """
        This method retrieves the datasets specified in the configuration object.

        :return:
            Iterable[Tuple[str, pd.DataFrame]], An iterable of tuples containing the dataset name and the dataset
        """
        for dataset_path in self._config.datasets:
            path = Path(dataset_path)

            if path.is_file() and path.suffix == ".csv":
                yield path.stem, pd.read_csv(path)
            elif path.is_dir():
                dataset_dir = path
                for path in dataset_dir.glob("**/*.csv"):
                    yield path.stem, pd.read_csv(path)
            else:
                raise Exception("Wrong dataset path, should be csv file or dir with csv files")

    def _get_resampler(self, train_without_resampling: bool, dataset_name: str) -> Iterable[Tuple[str, Union[BaseSampler, None]]]:
        """
        This method retrieves the resamplers specified in the configuration object.

        :param train_without_resampling:
            bool, A flag indicating whether to include an option to not use a resampler
        :param dataset_name:
            str, The name of the dataset for which use specific resampler configuration

        :return:
            Iterable[Tuple[str, Union[BaseSampler, None]]], An iterable of tuples containing the resampler name and the resampler
        """
        for resampler, params_dict in self._config.resampling_methods.items():
            if not hasattr(resampler, "fit_resample"):
                raise ValueError("Your resampler must implement fit_resample method")
            if dataset_name and "default" not in params_dict:
                raise KeyError("Must define default params for all datasets or for the specific dataset")

            params = params_dict.get(dataset_name, params_dict.get("default"))
            yield self._get_name(resampler(**params))
        if train_without_resampling:
            yield "Not defined", None

    def _get_classifier(self) -> Iterable[Tuple[str, ClassifierMixin, Dict]]:
        """This method retrieves the classifiers specified in the configuration object.

        :return:
            Iterable[Tuple[str, ClassifierMixin, Dict]], An iterable of tuples containing the classifier name,
            the classifier and a dictionary of classifier parameters"""
        for classifier, params_list in self._config.classifiers.items():
            if not hasattr(classifier, "fit") or not hasattr(classifier, "predict"):
                raise ValueError("Your classifier must implement fit and predict methods")
            for params in params_list:
                yield *self._get_name(classifier(**params)), params

    def _get_name(self, estimator: Union[ClassifierMixin, BaseSampler]) -> Tuple[str, Union[ClassifierMixin, BaseSampler]]:
        return _name_estimators([estimator])[0]


@click.command()
@click.argument("output_path")
@click.option(
    "--run-analysis",
    is_flag=True,
    help="Option specifying whether it should be run analysis pipeline",
)
@click.option(
    "--summary",
    is_flag=True,
    help="Option specifying whether it should be run summary",
)
@click.option(
    "--posthoc-analysis",
    is_flag=True,
    help="Option specifying whether it should be run posthoc analysis",
)
@click.option("--config-json", help="Path to json file which contain config for pipeline analysis")
@click.option("--query-json", help="Path to json file which contain query dict for generating summary")
@click.option("--posthoc-query-json", help="Path to json file which contain query dict for posthoc analysis")
@click.option(
    "--aggregate-json", help="Optional, path to json file which contain paths (in list) to aggregate functions, e.g. ['numpy.max', ...]"
)
@click.option(
    "--posthoc-func-json",
    help="Path to json file which contain dict with paths to posthoc analysis"
    "functions and their params, e.g. {'scikit_posthoc.posthoc_dunn':{}}",
)
@click.option(
    "--train-without-resampling",
    is_flag=True,
    help="Option specifying if the analysis would be run without using resampling",
)
@click.option(
    "--save-to-csv",
    is_flag=True,
    help="Option defines if results from summary should be save to csv",
)
def main(
    output_path,
    run_analysis,
    summary,
    posthoc_analysis,
    config_json,
    query_json,
    posthoc_query_json,
    aggregate_json,
    posthoc_func_json,
    train_without_resampling,
    save_to_csv,
):
    """
    This function helps to use pipeline analysis, summary and posthoc tests by CLI.
    Output path is path to result csv file from analysis pipeline.
    """
    print("Start")
    if run_analysis:
        print("Run analysis pipeline")
        config = Config.from_json(config_json)
        pipeline = AnalysisPipeline(config)
        pipeline.run_analysis(output_path, train_without_resampling)

    if summary:
        print("Run generate summary")
        with open(query_json, "r") as f:
            query_dict = json.load(f)

        aggregate_func = []
        if aggregate_json is not None:
            with open(aggregate_json, "r") as f:
                aggregate_func_paths = json.load(f)
            aggregate_func = list(map(import_from_string, aggregate_func_paths))

        AnalysisPipeline.generate_summary(query_dict, output_path, save_to_csv, aggregate_func=aggregate_func)

    if posthoc_analysis:
        print("Run generate posthoc analysis")

        with open(posthoc_query_json, "r") as f:
            query_dict = json.load(f)

        with open(posthoc_func_json, "r") as f:
            posthoc_func_paths = json.load(f)
        posthoc_func = [[import_from_string(func_path), params] for func_path, params in posthoc_func_paths.items()]

        AnalysisPipeline.generate_posthoc_analysis(query_dict, output_path, posthoc_func_list=posthoc_func, save_to_csv=save_to_csv)

    print("Done")


if __name__ == "__main__":  # pragma no cover
    main()
