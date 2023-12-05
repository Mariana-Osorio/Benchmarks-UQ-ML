import warnings
import os
import dill
import numpy as np
from copy import deepcopy
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from configs.all_models_config import model_keyword_dic, total_config
from metrics import cross_validate_y_score
from case_studies import Replications, ReplicationsList
from models.utils import model_params, iter_or_distr, load_object, \
    euclidean_distance
from collections.abc import Iterable
import time


class Competitor:
    def __init__(self, func, config: dict, name: str = None):
        self.func = func
        model_config, non_model_config = model_params(
            config, return_non_model_params=True)
        self.config = non_model_config
        self.model_config = model_config
        self.model_distr = None
        self.name = config["model_name"]["value"] if name is None else name
        self.pipeline = False if config["pre_processing"] is None else True
        self.hyper_tune = False if config["random"] is False else True
        self.cv_results = None
        self.hyperparam_search = {}

        if not self.hyper_tune:
            if not self.pipeline:
                self.__model = self.func(**self.model_config)
            else:
                if isinstance(config["pre_processing"], Iterable):
                    self.__model = make_pipeline(
                        *config["pre_processing"],
                        self.func(**self.model_config))
                else:
                    self.__model = make_pipeline(
                        config["pre_processing"],
                        self.func(**self.model_config))
        else:
            non_hyperopt_config = {}
            self.model_distr = {}
            for key in self.model_config:
                if not iter_or_distr(self.model_config[key]):
                    non_hyperopt_config[key] = self.model_config[key]
                else:
                    self.model_distr[key] = self.model_config[key]

            if not self.pipeline:
                self.__model = self.func(**non_hyperopt_config)
            else:
                if isinstance(config["pre_processing"], Iterable):
                    self.__model = make_pipeline(
                        *config["pre_processing"],
                        self.func(**non_hyperopt_config))
                else:
                    self.__model = make_pipeline(
                        config["pre_processing"],
                        self.func(**non_hyperopt_config))

                new_model_distr = {}
                for key in self.model_distr:
                    new_model_distr[f'{self.__model.steps[-1][0]}'
                                    f'__{key}'] = self.model_distr[key]
                self.model_distr = new_model_distr

    @classmethod
    def from_name(cls, name: str, config: dict = {}):
        assert name in model_keyword_dic.keys(), \
            f"model_name must be one of {model_keyword_dic.keys()}."
        func = model_keyword_dic[name]
        if "random" not in config.keys() or config["random"] is False:
            config["random"] = False
            config_temp = deepcopy(total_config[name]['default'])
        else:
            config_temp = deepcopy(total_config[name]['random'])
        if "pre_processing" not in config.keys():
            config["pre_processing"] = None

        config = dict(config_temp, **config)

        if config["model_impl"] == "uqpylab-sklearn":
            config["model__uq"] = None
            if name in ["pce", "pck"]:
                if "InputOpts" in config.keys():
                    config["model__InputOpts"] = config.pop("InputOpts")
                elif "Input" in config.keys():
                    config["model__InputOpts"] = config.pop("Input")

        return cls(func, config, name)

    def __repr__(self):
        return f"Competitor(name={self.name}, pipeline={self.pipeline}, " \
                f"hyper_tune={self.hyper_tune})"

    def hyperparameter_tune(self, X, y, n_iter: int = 5,
                            cv: int = 5, random_state: int = 2652124,
                            verbose: int = 0, refit=False):
        if not self.hyper_tune:
            raise AssertionError("Hyperparameter optimization not enabled!")
        if len(y.shape) != 1:
            y = y.ravel()
        clf = RandomizedSearchCV(self.__model, self.model_distr,
                                 n_iter=n_iter,
                                 scoring="neg_root_mean_squared_error",
                                 cv=cv, random_state=random_state,
                                 verbose=verbose, refit=False)
        start = time.time()
        search = clf.fit(X, y)
        search_time = time.time() - start
        self.hyperparam_search["time"] = search_time
        self.hyperparam_search["iters"] = n_iter
        self.hyperparam_search["best_params"] = search.best_params_
        self.hyperparam_search["cv_results_"] = search.cv_results_
        self.model_config.update(search.best_params_)

        if self.pipeline:
            new_model_config = {}
            for key in self.model_config:
                new_model_config[f'{self.__model.steps[-1][0]}'
                                 f'__{key}'] = self.model_config[key]
            self.model_config = new_model_config

        self.model_config.update(search.best_params_)
        self.__model.set_params(**self.model_config)

    def fit(self, X, y, n_iter: int = 10, cv: int = 5,
            random_state: int = 2652124, verbose: int = 0):
        if len(y.shape) != 1:
            y = y.ravel()

        if self.hyper_tune:
            self.hyperparameter_tune(
                X, y, n_iter=n_iter, cv=cv,
                random_state=random_state, verbose=verbose)

        self.model = clone(self.__model)
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            raise AssertionError("Model not trained yet! If competitor was "
                                 "cross-validated, use `predict_cv` instead.")
        return self.model.predict(X)

    def rep_cross_validate(self, replications: Replications,
                           returns: bool = False, verbose: int = 0,
                           hypertune_iter: int = 20,
                           random_state: int = 2652124):

        if verbose > 0 and hasattr(self.__model, "verbose"):
            if self.pipeline:
                setattr(self.__model[-1], "verbose", verbose)
            else:
                setattr(self.__model, "verbose", verbose)

        if self.name in ["mlp", "resnet", "ft_transformer"]:
            X = replications.X.astype(np.float32)
            y = replications.y.astype(np.float32)
        else:
            X = replications.X
            y = replications.y

        if len(y.shape) != 1:
            y = y.ravel()

        replication_ixs = replications.ixs
        if self.hyper_tune:
            if verbose > 0:
                print("Random search hyperparameter optimization")
            self.hyperparameter_tune(
                X, y, n_iter=hypertune_iter, cv=replication_ixs[:1],
                random_state=random_state, verbose=verbose)
            if verbose > 0:
                print(f"Found best hyperparameters in {hypertune_iter} iters.")

        if verbose > 0:
            print(f'------------ Competitor Replications Training ------------'
                  f'\nmodel: {self.name}\nreplications: {replications.num}'
                  f'\ncross-validation: '
                  f'{replications.cv}\nk-folds: '
                  f'{replications.k}\ntotal models: {len(replications.ixs)}\n')

            if replications.k is not None and verbose > 1:
                print(f'In each replication the {self.name}-competitor is'
                      f' {replications.k}-fold cross-\nvalidated on the whole'
                      f' case-study dataset: {replications.k} equal'
                      f'\ncompetitor models are trained and their metrics'
                      f' averaged\non {replications.k} randomly chosen '
                      f'training and testing folds.')
                print(f'- Training set: random subset of '
                      f'{replications.train_N} case-study points\n  ('
                      f'{replications.train_N/len(replications.X)*100:.2f}% of'
                      f' the case-study dataset).\n- Testing set: remaining'
                      f' {replications.test_N} points ('
                      f'{replications.test_N/len(replications.X)*100:.2f}%).')

            elif replications.k is None and verbose > 1:
                print(f'In each replication one {self.name}-competitor is '
                      f'trained\non {replications.train_N} case-study points'
                      f' and tested on an unseen\nvalidation set of '
                      f'{replications.test_N} points.\nSince the case-study'
                      f' model is available,\nthe training set is a different'
                      f' random realization\nof size {replications.train_N} in'
                      f' each replication,\nthe test set is the same random'
                      f' realization of\nsize {replications.test_N} among'
                      f' all replications.')

            print(f'\n\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:'
                  f' Replication model training started')

        scores = cross_validate_y_score(self.__model, X, y, cv=replication_ixs,
                                        return_train_score=True,
                                        verbose=verbose)
        if verbose > 0:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: '
                  f'Replication model training finished\n\n-----------------'
                  f'-----------------------------------------')

        self.cv_results = scores

        if returns:
            return self.cv_results


class CompetitorMetrics():
    def __init__(self, replications: Replications, cv_competitor: Competitor):
        self.replications = replications
        self.cv_competitor = cv_competitor
        self.name = f"{self.cv_competitor.name}_{self.replications.name}"
        self.__metric_names = list(self.cv_competitor.cv_results.keys())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, replication=' \
            f'{self.replications.num}, cv={self.replications.cv},\n' \
            f'                   k={self.replications.k}, total_models=' \
            f'{len(self.replications.ixs)}, total_metrics = ' \
            f'{len(self.__metric_names)})'

    def train_N(self):
        return self.replications.train_N

    def get_metric(self, metric: str, test: bool = True,
                   return_name: bool = False):
        if metric.startswith("test_"):
            test = True
        elif metric.startswith("train_"):
            test = False
        if test and "test_" not in metric and "time" not in metric:
            metric = f'test_{metric}'
        elif not test and "train_" not in metric and "time" not in metric:
            metric = f'train_{metric}'

        if metric in self.__metric_names:
            if return_name:
                return self.cv_competitor.cv_results[metric], metric
            else:
                return self.cv_competitor.cv_results[metric]
        else:
            raise ValueError(f'No metric {metric} available. Available '
                             f'metrics: {self.__metric_names}')

    def get_metric_rep(self, metric: str, rep: int = 0, test: bool = True,
                       return_name: bool = False):
        metric_results, metric = self.get_metric(metric, test=test,
                                                 return_name=True)
        if return_name:
            if self.replications.cv:
                return metric_results[rep*self.replications.k:
                                      (rep+1)*self.replications.k], \
                    f'{metric}_rep{rep}'
            else:
                return metric_results[rep], f'{metric}_rep{rep}'
        else:
            if self.replications.cv:
                return metric_results[rep*self.replications.k:
                                      (rep+1)*self.replications.k]
            else:
                return metric_results[rep]

    def get_metric_func(self, func, metrics: list, metric_info: list = None,
                        name=None, return_name: bool = False):
        metric_values = []
        metric_info = [None]*len(metrics) if metric_info is None \
            else metric_info
        for metric, params in zip(metrics, metric_info):
            if params is not None:
                metric_values.append(self.get_metric(metric, **params))
            else:
                metric_values.append(self.get_metric(metric))

        if return_name:
            name = func.__name__ if name is None else name
            return func(*metric_values), name
        else:
            return func(*metric_values)

    def get_normalized_metric(self, metric: str, test: bool = True,
                              return_name: bool = False):
        if metric.startswith("test_"):
            test = True
        elif metric.startswith("train_"):
            test = False
        if test and "test_" not in metric:
            metric = f'test_{metric}'
        elif not test and "train_" not in metric:
            metric = f'train_{metric}'

        if "rmse" in metric:
            std = self.get_metric(metric.replace("rmse", "var"),
                                  test=test)**0.5
            if return_name:
                return self.get_metric(metric, test=test) / std, \
                    f"{metric}_normalized"
            else:
                return self.get_metric(metric, test=test) / std
        elif "mae" in metric:
            std = self.get_metric(metric.replace("mae", "var"),
                                  test=test)**0.5
            if return_name:
                return self.get_metric(metric, test=test) / std, \
                    f"{metric}_normalized"
            else:
                return self.get_metric(metric, test=test) / std
        elif "mse" in metric:
            var = self.get_metric(metric.replace("mse", "var"),
                                  test=test)
            if return_name:
                return self.get_metric(metric, test=test) / var, \
                    f"{metric}_normalized"
            else:
                return self.get_metric(metric, test=test) / var
        else:
            warnings.warn(f'Returning unnormalized metric {metric}')
            if return_name:
                return self.get_metric(metric, test=test), metric
            else:
                return self.get_metric(metric, test=test)

    def dict_summary(self, metric, metric_data):
        summary = {'metric': metric}

        if not self.replications.cv:
            for i in range(self.replications.num):
                summary[f'replication_{i}'] = metric_data[i]
        else:
            for i in range(self.replications.num):
                summary[f'replication_{i}_cv'] = np.mean(
                    metric_data[i*self.replications.k:
                                (i+1)*self.replications.k])

        summary['mean'] = np.mean(metric_data)
        summary['std'] = np.std(metric_data)
        summary['min'] = np.min(metric_data)
        summary['max'] = np.max(metric_data)

        return summary

    def get_metric_summary(self, metric: str, test: bool = True,
                           normalized: bool = False, func=None, name=None):
        if "y_pred" in metric:
            raise ValueError("Summary not available for y_pred.")
        if normalized and ("mse" in metric or "rmse" in metric
                           or "mae" in metric):
            metric_data, metric = self.get_normalized_metric(metric,
                                                             test=test,
                                                             return_name=True)
        elif func is not None:
            metric_data, metric = self.get_metric_func(func, metric,
                                                       return_name=True,
                                                       name=name)
        else:
            metric_data, metric = self.get_metric(metric, test=test,
                                                  return_name=True)

        summary = self.dict_summary(metric, metric_data)

        return summary

    def summary(self, test: bool = True, times: bool = False,
                normalized: bool = False):
        if test:
            keywords = ["time", "test"] if times else ["test"]
        else:
            keywords = ["time", "train"] if times else ["train"]

        metrics = []
        data = []
        for k in keywords:
            metrics += [m for m in self.__metric_names if k in m]

        for i, metric in enumerate(metrics):
            if "y_pred" in metric:
                continue
            data.append(self.get_metric_summary(metric, test=test))
            if "mse" in metric or "mae" in metric:
                data.append(self.get_metric_summary(metric, test=test,
                                                    normalized=True))
            if "pred" in metric:
                data.append(self.get_metric_summary(metrics[i-1:i+1],
                                                    func=euclidean_distance,
                                                    name=f"{metric}_absdiff"))

        return data

    def save(self, report_dir="", bench_name=None, return_path=False):
        '''
        Saves the benchmark results, CompetitorMetrics object
        in a pickle file in the report_dir directory.
        '''
        if report_dir != "" and not os.path.isdir(report_dir):
            os.mkdir(report_dir)
        bench_name = self.name if bench_name is None else bench_name
        dataset_file = os.path.join(report_dir, f"{bench_name}.obj")
        with open(dataset_file, 'wb') as f:
            dill.dump(self, f, protocol=dill.HIGHEST_PROTOCOL)
            path = os.path.realpath(f.name)
        if return_path:
            return path


class CompetitorBenchmarkResults:
    def __init__(self, metrics_list: list) -> None:
        assert all(x == metrics_list[0].cv_competitor.name for x in
                   [comp.cv_competitor.name for comp in metrics_list]), \
            "All competitors must be of the same type."
        assert all([np.array_equal(metrics_list[0].replications.X, x)
                    for x in [comp.replications.X
                              for comp in metrics_list[1:]]]), \
            "All competitors must be trained on the same dataset."
        self.metrics_list = metrics_list
        self.name = metrics_list[0].cv_competitor.name
        self.train_N = [comp.replications.train_N for comp in metrics_list]

    @classmethod
    def from_objs(cls, obj_paths: list):
        metrics_list = []
        for obj_path in obj_paths:
            metrics_list.append(load_object(obj_path))
        return cls(metrics_list)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(competitor={self.name}' \
            f', train_N={self.train_N})'

    def __len__(self):
        return len(self.metrics_list)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.metrics_list[key]
        else:
            raise TypeError(f'Expected int, got {type(key).__name__}')

    def get_agg_metric(self, metric: str, agg_func):
        data = []
        if metric == "test_mse_norm":
            for i in range(len(self)):
                mse = self[i].get_metric("test_mse")
                mse = mse[~np.isnan(mse)]
                var = self[i].get_metric("test_var")
                var = var[~np.isnan(var)]
                data.append(agg_func(mse/var))

        elif metric == "train_mse_norm":
            for i in range(len(self)):
                mse = self[i].get_metric("train_mse")
                mse = mse[~np.isnan(mse)]
                var = self[i].get_metric("train_var")
                var = var[~np.isnan(var)]
                data.append(agg_func(mse/var))

        elif metric == "test_mae_norm":
            for i in range(len(self)):
                mae = self[i].get_metric("test_mae")
                mae = mae[~np.isnan(mae)]
                std = self[i].get_metric("test_var")**0.5
                std = std[~np.isnan(std)]
                data.append(agg_func(mae/std))

        elif metric == "train_mae_norm":
            for i in range(len(self)):
                mae = self[i].get_metric("train_mae")
                mae = mae[~np.isnan(mae)]
                std = self[i].get_metric("train_var")**0.5
                std = std[~np.isnan(std)]
                data.append(agg_func(mae/std))

        elif metric == "test_rmse_norm":
            for i in range(len(self)):
                rmse = self[i].get_metric("test_rmse")
                rmse = rmse[~np.isnan(rmse)]
                std = self[i].get_metric("test_var")**0.5
                std = std[~np.isnan(std)]
                data.append(agg_func(rmse/std))

        elif metric == "train_rmse_norm":
            for i in range(len(self)):
                rmse = self[i].get_metric("train_rmse")
                rmse = rmse[~np.isnan(rmse)]
                std = self[i].get_metric("train_var")**0.5
                std = std[~np.isnan(std)]
                data.append(agg_func(rmse/std))

        else:
            for i in range(len(self)):
                metric_data = self[i].get_metric(metric)
                metric_data = metric_data[~np.isnan(metric_data)]
                data.append(agg_func(metric_data))
        return np.array(data)


def competitor_benchmark(replications: ReplicationsList,
                         competitor_name: str, config: dict,
                         report_dir="", verbose: int = 0,
                         hypertune_iter: int = 20,
                         return_path: bool = False):
    '''
    Runs a benchmark on a competitor model on a list of replications.
    '''
    benchmark = []
    info = {}
    info['competitor_name'] = competitor_name
    info['competitor_params'] = config
    info['case_study'] = replications.case_study
    info['replication_names'] = replications.names
    info['train_N'] = replications.train_N
    info['saved_objs'] = []
    info["hyperparam_search"] = []
    start = time.time()
    for i, rep in enumerate(replications):
        competitor = Competitor.from_name(competitor_name,
                                          config=config)
        try:
            competitor.rep_cross_validate(rep, hypertune_iter=hypertune_iter,
                                          verbose=verbose)
            info["hyperparam_search"].append(competitor.hyperparam_search)
        except Exception as e:
            print(f"Error in {rep.name}: {e}")
            info["hyperparam_search"].append(competitor.hyperparam_search)
            continue
        metrics = CompetitorMetrics(rep, competitor)
        benchmark.append(metrics)
        obj_path = metrics.save(report_dir=report_dir, return_path=True)
        info['saved_objs'].append(obj_path)
        if verbose > 0:
            print(f'\nReplication {i+1} finished\n')
    total_time = time.time() - start
    info['total_time'] = total_time
    info_file = f"{report_dir}/{info['competitor_name']}_" \
        f"{info['case_study']}_info.txt"
    with open(info_file, 'w') as f:
        for key in info.keys():
            f.write(f"{key}: {info[key]}\n")

    if return_path:
        return CompetitorBenchmarkResults(benchmark), info_file
    else:
        return CompetitorBenchmarkResults(benchmark)
