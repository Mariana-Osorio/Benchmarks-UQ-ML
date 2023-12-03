import typing as ty
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import pandas as pd
import dataframe_image as dfi
import os
from sklearn.model_selection import cross_validate


def calculate_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    y_info: ty.Optional[ty.Dict[str, ty.Any]] = None,
    i: int = [0],
    write_y_pred: bool = False
) -> ty.Dict[str, float]:
    '''
    Calculates regression metrics for test and train data.

    Parameters:
        y (array): Dependent variable.
        prediction (array): Predicted dependent variable.
        y_info (dict): Dictionary with information about y.

    Returns:
        metric (dict): Dictionary with the following keys:
            mse (float): Mean squared error.
            rmse (float): Root mean squared error.
            mae (float): Mean absolute error.
            r2 (float): Coefficient of determination.
            mean (float): Mean of y.
            pred_mean (float): Mean of predicted y.
            std (float): Standard deviation of y.
            pred_std (float): Standard deviation of predicted y.

    '''
    mse = skm.mean_squared_error(y, prediction)
    rmse = mse ** 0.5
    mae = skm.mean_absolute_error(y, prediction)
    r2 = skm.r2_score(y, prediction)
    mean = np.mean(y)
    pred_mean = np.mean(prediction)
    var = np.var(y)
    pred_var = np.var(prediction)
    i[0] += 1
    if write_y_pred:
        if len(prediction.shape) != 1:
            prediction = prediction.ravel()
        if not os.path.isfile("y_pred.txt"):
            with open("y_pred.txt", "w") as file:
                for y_pred in prediction:
                    file.write(f"{i[0]},{y_pred}\n")
        else:
            with open("y_pred.txt", "a") as file:
                for y_pred in prediction:
                    file.write(f"{i[0]},{y_pred}\n")

    if y_info:
        if y_info['policy'] == 'mean_std':
            rmse *= y_info['std']
        else:
            assert False
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'mean': mean, 'pred_mean': pred_mean, 'var': var,
            'pred_var': pred_var, 'y_pred_i': i[0]}


def score_competitor(estimator, X, y, write_y_pred=True):
    '''
    Calculates regression metrics for an estimator and
    returns them as the score dictionary.

    Parameters:
        estimator (sklearn estimator): Estimator to score.
        X (array): Independent variable.
        y (array): Dependent variable.

    Returns:
        metric (dict): Dictionary with the following keys:
            mse (float): Mean squared error.
            rmse (float): Root mean squared error.
            mae (float): Mean absolute error.
            r2 (float): Coefficient of determination.
            mean (float): Mean of y.
            pred_mean (float): Mean of predicted y.
            std (float): Standard deviation of y.
            pred_std (float): Standard deviation of predicted y.

    '''
    prediction = estimator.predict(X)
    return calculate_metrics(y, prediction, None, write_y_pred=write_y_pred)


def cross_validate_y_score(estimator, X, y, cv=5, scoring=score_competitor,
                           return_train_score=False, verbose=0):
    '''
    Calculates regression metrics for an estimator
    using cross validation and returns them as the score dictionary
    including the y_prediction for test data per fold.
    '''
    # Get cross-validation metrics
    scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring,
                            return_train_score=return_train_score,
                            verbose=verbose)
    # Scoring function writes y_pred.txt with predictions per fold
    y_pred_data = np.genfromtxt("y_pred.txt", delimiter=",")
    # Get test_y_pred per fold
    scores["test_y_pred"] = []
    for i in scores["test_y_pred_i"]:
        scores["test_y_pred"].append(y_pred_data[y_pred_data[:, 0] == i, 1])
    del scores["test_y_pred_i"]
    # Get train_y_pred per fold
    if "train_y_pred_i" in scores:
        scores["train_y_pred"] = []
        for i in scores["train_y_pred_i"]:
            scores["train_y_pred"].append(
                y_pred_data[y_pred_data[:, 0] == i, 1])
        del scores["train_y_pred_i"]
    # Remove y_pred.txt
    os.remove("y_pred.txt")

    return scores


metric_titles = {
    'fit_time': 'training time',
    'score_time': 'prediction time',
    'test_mse': 'MSE on test data',
    'test_mse_norm': 'normalized MSE on test data',
    'train_mse': 'MSE on train data',
    'train_mse_norm': 'normalized MSE on train data',
    'test_rmse': 'RMSE on test data',
    'test_rmse_norm': 'normalized RMSE on test data',
    'train_rmse': 'RMSE on train data',
    'train_rmse_norm': 'normalized RMSE on train data',
    'test_mae': 'MAE on test data',
    'test_mae_norm': 'normalized MAE on test data',
    'train_mae': 'MAE on train data',
    'train_mae_norm': 'normalized MAE on train data',
    'test_r2': 'R2 on test data',
    'train_r2': 'R2 on train data',
    'test_mean': 'Mean of test data',
    'train_mean': 'Mean of train data',
    'test_var': 'Variance of test data',
    'train_var': 'Variance of train data'
}


def plot_benchmarks(benchmark_list: list, metric, name: str = "Benchmark",
                    log=False, ):
    '''
    Plots the benchmark results.

    Parameters:
        benchmark_list (list): List of benchmark results.
        metric (str): Metric to plot.

    '''

    import pandas as pd

    cmap = plt.get_cmap("tab10")
    for i, model in enumerate(benchmark_list):
        plt.plot(model.train_N, pd.DataFrame(model.metric(metric))["mean"],
                 label=model.name, marker='o', color=cmap(i))
        if not log:
            plt.errorbar(model.train_N,
                         pd.DataFrame(model.metric(metric))["mean"],
                         yerr=pd.DataFrame(model.metric(metric))["std"],
                         fmt='o', color=cmap(i))
    plt.legend()
    plt.xlabel('train_N')
    plt.xticks(model.train_N)
    if log:
        plt.ylabel(f'log({metric_titles[metric]})')
        plt.yscale('log')
    plt.title(f'{name}: {metric_titles[metric]}')


def get_benchmark_metrics(benchmark_list: list):
    metrics = ['fit_time', 'score_time', 'test_mse', 'test_mse_norm',
               'train_mse', 'train_mse_norm', 'test_rmse', 'test_rmse_norm',
               'train_rmse', 'train_rmse_norm', 'test_mae', 'test_mae_norm',
               'train_mae', 'train_mae_norm', 'test_r2', 'train_r2',
               'test_mean', 'train_mean', 'test_pred_mean', 'train_pred_mean',
               'test_var', 'train_var', 'test_pred_var', 'train_pred_var']
    model_data = {}
    for i, model in enumerate(benchmark_list):
        model_data[model.name] = {}
        for metric in metrics:
            model_data[model.name][metric] = {}
            model_data[model.name][metric]['train_N'] = \
                np.array(model.train_N)
            model_data[model.name][metric]['mean'] = \
                model.get_agg_metric(metric, np.mean)
            model_data[model.name][metric]['std'] = \
                model.get_agg_metric(metric, np.std)
    return model_data


def plot_metric(metric: str, benchmark_list: list, name: str = "Benchmark",
                model_data: dict = None, log: bool = False):
    metrics = ['fit_time', 'score_time', 'test_mse', 'test_mse_norm',
               'train_mse', 'train_mse_norm', 'test_rmse', 'test_rmse_norm',
               'train_rmse', 'train_rmse_norm', 'test_mae', 'test_mae_norm',
               'train_mae', 'train_mae_norm', 'test_r2', 'train_r2']
    if model_data is None:
        model_data = get_benchmark_metrics(benchmark_list)
    cmap = plt.get_cmap("tab10")
    if metric in metrics:
        for i, model in enumerate(benchmark_list):
            plt.plot(model_data[model.name][metric]['train_N'],
                     model_data[model.name][metric]['mean'],
                     label=model.name, marker='o', color=cmap(i))
            if not log:
                plt.errorbar(model_data[model.name][metric]['train_N'],
                             model_data[model.name][metric]['mean'],
                             yerr=model_data[model.name][metric]['std'],
                             fmt='o', color=cmap(i))

    elif metric == "test_mean":
        log = False
        for i, model in enumerate(benchmark_list):
            plt.plot(model_data[model.name]['test_pred_mean']['train_N'],
                     model_data[model.name]['test_pred_mean']['mean'],
                     label=model.name, marker='o', color=cmap(i))
            plt.errorbar(model_data[model.name]['test_pred_mean']['train_N'],
                         model_data[model.name]['test_pred_mean']['mean'],
                         yerr=model_data[model.name]['test_pred_mean']['std'],
                         fmt='o', color=cmap(i))
        plt.plot(model_data[model.name][metric]['train_N'],
                 model_data[model.name][metric]['mean'],
                 label="test_mean", color='black')

    elif metric == "train_mean":
        log = False
        for i, model in enumerate(benchmark_list):
            plt.plot(model_data[model.name]['train_pred_mean']['train_N'],
                     model_data[model.name]['train_pred_mean']['mean'],
                     label=model.name, marker='o', color=cmap(i))
            plt.errorbar(model_data[model.name]['train_pred_mean']['train_N'],
                         model_data[model.name]['train_pred_mean']['mean'],
                         yerr=model_data[model.name]['train_pred_mean']['std'],
                         fmt='o', color=cmap(i))
        plt.plot(model_data[model.name][metric]['train_N'],
                 model_data[model.name][metric]['mean'],
                 label="train_mean", color='black')

    elif metric == "test_var":
        log = False
        for i, model in enumerate(benchmark_list):
            plt.plot(model_data[model.name]['test_pred_var']['train_N'],
                     model_data[model.name]['test_pred_var']['mean'],
                     label=model.name, marker='o', color=cmap(i))
            plt.errorbar(model_data[model.name]['test_pred_var']['train_N'],
                         model_data[model.name]['test_pred_var']['mean'],
                         yerr=model_data[model.name]['test_pred_var']['std'],
                         fmt='o', color=cmap(i))
        plt.plot(model_data[model.name][metric]['train_N'],
                 model_data[model.name][metric]['mean'],
                 label="test_var", color='black')

    elif metric == "train_var":
        log = False
        for i, model in enumerate(benchmark_list):
            plt.plot(model_data[model.name]['train_pred_var']['train_N'],
                     model_data[model.name]['train_pred_var']['mean'],
                     label=model.name, marker='o', color=cmap(i))
            plt.errorbar(model_data[model.name]['train_pred_var']['train_N'],
                         model_data[model.name]['train_pred_var']['mean'],
                         yerr=model_data[model.name]['train_pred_var']['std'],
                         fmt='o', color=cmap(i))
        plt.plot(model_data[model.name][metric]['train_N'],
                 model_data[model.name][metric]['mean'],
                 label="train_var", color='black')

    plt.legend(title="Model")
    plt.xlabel('train_N')
    plt.ylabel(f'{metric_titles[metric]}')
    if log:
        plt.ylabel(f'log({metric_titles[metric]})')
        plt.yscale('log')
    plt.xticks(model.train_N)
    plt.xticks(rotation=90)
    plt.title(f'{name}: {metric_titles[metric]}')


def y_y_plot(benchmark_list: list, name: str = "Benchmark", figsize=(14, 10),
             title_pos=0.95):
    fig, axs = plt.subplots(len(benchmark_list), len(benchmark_list[0]),
                            figsize=figsize, squeeze=False,
                            sharex=True, sharey=True)
    cmap = plt.get_cmap("tab20")
    for k, benchmark in enumerate(benchmark_list):
        for j, model in enumerate(benchmark):
            for rep_num in range(model.replications.num):
                y_pred_folds = model.get_metric_rep("test_y_pred", rep_num)
                if model.replications.cv:
                    for i in range(len(y_pred_folds)):
                        true_y = model.replications.y[
                            model.replications.ixs[i][1]]
                        if len(y_pred_folds) != 0:
                            axs[k, j].scatter(
                                true_y, y_pred_folds[i], color=cmap(i),
                                label=f"fold={i}", s=5)
                else:
                    true_y = model.replications.y[model.replications.ixs[
                        rep_num][1]]
                    if len(y_pred_folds) != 0:
                        axs[k, j].scatter(
                            true_y, y_pred_folds, color=cmap(rep_num),
                            label=f"rep={rep_num}", s=5)

            if j == 0:
                axs[k, j].set_ylabel("Predicted y")
            if j == len(benchmark) - 1:
                axs[k, j].yaxis.set_label_position("right")
                axs[k, j].set_ylabel(f"{model.cv_competitor.name}",
                                     rotation=270,
                                     labelpad=12, fontsize=11)
            if k == len(benchmark_list) - 1:
                axs[k, j].set_xlabel("True y")

            if model.replications.cv and k == 0:
                axs[k, j].set_title(
                    f"train_N={model.replications.train_N}"
                    f"\n{model.replications.k}-fold")
                if j == len(benchmark) - 1:
                    axs[k, j].legend(bbox_to_anchor=(1.15, 1),
                                     loc='upper left')
            elif not model.replications.cv and k == 0:
                axs[k, j].set_title(
                    f"train_N={model.replications.train_N}")
                if j == len(benchmark) - 1:
                    axs[k, j].legend(bbox_to_anchor=(1.15, 1),
                                     loc='upper left')

    if model.replications.cv:
        plt.suptitle(f"{name}: Cross-validated Y-Y plot", fontsize=14,
                     y=title_pos)
    else:
        plt.suptitle(f"{name}: Y-Y plot", fontsize=14, y=title_pos)
    plt.xlim([true_y.min(), true_y.max()])
    plt.ylim([true_y.min(), true_y.max()])
    ticks = np.round(np.linspace(true_y.min(), true_y.max(), 4), 1)
    plt.xticks(ticks)
    plt.yticks(ticks)


def y_pred_dep_pairplot(benchmark_list: list, name: str = "Benchmark"):
    figs = []
    for benchmark in benchmark_list:
        X_num = benchmark.replications[0].X.shape[1]
        fig, axs = plt.subplots(X_num,
                                len(benchmark),
                                figsize=(2*len(benchmark), 2*X_num))
        cmap = plt.get_cmap("tab20")
        c = 0
        for x_num in range(X_num):
            for j, model in enumerate(benchmark):
                y_pred_folds = model.get_metric_rep("test_y_pred")
                for i in range(len(y_pred_folds)):
                    X = model.replications.X[model.replications[i][1], x_num]
                    true_y = model.replications.y[model.replications[i][1]]
                    axs[x_num, j].scatter(X, true_y,
                                          color=cmap(c), label="true_y", s=5)
                    c += 1
                    axs[x_num, j].scatter(X, y_pred_folds[i],
                                          color=cmap(c), label="pred_y", s=5)
                    c -= 1
                    axs[x_num, j].set_xlabel(
                        f"{model.replications.X_names[x_num]}")
                    axs[x_num, j].set_title(
                        f"{model.name}, {model.replications.k}-fold")
            handles, labels = axs[x_num, j].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[x_num, j].legend(by_label.values(), by_label.keys(),
                                 loc='upper center', bbox_to_anchor=(1.8, 1))
            c += 2
        fig.supylabel(f"{model.replications.y_name[0]}")
        plt.suptitle(
            f"{name}: Cross-validated {model.name} predictions - pairplots",
            y=0.99)
        plt.tight_layout(pad=2.0)
        figs.append(fig)
    return figs


def plot_benchmark_metrics(benchmark_list: list, name: str = "Benchmark",
                           save_dir: str = None):
    metrics = ['fit_time', 'score_time', 'test_mse', 'test_mse_norm',
               'train_mse', 'train_mse_norm', 'test_rmse', 'test_rmse_norm',
               'train_rmse', 'train_rmse_norm', 'test_mae', 'test_mae_norm',
               'train_mae', 'train_mae_norm', 'test_r2', 'train_r2',
               'test_mean', 'train_mean', 'test_var', 'train_var']
    model_data = get_benchmark_metrics(benchmark_list)
    for metric in metrics:
        log = True if "mse" in metric or "mae" in metric else False
        plot_metric(metric, benchmark_list, name,
                    model_data=model_data, log=log)
        if save_dir is not None:
            plt.savefig(f"{save_dir}/{name}_{metric}.png", bbox_inches='tight')
            plt.show()
        else:
            plt.show()
    y_y_plot(benchmark_list, name)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/{name}_y_y_plot.png", bbox_inches='tight')
        plt.show()
    else:
        plt.show()


def hypertune_dict(metrics):
    hyperparam_tune = {}
    hyperparam_tune["train_N"] = []
    hyperparam_tune["time"] = []
    hyperparam_tune["best_score"] = []
    hyperparam_tune["best_params"] = []
    for metric in metrics:
        results = metric.cv_competitor.hyperparam_search["cv_results_"]
        train_N = metric.replications.train_N

        hyperparam_tune["train_N"].append(train_N)
        hyperparam_tune["time"].append(
            metric.cv_competitor.hyperparam_search["time"])
        hyperparam_tune["best_score"].append(
            results["mean_test_score"][
                np.argwhere(results["rank_test_score"] == 1)[0]][0])
        hyperparam_tune["best_params"].append(
            metric.cv_competitor.hyperparam_search["best_params"])
        hyperparam_tune[train_N] = {}
        hyperparam_tune[train_N]["iterations"] = np.array(
            range(1, len(results["mean_test_score"])+1))
        hyperparam_tune[train_N]["mean_fit_time"] = \
            results["mean_fit_time"]
        hyperparam_tune[train_N]["mean_test_score"] = \
            results["mean_test_score"]
        best_score = []
        for score in results["mean_test_score"]:
            prev_score = best_score[-1] if len(best_score) > 0 else np.nan
            if np.isnan(prev_score) or score > prev_score:
                best_score.append(score)
            else:
                best_score.append(prev_score)
        hyperparam_tune[train_N]["best_score"] = np.array(best_score)
        hyperparam_tune[train_N]["rank_test_score"] = \
            results["rank_test_score"]
        for key in results:
            if key.startswith("param_"):
                hyperparam_tune[train_N][key] = results[key]

    return hyperparam_tune


def df_hypertune(benchmark, save_dir: str = None, name: str = "Benchmark"):
    results = hypertune_dict(benchmark)
    df = pd.DataFrame()
    df["train_N"] = results["train_N"]
    df["score"] = results["best_score"]
    df = pd.concat([df, pd.DataFrame(results["best_params"])], axis=1)

    if save_dir is not None:
        dfi.export(df,
                   os.path.join(save_dir,
                                f"{name}_{benchmark.name}_hypertune_df.png"),
                   table_conversion='chrome',
                   chrome_path='/usr/bin/brave-browser')

    return df


def plot_hypertune(benchmark_list: list, name: str = "Benchmark",
                   save_dir: str = None):
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    dfs = []
    for i, benchmark in enumerate(benchmark_list):
        dfs.append(df_hypertune(benchmark, save_dir=save_dir, name=name))
        hyperparam_data = hypertune_dict(benchmark)
        axs[0].plot(hyperparam_data["train_N"], hyperparam_data["best_score"],
                    label=benchmark.name, marker='o', color=cmap(i))
        axs[0].set_xlabel("Train N")
        axs[0].set_ylabel("Best Score (-MSE)")
        axs[0].set_title(f"{name}: hyperparameter tuning -MSE")
        axs[0].legend()
        axs[0].set_xticks(hyperparam_data["train_N"])
        axs[0].set_xticklabels(
            axs[0].get_xticklabels(), rotation=90, ha='right')
        axs[1].plot(hyperparam_data["train_N"], hyperparam_data["time"],
                    label=benchmark.name, marker='o', color=cmap(i))
        axs[1].set_xlabel("Train N")
        axs[1].set_ylabel("Time (s)")
        axs[1].set_title(f"{name}: hyperparameter tuning time")
        axs[1].legend()
        axs[1].set_xticks(hyperparam_data["train_N"])
        axs[1].set_xticklabels(
            axs[1].get_xticklabels(), rotation=90, ha='right')

    if save_dir is not None:
        plt.savefig(f"{save_dir}/{name}_hypertune.png", bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    return dfs


def plot_categorical_params(benchmark, param_name: str,
                            figsize=(6, 16), title_pos=0.98,
                            save_dir: str = None, name: str = "Benchmark"):
    fig, axs = plt.subplots(len(benchmark), 1, figsize=figsize, sharex=True)
    results = hypertune_dict(benchmark)
    cmap = plt.get_cmap("tab10")

    for i, train_N in enumerate(results["train_N"]):
        iters = results[train_N]["iterations"]
        mse = results[train_N]["mean_test_score"]
        best = results[train_N]["best_score"]
        param = list(results[train_N][f"param_{param_name}"].data)
        vals = np.unique(param)
        param = np.array(param)
        color_dic = {}
        for j, val in enumerate(vals):
            color_dic[str(val)] = cmap(j)
        color = color_dic[str(results["best_params"][i][param_name])]
        axs[i].plot(iters, best, label="best", color=color, zorder=-1)
        for j, val in enumerate(vals):
            if str(val) == "default" and \
             param_name == "TruncOptions__MaxInteraction":
                label = "M"
            else:
                label = str(val)
            axs[i].scatter(iters[param == val], mse[param == val],
                           label=label, marker='o',
                           color=color_dic[str(val)])
        axs[i].set_ylabel("-MSE")
        axs[i].set_title(f"Training size: {train_N}")
        axs[i].legend(title=f"{param_name}:", bbox_to_anchor=(1.01, 1),
                      loc='upper left')
    axs[i].set_xlabel("Iterations")
    plt.xticks(rotation=45)
    fig.suptitle(
        f"{name}: {benchmark.name}: {param_name} hyperparameter tuning",
        fontsize=14, y=title_pos)
    plt.tight_layout(pad=2.0)

    if save_dir is not None:
        plt.savefig(f"{save_dir}/{name}_{benchmark.name}"
                    f"_{param_name}_hypertune.png",
                    bbox_inches='tight')
        plt.show()
    else:
        plt.show()
