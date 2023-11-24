import os
import io
import math
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataframe_image as dfi
from contextlib import redirect_stdout
from itertools import count
from uqpylab import sessions
from scipy.io.arff import loadarff
from mdutils.mdutils import MdUtils
from sklearn.model_selection import RepeatedKFold
from models.utils import uq_input, uq_model, uq_input_sample, \
    uq_model_eval, uq_model_sample, AttrDict
from fractions import Fraction


class Dataset:
    _ids = count(0)

    def __init__(self, data, feature_names: list = None,
                 target_names: list = None, feature_ixs: list = None,
                 target_ixs: list = None) -> None:
        '''
        Class for Dataset.

        Parameters:
            data (array): Data for the dataset. The target variable(s)
            must be the last column(s). Only suuports numeric features.
            feature_names (list): List of feature_names. If None,
            feature_names are named x0, x1, x2, etc.
            target_names (list or str): List of target_names variables or
            string of target variable. If None, target_names is named y0,
            y1, y2, etc.
            feature_ixs (list): List of indices of feature_names. If None, all
            columns except the last n, where n is the number of targets
            are considered feature_names.
            target_ixs (list): List of indices of target variables. If None,
            the last n columns, where n is the number of targets are
            considered target variables.
        '''
        self.df = None
        if isinstance(data, pd.DataFrame):
            self.df = data
            self.data = data.to_numpy()
        else:
            self.data = np.array(data)
        self.__row_num = self.data.shape[0]
        self.__col_num = self.data.shape[1]

        if target_names is not None:
            if isinstance(target_names, str):
                self.target_names = [target_names]
            else:
                self.target_names = target_names
        else:
            if target_ixs is not None:
                self.target_names = [f'y{i}' for i in range(len(target_ixs))]
            else:
                self.target_names = ["y"]

        assert self.__col_num > len(self.target_names), 'Number of ' \
            'target variables must be less than number of columns in data.'

        if feature_names is not None:
            assert len(feature_names) + len(self.target_names) == \
                self.__col_num, 'Number of feature_names and target variables'\
                                ' must match number of columns in data.'
            self.feature_names = feature_names
        else:
            self.feature_names = [f'x{i}' for i in
                                  range(self.__col_num -
                                        len(self.target_names))]

        if target_ixs is not None and feature_ixs is None:
            assert set(target_ixs).issubset(set(range(self.__col_num))), \
                'Target indices must be in range of columns in data.'
            assert len(set(target_ixs)) == len(target_ixs), \
                'Target indices must be unique.'
            assert len(target_ixs) == len(self.target_names), \
                'Length of target_ixs must match length of target.'
            self.data = self.data[:, list(set(range(self.__col_num)) -
                                          set(target_ixs)) + target_ixs]
        elif target_ixs is None and feature_ixs is not None:
            assert set(feature_ixs).issubset(set(range(self.__col_num))), \
                'Feature indices must be in range of columns in data.'
            assert len(set(feature_ixs)) == len(feature_ixs), \
                'Feature indices must be unique.'
            assert len(feature_ixs) == len(self.feature_names), \
                'Length of feature_ixs must match length of feature_names.'
            self.data = self.data[:, feature_ixs +
                                  list(set(range(self.__col_num)) -
                                       set(feature_ixs))]
        elif target_ixs is not None and feature_ixs is not None:
            assert set(target_ixs + feature_ixs) == \
                set(range(self.__col_num)), 'Feature and target indices' \
                                            ' must cover all the columns.'
            assert len(set(target_ixs + feature_ixs)) == \
                len(target_ixs + feature_ixs), 'Feature and target indices' \
                                               ' must be unique.'
            self.data = self.data[:, feature_ixs + target_ixs]

        self.__target_ixs = list(range(self.__col_num - len(self.target_names),
                                       self.__col_num))
        self.__feature_ixs = list(range(self.__col_num -
                                        len(self.target_names)))
        self.columns = self.feature_names + self.target_names

        if self.df is None:
            self.df = pd.DataFrame(self.data, columns=self.columns)

        self.X = self.data[:, self.__feature_ixs]
        self.y = self.data[:, self.__target_ixs]

        self.id = next(self._ids)
        self.report_dict = None

    def __repr__(self):
        return f'Dataset(instances={len(self)},' \
                f'feature_names={self.feature_names}, '\
                f'target_names={self.target_names})'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, str):
            if key in self.target_names:
                return self.data[:, self.__target_ixs[
                    self.target_names.index(key)]]
            elif key in self.feature_names:
                return self.data[:, self.feature_names.index(key)]
            else:
                raise KeyError(f'Key {key} not found.')
        elif isinstance(key, list):
            if all(isinstance(item, int) for item in key):
                return self.data[:, key]
            elif all(isinstance(item, str) for item in key):
                return self.data[:, [self.columns.index(item) for item in key]]
            else:
                raise KeyError(f'Key {key} not found.')

    def __iter__(self):
        return iter(self.data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, feature_names: list = None,
                       target_names: list = None):
        '''
        Generates a CaseStudy object from a pandas DataFrame.

        Parameters:
            df (DataFrame): Pandas DataFrame.
            feature_names (list): List of feature_names. If None, all columns
            except last one are taken as feature_names.
            target_names (list or str): List of target variables or string of
            target variable. If None, last column is the target.
        '''
        if feature_names is None and target_names is None:
            target_names = list(df.columns)[-1]
            feature_names = list(df.columns)[:-1]
            return cls(df, feature_names=feature_names,
                       target_names=target_names)
        elif feature_names is None and target_names is not None:
            if isinstance(target_names, str):
                target_names = [target_names]
            assert isinstance(target_names, list), \
                'target_names must be a list.'
            if list(df.columns[-len(target_names):]) != target_names:
                assert set(target_names).issubset(df.columns), \
                    'Target variable(s) not found in DataFrame.'
                df = df[[col for col in df.columns if col not in target_names]
                        + target_names]
            feature_names = list(df.columns)[:-len(target_names)]
            return cls(df, feature_names=feature_names,
                       target_names=target_names)
        elif feature_names is not None and target_names is None:
            assert isinstance(feature_names, list), \
                'feature_names must be a list.'
            assert set(feature_names).issubset(df.columns), \
                'Feature(s) not found in DataFrame.'
            target_names = [col for col in df.columns
                            if col not in feature_names]
            if list(df.columns[-len(target_names):]) != target_names:
                df = df[[col for col in df.columns if col not in target_names]
                        + target_names]
            return cls(df, feature_names=feature_names,
                       target_names=target_names)
        else:
            assert isinstance(feature_names, list), \
                'feature_names must be a list.'
            if isinstance(target_names, str):
                target_names = [target_names]
            assert isinstance(target_names, list), \
                'target_names must be a list.'
            assert set(feature_names + target_names).issubset(df.columns), \
                'Feature(s) and/or target variable(s) not found in DataFrame.'
            df = df[feature_names + target_names]
            return cls(df, feature_names=feature_names,
                       target_names=target_names)

    @classmethod
    def from_data(cls, settings: dict, feature_names: list = None,
                  target_names: list = None):
        '''
        Generates a CaseStudy object from a data file. It reads the data with
        pandas, creates a CaseStudy object and returns it.

        Parameters:
            settings (dict): Dictionary with the following keys:
                file (str): Path to file.
                typ (str): Type of file. Options: "fwf", "csv", "xlsx", "arff".
                **kwargs: Keyword arguments for pandas read function.

        Returns:
            CaseStudy: CaseStudy object with the data from the file.
        '''
        file = settings["file"]
        assert os.path.isfile(file), f'File {file} not valid.'
        typ = settings["typ"] if "typ" in settings.keys() else "csv"
        exclude_cols = None if "exclude_cols" not in settings.keys() \
            else settings["exclude_cols"]
        read_settings = {key: settings[key] for key in settings.keys()
                         if key not in ["file", "typ", "exclude_cols"]}
        if typ == "fwf":
            df = pd.read_fwf(file, **read_settings)
        elif typ == "csv":
            df = pd.read_csv(file, **read_settings)
        elif typ == "xlsx":
            df = pd.read_excel(file, **read_settings)
        elif typ == "arff":
            data = loadarff(file)
            df = pd.DataFrame(data[0])
            if "names" in read_settings:
                df.columns = read_settings["names"]

        if exclude_cols is not None:
            df = df[[col for col in df.columns if col not in exclude_cols]]

        return cls.from_dataframe(df, feature_names=feature_names,
                                  target_names=target_names)

    @classmethod
    def from_model(cls, ModelOpts: dict, InputOpts: dict, N=100,
                   sampling_method="LHS", seed=100,
                   uq=None):
        '''
        Generates a CaseStudy object from a UQpyLab model. It creates the
        Model object with ModelOpts, the Input object with InputOpts,
        generates N samples with the sampling_method option and returns
        the CaseStudy object.

        Parameters:
            ModelOpts (dict): Dictionary with the following keys:
                ModelFun (str): A module-method deﬁnition of the model
                function. For example, the value A.B.C refers to a model
                function deﬁned by the method C which is retrieved by
                from A.B import C.
                mString (str): String containing the model expression.
                Parameters (list or dict): List or dictionary of non-random
                model parameters.
                isVectorized (bool): Boolean indicating whether the model
                function is vectorized.
            InputOpts (dict): Dictionary with the following keys:
                Marginals (list): List of dictionaries with the following keys:
                    Type (str): Type of marginal distribution, see UQpyLab
                    documentation for options.
                    Moments (variable length float): List of moments of the
                    distribution (mean and std).
                    Parameters (variable length float): List of parameters of
                    the marginal distribution.
                    Bounds (list): List of bounds of the marginal distribution.
                Copula (list): List of dictionaries with the options regarding
                the copula (or copulas) of the random vector, see UQpyLab
                documentation for options.
                Name (str): Name of the input object.
            N (int): Number of samples.
            sampling_method (str): Sampling method option of uq.getSample,
            options are: 'MC', 'LHS', 'Sobol', 'Halton'.
            seed (int): Seed for the random number generator.
            uq (UQpyLab session cli): UQpyLab session cli object. If None, a
            new session is created.

        Returns:
            CaseStudy: CaseStudy object with the UQpyLab evaluation data.
        '''
        X, y = uq_model_sample(ModelOpts, InputOpts, N, sampling_method,
                               seed, uq)
        data = np.hstack((X, y))

        return cls(data,
                   feature_names=[marg['Name'] for marg in
                                  InputOpts['Marginals']],
                   target_names=['y'])

    @classmethod
    def from_obj_file(cls, dataset_file):
        '''
        Generates a CaseStudy object from a pickle file.

        Parameters:
            dataset_file (str): Path to pickle file.

        Returns:
            CaseStudy: CaseStudy object with the data from the pickle file.
        '''
        with open(dataset_file, 'rb') as f:
            return dill.load(f)

    def save(self, report_dir="", dataset_name="dataset"):
        '''
        Saves the CaseStudy object in a pickle file in the report_dir
        directory.
        '''
        if report_dir == "" and dataset_name == "dataset" and \
                self.report_dict is not None:
            report_dir = self.report_dict["report_dir"]
            dataset_name = self.report_dict["dataset_name"]

        if report_dir != "" and not os.path.isdir(report_dir):
            os.mkdir(report_dir)
        dataset_file = os.path.join(report_dir, f"{dataset_name}.obj")
        with open(dataset_file, 'wb') as f:
            dill.dump(self, f, protocol=dill.HIGHEST_PROTOCOL)

    def describe(self):
        '''
        Uses function `pd.DataFrame.describe()` to print a description of
        the dataset.
        '''
        return self.df.describe()

    def info(self):
        '''
        Uses function `pd.DataFrame.info()` to print information about
        the dataset.
        '''
        f = io.StringIO()
        with redirect_stdout(f):
            self.df.info()
        return f.getvalue()

    def histograms(self, num_cols=3, fig_size=10, bins="auto"):
        '''
        Uses function `sns.histplot()` to plot histograms of the columns
        in the dataset.
        '''
        num_rows = math.ceil(len(self.df.columns)/num_cols)
        fig, axes = plt.subplots(num_rows, num_cols,
                                 figsize=(fig_size,
                                          fig_size*(num_rows/num_cols)))
        for i, ax in enumerate(axes.flatten()):
            if i >= len(self.df.columns):
                ax.set_visible(False)
            else:
                col = self.df.columns[i]
                val_counts = self.df[col].value_counts()
                hist = sns.histplot(self.df[col], ax=ax, bins=bins)
                ax.set_title(f"{col}\n{len(val_counts)} value counts"
                             f"({100*len(val_counts)/len(self.df):.2f}%)")
        plt.tight_layout(pad=1.4, w_pad=1.5, h_pad=2.0)
        plt.close(fig)
        return hist.figure

    def ind_pairplot(self, bins="auto", hue=None):
        '''
        Uses function `sns.pairplot()` to plot pairwise relationships of the
        features in the dataset.
        '''
        if hue is None:
            pairplot = sns.pairplot(self.df, vars=self.feature_names,
                                    diag_kws={"bins": bins})
        else:
            pairplot = sns.pairplot(self.df, hue=hue, diag_kind="hist",
                                    diag_kws={"bins": bins})
        plt.close(pairplot.figure)

        return pairplot.figure

    def dep_pairplot(self, num_rows=1, num_cols=None, fig_size=10,
                     target=None, bins="auto"):
        '''
        Uses function `sns.pairplot()` to plot pairwise relationships of the
        features and targets in the dataset.
        '''
        if target is None:
            target = self.target_names[0]
        else:
            assert target in self.target_names, \
                f'Target {target} not found.'
        if num_cols is None:
            num_rows = num_rows if num_rows is not None else 1
            num_cols = math.ceil(self.__col_num/num_rows)
        else:
            num_rows = math.ceil(self.__col_num/num_cols)
        fig, axes = plt.subplots(num_rows, num_cols,
                                 figsize=(fig_size,
                                          fig_size*(num_rows/num_cols)))
        for i, ax in enumerate(axes.flatten()):
            if i >= len(self.feature_names) + 1:
                ax.set_visible(False)
                break
            if i == 0:
                pairplot = sns.histplot(self.df, x=target,  alpha=.7,
                                        ax=ax, bins=bins)
            else:
                pairplot = sns.scatterplot(self.df, x=self.columns[i-1],
                                           y=target, alpha=0.7, ax=ax)
            plt.tight_layout(pad=5.0)
            plt.close(pairplot.figure)
        return fig

        return pairplot.figure

    def corr_matrix(self, type="pearson", fig_size=10, font_size=10,
                    annot=True):
        '''
        Uses function `sns.heatmap()` to plot a correlation matrix of the
        features and targets in the dataset.
        '''
        corr = self.df.corr(method=type)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        if annot:
            heatmap = sns.heatmap(corr, mask=mask, cmap="RdBu", center=0,
                                  annot=True, vmin=-1, vmax=1,
                                  square=True, linewidths=.5,
                                  cbar_kws={"shrink": .5}, fmt='.2f',
                                  annot_kws={"fontsize": font_size})
        else:
            heatmap = sns.heatmap(corr, mask=mask, cmap="RdBu", center=0,
                                  vmin=-1, vmax=1, square=True, linewidths=.5,
                                  cbar_kws={"shrink": .5})
            heatmap.tick_params(axis='both', which='major',
                                labelsize=font_size)
        plt.close(heatmap.figure)

        return heatmap.figure

    def report_images(self, report_dir="", dataset_name="dataset",
                      description_file=None, fig_size=10, font_size=12,
                      num_cols=3, annot=True, bins="auto", categorical=None):
        '''
        Creates images for a report with the description, information,
        histograms, correlation matrix and pairplots of the dataset.
        The report images are saved
        in the report_dir/images directory.

        Parameters:
            report_dir (str): Directory where the report is saved.
            dataset_name (str): Name of the dataset.
        '''
        if report_dir != "" and not os.path.isdir(report_dir):
            os.mkdir(report_dir)

        if not os.path.isdir(os.path.join(report_dir, 'images')):
            os.mkdir(os.path.join(report_dir, 'images'))

        report_dict = {"report_dir": report_dir, "dataset_name": dataset_name}

        # Dataset description
        report_dict['description'] = ""
        if description_file is not None:
            with open(description_file) as f:
                report_dict['description'] = f.read()

        # df image
        report_dict["name_df"] = os.path.join('images',
                                              dataset_name + '_df.png')
        dfi.export(self.df, os.path.join(report_dir, report_dict["name_df"]),
                   max_rows=10, max_cols=10, table_conversion='chrome',
                   chrome_path='/usr/bin/brave-browser')

        # df.info text
        report_dict["df_info"] = self.info()

        # df.describe image
        report_dict["name_df_describe"] = os.path.join('images',
                                                       dataset_name +
                                                       '_df_describe.png')
        dfi.export(self.describe(),
                   os.path.join(report_dir, report_dict["name_df_describe"]),
                   max_rows=10, max_cols=10, table_conversion='chrome',
                   chrome_path='/usr/bin/brave-browser')

        # histograms image
        report_dict["name_hist"] = os.path.join('images',
                                                dataset_name + '_hist.png')
        hist = self.histograms(num_cols=num_cols, fig_size=fig_size, bins=bins)
        hist.savefig(os.path.join(report_dir, report_dict["name_hist"]),
                     bbox_inches='tight')

        # ind_pairplot image
        report_dict["name_ind_pairplot"] = os.path.join('images',
                                                        dataset_name +
                                                        '_ind_pairplot.png')
        ind_pairplot = self.ind_pairplot(bins=bins)
        ind_pairplot.savefig(os.path.join(report_dir,
                                          report_dict["name_ind_pairplot"]),
                             bbox_inches='tight')

        # pairplot considering categorical variables
        if categorical is not None:
            assert isinstance(categorical, list), \
                'categorical must be a list.'
            report_dict["name_pairplot_cat"] = []
            report_dict["cat"] = []
            for cat in categorical:
                name_pairplot = os.path.join('images', dataset_name +
                                             f'_pairplot_{cat}.png')
                report_dict["cat"].append(cat)
                report_dict["name_pairplot_cat"].append(name_pairplot)
                pairplot = self.ind_pairplot(bins=bins, hue=cat)
                pairplot.savefig(os.path.join(report_dir, name_pairplot),
                                 bbox_inches='tight')

        # dep_pairplot image
        report_dict["name_dep_pairplot"] = os.path.join(
            'images', dataset_name + '_dep_pairplot.png')
        dep_pairplot = self.dep_pairplot(num_cols=num_cols, fig_size=fig_size,
                                         bins=bins)
        dep_pairplot.savefig(
            os.path.join(report_dir, report_dict["name_dep_pairplot"]),
            bbox_inches='tight')

        # Pearson corr_matrix image
        report_dict["name_pearson_corr"] = os.path.join(
            'images', dataset_name + '_pearson_corr.png')
        corr_matrix = self.corr_matrix(fig_size=fig_size, font_size=font_size,
                                       annot=annot)
        corr_matrix.savefig(
            os.path.join(report_dir, report_dict["name_pearson_corr"]),
            bbox_inches='tight')

        # Spearman corr_matrix image
        report_dict["name_spearman_corr"] = os.path.join(
            'images', dataset_name + '_spearman_corr.png')
        corr_matrix = self.corr_matrix(type="spearman", fig_size=fig_size,
                                       font_size=font_size, annot=annot)
        corr_matrix.savefig(os.path.join(
            report_dir, report_dict["name_spearman_corr"]),
            bbox_inches='tight')

        self.report_dict = report_dict
        self.save()

    def write_report(self, report_dict):
        # Create markdown file report
        mdFile = MdUtils(file_name=os.path.join(report_dict["report_dir"],
                                                report_dict["dataset_name"]),
                         title=report_dict["dataset_name"] + '_dataset')
        mdFile.new_header(1, f'{report_dict["dataset_name"]} Dataset Overview')
        mdFile.new_line()
        mdFile.new_header(2, "Description")
        mdFile.write(report_dict["description"])
        mdFile.new_line()
        # df image
        mdFile.new_header(2, "Raw data")
        text = f'{report_dict["dataset_name"]} pandas dataframe'
        mdFile.new_line(mdFile.new_inline_image(text=text,
                                                path=report_dict["name_df"]))
        mdFile.new_line()
        mdFile.new_line()
        # df.info text
        mdFile.new_header(2, "Pandas dataframe information")
        mdFile.new_header(3, "Info")
        text = "`df.info()`: Contains column names, " \
            "types and non null count.\n"
        mdFile.new_line(text)
        mdFile.insert_code(report_dict["df_info"])
        mdFile.new_line()
        mdFile.new_line()
        # df.describe image
        mdFile.new_header(3, "Describe")
        text = "`df.describe()`: different aggregate statistics of the " \
            "columns.\n\n"
        mdFile.new_line(text)
        mdFile.new_line()
        text = f'{report_dict["dataset_name"]} df describe'
        mdFile.new_line(
            mdFile.new_inline_image(text=text,
                                    path=report_dict["name_df_describe"]))
        mdFile.new_line()
        mdFile.new_line()
        # histograms image
        mdFile.new_header(2, "Column information: Histograms")
        text = "\n`df[col].value_counts()`: tells us the number of unique"\
            " values for numeric columns, percentage (%) tells how big the"\
            " ratio of unique values compared to total values there are.\n\n"\
            "`sns.histplot()`: gives us the histogram of the data.\n\n"
        mdFile.write(text)
        text = f'{report_dict["dataset_name"]} hists'
        mdFile.new_line(mdFile.new_inline_image(text=text,
                                                path=report_dict["name_hist"]))
        mdFile.new_line()
        mdFile.new_line()
        # Matrix plots
        mdFile.new_header(2, "Matrix plots")
        text = "\n`sns.pairplot()`: each numeric independent variable in the"\
            " df will by shared across the y-axes across a single row and the"\
            " x-axes across a single column. For the diagonal a univariate"\
            " distribution plot is drawn to show the marginal distribution of"\
            " the data in each column.\n\n"
        mdFile.write(text)
        mdFile.new_line()
        # ind_pairplot image
        mdFile.new_header(3, "Independent variables")
        text = "Pairplot of all independent variables against each other to"\
            " find correlations.\n"
        mdFile.new_line(text)
        text = f'{report_dict["dataset_name"]} ind pairplot'
        mdFile.new_line(
            mdFile.new_inline_image(text=text,
                                    path=report_dict["name_ind_pairplot"]))
        mdFile.new_line()
        # pairplot considering categorical variables
        if "name_pairplot_cat" in report_dict.keys():
            mdFile.new_header(3, "Categorical variable analysis")
            text = "Pairplot of all variables against each other with the"\
                " categorical variable as hue to identify what to do with"\
                " each categorical variable.\n"
            mdFile.new_line(text)
            for name_pairplot, cat in zip(report_dict["name_pairplot_cat"],
                                          report_dict["cat"]):
                mdFile.new_header(4, f"{cat} variable")
                text = f'{cat} pairplot'
                mdFile.new_line(mdFile.new_inline_image(text=text,
                                                        path=name_pairplot))
                mdFile.new_line()
        # dep_pairplot image
        mdFile.new_header(3, "Dependent variable")
        text = "Pairplot of all independent variables against the target"\
            " variable for regression."
        mdFile.new_line(text)
        text = f'{report_dict["dataset_name"]} dep pairplot'
        mdFile.new_line(
            mdFile.new_inline_image(text=text,
                                    path=report_dict["name_dep_pairplot"]))
        mdFile.new_line()
        mdFile.new_line()
        # Pearson corr_matrix image
        mdFile.new_header(2, "Pearson correlation matrix")
        text = f'{report_dict["dataset_name"]} pearson corr'
        mdFile.new_line(
            mdFile.new_inline_image(text=text,
                                    path=report_dict["name_pearson_corr"]))
        mdFile.new_line()
        mdFile.new_line()
        # Spearman corr_matrix image
        mdFile.new_header(2, "Spearman correlation matrix")
        text = f'{report_dict["dataset_name"]} pearson corr'
        mdFile.new_line(
            mdFile.new_inline_image(text=text,
                                    path=report_dict["name_spearman_corr"]))
        # Create markdown
        mdFile.new_table_of_contents(table_title='Contents', depth=3)
        mdFile.create_md_file()

    def make_report(self, report_dir="", dataset_name="dataset",
                    description_file=None, fig_size=10, font_size=12,
                    num_cols=3, annot=True, bins="auto", categorical=None,
                    make_images=False):
        '''
        Creates a report with the description, information, histograms,
        correlation matrix and pairplots of the dataset. The report is
        saved as an markdown file in the report_dir directory.

        Parameters:
            report_dir (str): Directory where the report is saved.
            dataset_name (str): Name of the dataset.
        '''
        if self.report_dict is None or make_images:
            self.report_images(report_dir=report_dir,
                               dataset_name=dataset_name,
                               description_file=description_file,
                               fig_size=fig_size,
                               font_size=font_size,
                               num_cols=num_cols,
                               annot=annot,
                               bins=bins,
                               categorical=categorical)
        self.write_report(self.report_dict)


class Replications:
    def __init__(self, X, y, num: int, ixs: list, k: int = None,
                 InputOpts=None, Model=None, name=None,
                 X_names=None, y_name=None, info={}) -> None:
        self.X = X
        self.X_names = [f"x_{i}" for i in range(X.shape[1])] \
            if X_names is None else X_names
        self.y = y
        self.y_name = "y" if y_name is None else y_name
        self.num = num
        assert len(ixs) >= num and (len(ixs)/num).is_integer(), \
            'Unvalid ixs.'
        self.ixs = ixs
        self.k = k
        self.InputOpts = InputOpts
        self.Model = Model
        self.cv = True if k is not None else False
        self.train_N = len(self.ixs[0][0])
        self.test_N = len(self.ixs[0][1])
        self.case_study = "case_study" if name is None else name
        name = f'{self.case_study}_replications'
        self.name = f'{self.num}_{self.k}fold_{name}_{self.train_N}' \
            if self.cv else f'{self.num}_{name}_{self.train_N}'
        self.info = info

    @classmethod
    def KFold(cls, X, y, num: int = 3, k: int = 10,
              random_state: int = 2652124, name=None,
              X_names=None, y_name=None):
        '''
        Generates a Replications object with KFold indices.

        Parameters:
            X (array): Features.
            y (array): Targets.
            num (int): Number of replications.
            k (int): Number of folds.
            random_state (int): Seed for the random number generator.
        '''

        assert isinstance(k, (int, float, complex)) and not \
            isinstance(k, bool), \
            'k must be an integer or a fraction with numerator 1.'

        if not isinstance(k, int):
            frac = Fraction(k).limit_denominator(20)
            assert frac.numerator == 1, \
                'k must be an integer or a fraction with numerator 1.'
            k = frac.denominator
            rkf = RepeatedKFold(n_repeats=num, n_splits=k,
                                random_state=random_state)
            ixs = []
            for test_ixs, train_ixs in rkf.split(X):
                ixs.append((train_ixs, test_ixs))
            info = {"type": f"{frac}-fold cross-validation"}
            info["train"] = "1 fold"
            info["test"] = f"{k-1} folds"
        else:
            rkf = RepeatedKFold(n_repeats=num, n_splits=k,
                                random_state=random_state)
            ixs = []
            for train_ixs, test_ixs in rkf.split(X):
                ixs.append((train_ixs, test_ixs))
            info = {"type": f"{k}-fold cross-validation"}
            info["train"] = f"{k-1} folds"
            info["test"] = "1 fold"
        return cls(X, y, num, ixs, k=k, name=name, X_names=X_names,
                   y_name=y_name, info=info)

    @classmethod
    def Model(cls, num, ModelOpts: dict, InputOpts: dict, train_N: int = 100,
              test_N: int = 10000, sampling_method="LHS",
              random_state: int = 2652124, uq=None, name=None,
              X_names=None, y_name=None):
        '''
        Generates a Replications object with UQpyLab model evaluations.
        '''
        if uq is None:
            mySession = sessions.cloud()
            uq = mySession.cli
            mySession.reset()

        # Create model and input objects
        myModel = uq_model(ModelOpts, uq=uq)
        myInput = uq_input(InputOpts, uq=uq)

        # Generate replications
        X = []
        y = []
        for i in range(num):
            seed = random_state + i
            X_train = uq_input_sample(myInput, train_N, sampling_method,
                                      seed, uq)
            X += list(X_train)
            y += list(uq_model_eval(myModel, X_train, uq))
        X_test = uq_input_sample(myInput, test_N, sampling_method,
                                 random_state-1, uq)
        X += list(X_test)
        y += list(uq_model_eval(myModel, X_test, uq))

        # Generate ixs
        ixs = []
        for i in range(num):
            ixs.append((list(range(i*train_N, (i+1)*train_N)),
                        list(range(train_N*num, train_N*num + test_N))))

        info = {"type": "holdout validation",
                "train": f"{train_N} samples",
                "test": f"{test_N} samples"}

        return cls(np.array(X), np.array(y), num, ixs,
                   InputOpts=InputOpts, Model=myModel, name=name,
                   X_names=X_names, y_name=y_name, info=info)

    @classmethod
    def Model_with_test(cls, num, myModel, InputOpts,
                        X_test, y_test, train_N: int = 100,
                        sampling_method="LHS", random_state: int = 2652124,
                        uq=None, name=None, X_names=None, y_name=None):
        '''
        Generates a Replications object with UQpyLab model evaluations.
        The test set is already given.
        '''
        assert len(X_test) == len(y_test), \
            'X_test and y_test must have the same length.'

        test_N = len(X_test)

        if uq is None:
            mySession = sessions.cloud()
            uq = mySession.cli
            mySession.reset()

        myInput = uq_input(InputOpts, uq=uq)

        # Generate replications
        X = []
        y = []
        for i in range(num):
            seed = random_state + i
            X_train = uq_input_sample(myInput, train_N, sampling_method,
                                      seed, uq)
            X += list(X_train)
            y += list(uq_model_eval(myModel, X_train, uq))
        X += list(X_test)
        y += list(y_test)

        # Generate ixs
        ixs = []
        for i in range(num):
            ixs.append((list(range(i*train_N, (i+1)*train_N)),
                        list(range(train_N*num, train_N*num + test_N))))

        info = {"type": "holdout validation",
                "train": f"{train_N} samples",
                "test": f"{test_N} samples"}

        return cls(np.array(X), np.array(y), num, ixs, InputOpts=InputOpts,
                   Model=myModel, name=name, X_names=X_names, y_name=y_name,
                   info=info)

    def __repr__(self):
        repr_str = f'Replications(num={self.num}'
        if "type" in self.info:
            repr_str += f',\n\ttype={self.info["type"]},\n\ttrain=' \
                f'{self.info["train"]},\n\ttest={self.info["test"]})'
        else:
            repr_str += ')'
        return repr_str

    def __len__(self):
        return self.num

    def __getitem__(self, ix):
        return self.ixs[ix]

    def get_rep(self, ix):
        if not self.cv:
            return self.ixs[ix:ix+1]
        else:
            if ix >= self.num:
                raise IndexError(f'Replication ix={ix} must be less than '
                                 f'total replications num={self.num}.')
            return self.ixs[ix*self.k:(ix+1)*self.k]

    def save(self, report_dir="", rep_name="replication"):
        '''
        Saves the Replications object in a pickle file in the report_dir
        directory.
        '''
        if report_dir != "" and not os.path.isdir(report_dir):
            os.mkdir(report_dir)
        dataset_file = os.path.join(report_dir, f"{rep_name}.obj")
        with open(dataset_file, 'wb') as f:
            dill.dump(self, f, protocol=dill.HIGHEST_PROTOCOL)

    def X_replication(self, i: int = 0):
        '''
        Returns the slice of the ith replication of the features.
        '''
        return self.X[self.ixs[i][0]], self.X[self.ixs[i][1]]

    def y_replication(self, i: int = 0):
        '''
        Returns the slice of the ith replication of the targets.
        '''
        return self.y[self.ixs[i][0]], self.y[self.ixs[i][1]]


class ReplicationsList:
    def __init__(self, replications: list) -> None:
        assert all([type(rep).__name__ == "Replications"
                    for rep in replications]), \
                        'replications must be a list of Replications objects.'
        # assert case study is the same among all reps
        assert len(set([rep.case_study for rep in replications])) == 1, \
            'All replications must be from the same case study.'
        self.X = replications[0].X
        self.X_names = replications[0].X_names
        self.y = replications[0].y
        self.y_name = replications[0].y_name
        self.num = [rep.num for rep in replications]

        self.ixs = [rep.ixs for rep in replications]
        self.k = [rep.k for rep in replications]
        self.InputOpts = replications[0].InputOpts
        self.Model = replications[0].Model
        self.cv = replications[0].cv

        self.train_N = [len(rep.ixs[0][0]) for rep in replications]
        self.test_N = [len(rep.ixs[0][1]) for rep in replications]
        self.names = [rep.name for rep in replications]
        self.case_study = replications[0].case_study
        self.info = [rep.info for rep in replications]

    @classmethod
    def from_Model(cls, num, ModelOpts: dict, InputOpts: dict,
                   train_N: list = [25, 50, 100], test_N: int = 1000000,
                   sampling_method="LHS", random_state: int = 2652124,
                   uq=None, name=None):
        '''
        Generates a ReplicationsList object with UQpyLab model evaluations.
        '''
        if uq is None:
            mySession = sessions.cloud()
            uq = mySession.cli
            mySession.reset()

        replications = [Replications.Model(num, ModelOpts, InputOpts,
                                           train_N=train_N[0], test_N=test_N,
                                           sampling_method=sampling_method,
                                           random_state=random_state, uq=uq,
                                           name=name)]
        for i in range(1, len(train_N)):
            replications.append(Replications.Model_with_test(
                num, replications[0].Model, InputOpts,
                replications[0].X_replication()[1],
                replications[0].y_replication()[1],
                train_N=train_N[i], sampling_method=sampling_method,
                random_state=random_state+i, uq=uq, name=name))

        new_X = []
        new_y = []
        new_ixs = []
        last_ix = 0
        for i, rep in enumerate(replications):
            ixs = []
            for j in range(num):
                X = rep.X_replication(j)[0]
                new_X.append(X)
                new_y.append(rep.y_replication(j)[0])
                len_X = len(X)
                ixs.append(np.arange(last_ix, last_ix+len_X))
                last_ix += len_X
            new_ixs.append(ixs)
        new_X.append(rep.X_replication()[1])
        new_y.append(rep.y_replication()[1])
        test_ixs = np.arange(last_ix, last_ix+len(rep.X_replication()[1]))
        for ixs in new_ixs:
            for i in range(len(ixs)):
                ixs[i] = (ixs[i], test_ixs)
        new_X = np.concatenate(new_X)
        new_y = np.concatenate(new_y)
        for i, rep in enumerate(replications):
            rep.X = new_X
            rep.y = new_y
            rep.ixs = new_ixs[i]

        return cls(replications)

    @classmethod
    def from_KFold(cls, X, y, num: int = 3, k: list = [5, 10],
                   random_state: int = 2652124, name=None,
                   X_names=None, y_name=None):
        '''
        Generates a ReplicationsList object with different KFolds.
        '''
        replications = []
        name = "case_study" if name is None else name
        for k_iter in k:
            replications.append(Replications.KFold(X, y, num, k_iter,
                                                   random_state=random_state,
                                                   name=name,
                                                   X_names=X_names,
                                                   y_name=y_name))
        return cls(replications)

    def __len__(self):
        return len(self.ixs)

    def __repr__(self):
        return f'ReplicationsList(num={len(self)})'

    def __getitem__(self, ix):
        return self.rep_dict(ix)

    def __iter__(self):
        rep_iter = [self.rep_dict(ix) for ix in range(len(self))]
        return iter(rep_iter)

    def rep_dict(self, ix: int):
        rep_dict = AttrDict()
        rep_dict["X"] = self.X
        rep_dict["X_names"] = self.X_names
        rep_dict["y"] = self.y
        rep_dict["y_name"] = self.y_name
        rep_dict["num"] = self.num[ix]
        rep_dict["ixs"] = self.ixs[ix]
        rep_dict["k"] = self.k[ix]
        rep_dict["InputOpts"] = self.InputOpts
        rep_dict["Model"] = self.Model
        rep_dict["cv"] = self.cv
        rep_dict["train_N"] = self.train_N[ix]
        rep_dict["test_N"] = self.test_N[ix]
        rep_dict["name"] = self.names[ix]
        rep_dict["info"] = self.info[ix]
        return rep_dict

    def save(self, report_dir="", rep_name="replications_list"):
        '''
        Saves the ReplicationsList object in a pickle file in the report_dir
        directory.
        '''
        if report_dir != "" and not os.path.isdir(report_dir):
            os.mkdir(report_dir)
        dataset_file = os.path.join(report_dir, f"{rep_name}.obj")
        with open(dataset_file, 'wb') as f:
            dill.dump(self, f, protocol=dill.HIGHEST_PROTOCOL)

    def X_replication(self, i: int = 0, j: int = 0):
        indices = self.ixs[i][j]
        return self.X[indices[0]], self.X[indices[1]]

    def y_replication(self, i: int = 0, j: int = 0):
        indices = self.ixs[i][j]
        return self.y[indices[0]], self.y[indices[1]]
