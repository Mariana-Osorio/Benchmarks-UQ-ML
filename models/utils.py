# Functionality used across all modules

import dill
from uqpylab import sessions
from copy import deepcopy
import numpy as np
import skorch
import typing as ty
from torch import Tensor, sigmoid
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
from scipy.stats import uniform


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def get_params(self, deep=True):
        return self.__dict__

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def model_params(config, return_non_model_params=False):
    config = deepcopy(config)
    model_params = {}
    non_model_params = {}
    for key in config.keys():
        if key.startswith("model__"):
            model_params[key[len("model__"):]] = config[key]
        else:
            non_model_params[key] = config[key]
    if return_non_model_params:
        return model_params, non_model_params
    else:
        return model_params


def iter_or_distr(obj):
    if hasattr(obj, 'dist') or hasattr(obj, '__iter__') \
     and not isinstance(obj, str):
        return True
    else:
        return False


def get_min_max(arr, extra=0.1):
    min = np.min(arr)
    max = np.max(arr)
    ran = max - min
    min = min - extra * ran
    max = max + extra * ran
    return min, max


def get_params_uq_model(model):
    params = {}
    for key in model['Options']:
        if key in ['ExpDesign', 'Type', 'MetaType']:
            pass
        elif key == 'Input':
            params[key] = \
                deepcopy(model['Options'][key]['Marginals'])
        else:
            if key == "PCE":
                params[key] = deepcopy(model['Options'][key]['Options'])
                params[key]['Input'] = \
                    deepcopy(model['Options'][key]['Input']['Marginals'])
            else:
                params[key] = deepcopy(model['Options'][key])
    return params


def uq_model(ModelOpts: dict, uq=None):
    '''
    Creates a UQpyLab model object.
    '''
    # UQPylab cloud session
    if uq is None:
        mySession = sessions.cloud()
        uq = mySession.cli
        mySession.reset()

    myModel = uq.createModel(ModelOpts)

    return myModel


def uq_input(InputOpts: dict, uq=None):
    '''
    Creates a UQpyLab input object.
    '''
    # UQPylab cloud session
    if uq is None:
        mySession = sessions.cloud()
        uq = mySession.cli
        mySession.reset()

    myInput = uq.createInput(InputOpts)

    return myInput


def uq_input_sample(myInput, N=100, sampling_method="LHS", seed=100, uq=None):
    '''
    Generates a sample of a UQpyLab input object.
    '''
    # UQPylab cloud session
    if uq is None:
        mySession = sessions.cloud()
        uq = mySession.cli
        mySession.reset()

    # Set seed
    uq.rng(seed, 'twister')

    # Generate samples
    X = uq.getSample(myInput, N, sampling_method)

    return X


def uq_model_eval(myModel, X, uq=None):
    '''
    Evaluates a UQpyLab model object.
    '''
    # UQPylab cloud session
    if uq is None:
        mySession = sessions.cloud()
        uq = mySession.cli
        mySession.reset()

    # Evaluate model
    y = uq.evalModel(myModel, X)

    return y


def uq_model_sample(ModelOpts: dict, InputOpts: dict, N=100,
                    sampling_method="LHS", seed=100, uq=None):
    '''
    Generates a sample of a UQpyLab model. It creates the
    Model object with ModelOpts, the Input object with InputOpts,
    and generates N samples with the sampling_method option.

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
        (X, y): samples and evaluations of the model.
    '''
    # UQPylab cloud session
    if uq is None:
        mySession = sessions.cloud()
        uq = mySession.cli
        mySession.reset()

    # Create model and input objects
    myModel = uq_model(ModelOpts, uq=uq)
    myInput = uq_input(InputOpts, uq=uq)

    # Generate samples and evaluate model
    X = uq_input_sample(myInput, N, sampling_method, seed, uq=uq)
    y = uq_model_eval(myModel, X, uq=uq)

    return X, y


def load_object(file):
    '''
    Loads an object from a pickle file.

    Parameters:
        file (str): Path to pickle file.

    Returns:
        object: Object loaded from the pickle file.
    '''
    with open(file, 'rb') as f:
        return dill.load(f)


class NeuralNetRegressorBis(skorch.NeuralNetRegressor):
    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y)


optimizer_dict = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD
}


def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )


def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )


class InputShapeSetterResnet(skorch.callbacks.Callback):
    def __init__(self, regression=False, batch_size=None,
                 categorical_indicator=None, categories=None):
        self.categorical_indicator = categorical_indicator
        self.regression = regression
        self.batch_size = batch_size
        self.categories = categories

    def on_train_begin(self, net, X, y):
        if self.categorical_indicator is None:
            d_numerical = X.shape[1]
            categories = None
        else:
            d_numerical = X.shape[1] - sum(self.categorical_indicator)
            if self.categories is None:
                categories = list((
                    X[:, self.categorical_indicator].max(0) + 1).astype(int))
            else:
                categories = self.categories
        net.set_params(module__d_numerical=d_numerical,
                       module__categories=categories,
                       module__d_out=2 if not self.regression else 1
                       )


class q_uniform():
    def __init__(self, low: int, high: int, q=1, name="q_uniform"):
        """A quantized uniform discrete random variable."""
        self.low = low
        self.high = high
        self.q = q
        self.name = name
        self.rv = uniform(self.low, self.high)

    def rvs(self, size=None, random_state=2652124):
        uniforms = self.rv.rvs(size=size, random_state=random_state)
        return (np.round(uniforms/self.q)*self.q).astype(int)


def euclidean_distance(x, y):
    return np.abs(y - x)


def add_dicts(dict1, dict2):
    assert not (dict1 == {} and dict2 == {}), "Both dicts are empty!"
    if dict1 == {}:
        return dict2
    elif dict2 == {}:
        return dict1
    else:
        assert dict1.keys() == dict2.keys(), "Keys are not the same!"
        summed_dict = {}
        for key in dict1.keys():
            if isinstance(dict1[key], np.ndarray):
                summed_dict[key] = np.hstack((dict1[key], dict2[key]))
            else:
                summed_dict[key] = dict1[key] + dict2[key]
        return summed_dict
