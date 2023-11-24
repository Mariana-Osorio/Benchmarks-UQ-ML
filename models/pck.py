# See https://storage.googleapis.com/uqpylab-doc-html/
# UserManual_PolynomialChaos.html for full Sparse PCE options
# See https://storage.googleapis.com/uqpylab-doc-html/UserManual_Kriging.html
# for full Kriging options

import numpy as np
from copy import deepcopy
from uqpylab import sessions
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, \
    check_is_fitted
from models.utils import AttrDict, uq_input, get_min_max, get_params_uq_model

# For PCK we will use the default parameters set to the PCE. Method is not
# included in the default parameters because of a bug that only allows LARS.
default_params_pce = {
    'InputOpts': None,
    'Input': None,
    'uq': None,
    'InputType': 'uniform',
    'Degree': 'default',
    'DegreeEarlyStop': 'default',
    'TruncOptions': 'default',
    'qNormEarlyStop': 'default',
}

default_params_kriging = {
    "Corr": AttrDict({
        'Family': 'Matern-5_2',
        'Type': 'Ellipsoidal',
        'Isotropic': True,
        'Nugget': 10**(-10),
    }),
    "EstimMethod": 'CV'
}

default_params_pck = {
    "Mode": "Sequential"
}


def get_MetaOpts_pce(X,
                     InputOpts=default_params_pce['InputOpts'],
                     Input=default_params_pce['Input'],
                     uq=default_params_pce['uq'],
                     InputType=default_params_pce['InputType'],
                     Degree=default_params_pce['Degree'],
                     DegreeEarlyStop=default_params_pce['DegreeEarlyStop'],
                     TruncOptions=default_params_pce['TruncOptions'],
                     qNormEarlyStop=default_params_pce['qNormEarlyStop']):

    if InputOpts is None and Input is None:
        # Non-parametric inference of the input marginals
        # by kernel density estimation
        if uq is None:
            mySession = sessions.cloud()
            uq = mySession.cli
            mySession.reset()

        if InputType == 'ks':
            InputOpts = {
                "Marginals": [{'Type': 'ks', 'Parameters': X[:, i].tolist()}
                              for i in range(X.shape[1])]
            }
        elif InputType == 'uniform':
            InputOpts = {
                "Marginals": [
                    {'Type': 'Uniform', 'Parameters':
                     [*get_min_max(X[:, i], extra=0.1)]}
                    for i in range(X.shape[1])]
            }

        Input = uq.createInput(InputOpts)

    elif InputOpts is not None:
        # Create input marginal object from opts
        if uq is None:
            mySession = sessions.cloud()
            uq = mySession.cli
            mySession.reset()
        Input = uq_input(InputOpts, uq=uq)

    MetaOpts = {'Input': Input}
    if Degree != 'default':
        MetaOpts['Degree'] = Degree
    if DegreeEarlyStop != 'default':
        MetaOpts['DegreeEarlyStop'] = DegreeEarlyStop
    if TruncOptions != 'default':
        MetaOpts['TruncOptions'] = deepcopy(TruncOptions)
    if qNormEarlyStop != 'default':
        MetaOpts['qNormEarlyStop'] = qNormEarlyStop

    return MetaOpts


def get_MetaOpts_kriging(Corr=default_params_kriging['Corr'],
                         EstimMethod=default_params_kriging['EstimMethod']):
    MetaOpts = {}
    MetaOpts['Corr'] = Corr
    MetaOpts['EstimMethod'] = EstimMethod

    return MetaOpts


def get_MetaOpts_pck(X, y, Mode=default_params_pck['Mode']):
    MetaOpts = {
        'Type': 'Metamodel',
        'MetaType': 'PCK',
        'Mode': Mode
    }
    MetaOpts["ExpDesign"] = {
        'X': X.tolist(),
        'Y': y.tolist()
    }
    return MetaOpts


class PCK_Regressor_UQLab(BaseEstimator, RegressorMixin):
    '''
    Scikit-learn wrapper for UQLab PCE-Kriging (PCK) regression.

    Parameters
    ----------
    PCK Parameters:
        Mode: str, default 'Sequential', PCK combination mode.
    PCE Parameters:
        Degree : int or list, Maximum polynomial degree or set of polynomial
                degrees for degree-adaptive polynomial chaos.
        DegreeEarlyStop : bool, default True, Toggle polynomial degree early
                          stop criterion on/off.
        PolyTypes : list of str, default 'auto', List of polynomial families
                    to be used to build the PCE basis. The default is
                    dependent on input distributions.
        PolyTypesParams : list of float, default None, Set of parameters to be
                        used to build the PCE basis. It is only used when
                        PolyTypes contains Jacobi or Laguerre polynomials.
        TruncOptions : dict, default {'qNorm': 1.0, 'MaxInteraction': 2}, Basis
                    truncation.
        qNormEarlyStop : bool, default True, Toggle hyperbolic truncation norm
                        early stop criterion on/off.
        Method : str, default 'LARS', Coefﬁcients calculation method.
        Quadrature : dict, default {'Level': 'auto', 'Type': 'default'},
                     Quadrature options.
        OLS : dict, default {'TargetAccuracy': 0.0}, OLS-specific options.
        LARS : dict, default {'LarsEarlyStop': 'N_dependent',
            'TargetAccuracy': 0.0, 'KeepIterations': False, 'HybridLoo': True,
            'ModifiedLoo': 1}, LARS-specific options.
        OMP : dict, default {'OmpEarlyStop': 'N_dependent',
            'TargetAccuracy': 0.0, 'KeepIterations': False,
            'ModifiedLoo': 1}, OMP-specific options.
        SP : dict, default {'NNZ': [], 'CVMethod': 'loo', 'NumFolds': 5,
            'ModifiedLoo': 1}, SP-specific options.
        BCS : dict, default {'NumFolds': 10, 'ModifiedLoo': 0}, BCS-specific
            options.
        Bootstrap : str, default 'default', Bootstrapping options.
    Kriging Parameters:
        Trend: dict, Options to specify the Kriging trend.
        Corr: dict, Options to specify the correlation function.
        EstimMethod: str, Select the method to estimate Kriging parameters
                    options are: 'CV', 'ML'.
        CV: dict, Options relevant to the cross-validation estimation method.
        Optim: dict, Options related to the optimization in the estimation
            of Kriging parameters.
    '''
    def __init__(self, uq=None, random_state: int = 2652124,
                 verbose: int = 1,
                 Mode=default_params_pck['Mode'],
                 InputOpts=None, Input=None, InputType='uniform',
                 Degree=default_params_pce['Degree'],
                 DegreeEarlyStop=default_params_pce['DegreeEarlyStop'],
                 TruncOptions=default_params_pce['TruncOptions'],
                 qNormEarlyStop=default_params_pce['qNormEarlyStop'],
                 Corr=default_params_kriging['Corr'],
                 EstimMethod=default_params_kriging['EstimMethod']):
        self.uq = uq
        self.random_state = random_state
        self.verbose = verbose
        self.Mode = Mode
        self.InputOpts = InputOpts
        self.Input = Input
        self.InputType = InputType
        self.Degree = Degree
        self.DegreeEarlyStop = DegreeEarlyStop
        self.TruncOptions = TruncOptions
        self.qNormEarlyStop = qNormEarlyStop
        self.Corr = Corr
        self.EstimMethod = EstimMethod
        self.__model = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if self.uq is None:
            mySession = sessions.cloud()
            self.uq = mySession.cli
            mySession.reset()
            mySession.timeout = 900

        self.uq.rng(self.random_state, 'twister')

        MetaOpts = get_MetaOpts_pck(self.X_, self.y_, self.Mode)
        MetaOpts["PCE"] = get_MetaOpts_pce(
            self.X_, InputOpts=self.InputOpts, Input=self.Input,
            InputType=self.InputType, uq=self.uq, Degree=self.Degree,
            DegreeEarlyStop=self.DegreeEarlyStop,
            TruncOptions=self.TruncOptions, qNormEarlyStop=self.qNormEarlyStop)
        MetaOpts["Kriging"] = get_MetaOpts_kriging(
            self.Corr, self.EstimMethod)

        self.__model = self.uq.createModel(MetaOpts)

        if self.verbose > 0:
            print("PCK model fitted successfully.")
        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = X.astype(np.float64) if not isinstance(X, np.floating) else X
        y_pred = self.uq.evalModel(self.__model, X)
        return y_pred.reshape(-1, 1)

    def params(self):
        check_is_fitted(self, ['X_', 'y_'])
        params = get_params_uq_model(self.__model)
        return params

    def model_params(self):
        check_is_fitted(self, ['X_', 'y_'])
        model_params = deepcopy(self.__model['PCK'])
        return model_params
