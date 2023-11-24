# See https://storage.googleapis.com/uqpylab-doc-html/
# UserManual_PolynomialChaos.html for full Sparse PCE options

import numpy as np
from copy import deepcopy
from uqpylab import sessions
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, \
    check_is_fitted
from models.utils import AttrDict, uq_input, get_min_max, get_params_uq_model

default_params = {
    'InputOpts': None,
    'Input': None,
    'uq': None,
    'InputType': 'uniform',
    'Degree': list(range(2, 16)),
    'DegreeEarlyStop': True,
    'TruncOptions': AttrDict({
        'qNorm': np.arange(0.5, 1.01, 0.1).tolist(),
        'MaxInteraction': 2,
    }),
    'qNormEarlyStop': True,
    'Method': 'LARS',
    'OLS': AttrDict({
        'TargetAccuracy': 0.0
    }),
    'LARS': AttrDict({
        'LarsEarlyStop': 'default',
        'TargetAccuracy': 0.0,
        'KeepIterations': False,
        'HybridLoo': True,
        'ModifiedLoo': 1
    }),
    'OMP': AttrDict({
        'OmpEarlyStop': 'default',
        'TargetAccuracy': 0.0,
        'KeepIterations': False,
        'ModifiedLoo': 1
    }),
    'SP': AttrDict({
        'NNZ': [],
        'CVMethod': 'loo',
        'NumFolds': 5,
        'ModifiedLoo': 1
    }),
    'BCS': AttrDict({
        'NumFolds': 10,
        'ModifiedLoo': 0
    }),
    'Bootstrap': 'default'
}


def get_MetaOpts(X, y,
                 InputOpts=default_params['InputOpts'],
                 Input=default_params['Input'],
                 uq=default_params['uq'],
                 InputType=default_params['InputType'],
                 Degree=default_params['Degree'],
                 DegreeEarlyStop=default_params['DegreeEarlyStop'],
                 TruncOptions=default_params['TruncOptions'],
                 qNormEarlyStop=default_params['qNormEarlyStop'],
                 Method=default_params['Method'],
                 OLS=default_params['OLS'],
                 LARS=default_params['LARS'],
                 OMP=default_params['OMP'],
                 SP=default_params['SP'],
                 BCS=default_params['BCS'],
                 Bootstrap=default_params['Bootstrap']):

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

    MetaOpts = {
        'Type': 'Metamodel',
        'MetaType': 'PCE',
        'Input': Input
    }
    MetaOpts["ExpDesign"] = {
        'X': X.tolist(),
        'Y': y.tolist()
    }
    MetaOpts['Degree'] = Degree
    MetaOpts['DegreeEarlyStop'] = DegreeEarlyStop
    MetaOpts['TruncOptions'] = deepcopy(TruncOptions)
    if MetaOpts['TruncOptions']['MaxInteraction'] == 'default':
        del MetaOpts['TruncOptions']['MaxInteraction']
    MetaOpts['qNormEarlyStop'] = qNormEarlyStop
    MetaOpts['Method'] = Method
    if Method == 'OLS':
        MetaOpts['OLS'] = deepcopy(OLS)
    elif Method == 'LARS':
        MetaOpts['LARS'] = deepcopy(LARS)
        if MetaOpts['LARS']['LarsEarlyStop'] == 'default':
            del MetaOpts['LARS']['LarsEarlyStop']
    elif Method == 'OMP':
        MetaOpts['OMP'] = deepcopy(OMP)
        if MetaOpts['OMP']['OmpEarlyStop'] == 'default':
            del MetaOpts['OMP']['OmpEarlyStop']
    elif Method == 'SP':
        MetaOpts['SP'] = deepcopy(SP)
    elif Method == 'BCS':
        MetaOpts['BCS'] = deepcopy(BCS)
    if Bootstrap != 'default':
        MetaOpts['Bootstrap'] = Bootstrap

    return MetaOpts


class PCE_Regressor_UQLab(BaseEstimator, RegressorMixin):
    '''
    Scikit-learn wrapper for UQLab PCE regression.

    Parameters
    ----------
    Degree : int or list, Maximum polynomial degree or set of polynomial
             degrees for degree-adaptive polynomial chaos.
    DegreeEarlyStop : bool, default True, Toggle polynomial degree early stop
                      criterion on/off.
    PolyTypes : list of str, default 'auto', List of polynomial families to be
                used to build the PCE basis. The default is dependent on input
                distributions.
    PolyTypesParams : list of float, default None, Set of parameters to be
                      used to build the PCE basis. It is only used when
                      PolyTypes contains Jacobi or Laguerre polynomials.
    TruncOptions : dict, default {'qNorm': 1.0, 'MaxInteraction': 2}, Basis
                   truncation.
    qNormEarlyStop : bool, default True, Toggle hyperbolic truncation norm
                     early stop criterion on/off.
    Method : str, default 'LARS', Coefﬁcients calculation method.
    Quadrature : dict, default {'Level': 'auto', 'Type': 'default'}, Quadrature
                 options.
    OLS : dict, default {'TargetAccuracy': 0.0}, OLS-specific options.
    LARS : dict, default {'LarsEarlyStop': 'N_dependent',
           'TargetAccuracy': 0.0, 'KeepIterations': False, 'HybridLoo': True,
           'ModifiedLoo': 1}, LARS-specific options.
    OMP : dict, default {'OmpEarlyStop': 'N_dependent', 'TargetAccuracy': 0.0,
          'KeepIterations': False, 'ModifiedLoo': 1}, OMP-specific options.
    SP : dict, default {'NNZ': [], 'CVMethod': 'loo', 'NumFolds': 5,
         'ModifiedLoo': 1}, SP-specific options.
    BCS : dict, default {'NumFolds': 10, 'ModifiedLoo': 0}, BCS-specific
          options.
    Bootstrap : str, default 'default', Bootstrapping options.

    '''
    def __init__(self, uq=None, random_state: int = 2652124, verbose: int = 0,
                 InputOpts=None, Input=None, InputType='uniform',
                 Degree=default_params['Degree'],
                 DegreeEarlyStop=default_params['DegreeEarlyStop'],
                 TruncOptions=default_params['TruncOptions'],
                 qNormEarlyStop=default_params['qNormEarlyStop'],
                 Method=default_params['Method'],
                 OLS=default_params['OLS'], LARS=default_params['LARS'],
                 OMP=default_params['OMP'], SP=default_params['SP'],
                 BCS=default_params['BCS'],
                 Bootstrap=default_params['Bootstrap']):
        self.uq = uq
        self.random_state = random_state
        self.verbose = verbose
        self.InputOpts = InputOpts
        self.Input = Input
        self.InputType = InputType
        self.Degree = Degree
        self.DegreeEarlyStop = DegreeEarlyStop
        self.TruncOptions = TruncOptions
        self.qNormEarlyStop = qNormEarlyStop
        self.Method = Method
        self.OLS = OLS
        self.LARS = LARS
        self.OMP = OMP
        self.SP = SP
        self.BCS = BCS
        self.Bootstrap = Bootstrap
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

        MetaOpts = get_MetaOpts(self.X_, self.y_, InputOpts=self.InputOpts,
                                Input=self.Input, InputType=self.InputType,
                                uq=self.uq, Degree=self.Degree,
                                DegreeEarlyStop=self.DegreeEarlyStop,
                                TruncOptions=self.TruncOptions,
                                qNormEarlyStop=self.qNormEarlyStop,
                                Method=self.Method,
                                OLS=self.OLS, LARS=self.LARS,
                                OMP=self.OMP, SP=self.SP,
                                BCS=self.BCS,
                                Bootstrap=self.Bootstrap)
        self.MetaOpts = MetaOpts

        self.__model = self.uq.createModel(MetaOpts)
        self.is_fitted_ = True
        if self.verbose > 0:
            print("PCE model fitted successfully.")

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        X = X.astype(np.float64) if not isinstance(X, np.floating) else X
        y_pred = self.uq.evalModel(self.__model, X)
        return y_pred.reshape(-1, 1)

    def moments(self):
        check_is_fitted(self, ['X_', 'y_'])
        return self.__PCE['Moments']

    def params(self):
        check_is_fitted(self, ['X_', 'y_'])
        params = get_params_uq_model(self.__model)

        return params

    def model_params(self):
        check_is_fitted(self, ['X_', 'y_'])
        model_params = {}
        model_params['Basis'] = deepcopy(self.__model['PCE']['Basis'])
        model_params['Coefficients'] = \
            deepcopy(self.__model['PCE']['Coefficients'])
        model_params['Moments'] = deepcopy(self.__model['PCE']['Moments'])

        return model_params
