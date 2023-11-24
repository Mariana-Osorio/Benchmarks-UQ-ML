# See https://storage.googleapis.com/uqpylab-doc-html/UserManual_Kriging.html
# for full Kriging options

import numpy as np
from copy import deepcopy
from uqpylab import sessions
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, \
    check_is_fitted
from models.utils import AttrDict, get_params_uq_model


default_params = {
    "Trend": AttrDict({
        'Type': 'ordinary',
        'Degree': 0,
        'TruncOptions': {
            'qNorm': 1
        }
    }),
    "Corr": AttrDict({
        'Family': 'Matern-5_2',
        'Type': 'Ellipsoidal',
        'Isotropic': True,
        'Nugget': 10**(-10),
    }),
    "EstimMethod": 'CV',
    "CV": AttrDict({
        'LeaveKOut': 1
    }),
    "Optim": AttrDict({
        'InitialValue': 1.0,
        'Bounds': [10**(-3), 10.0],
        'Display': 'none',
        'MaxIter': 20,
        'Tol': 10**(-4),
        'Method': 'HGA'
    })
}


def get_MetaOpts(X, y,
                 Trend=default_params['Trend'],
                 Corr=default_params['Corr'],
                 EstimMethod=default_params['EstimMethod'],
                 CV=default_params['CV'],
                 Optim=default_params['Optim']):
    MetaOpts = {
        'Type': 'Metamodel',
        'MetaType': 'Kriging',
        'Regression': {
            'SigmaNSQ': 'auto'
        }
    }
    MetaOpts["ExpDesign"] = {
        'X': X.tolist(),
        'Y': y.tolist()
    }
    MetaOpts['Trend'] = Trend
    MetaOpts['Corr'] = Corr
    MetaOpts['EstimMethod'] = EstimMethod
    MetaOpts['CV'] = CV
    MetaOpts['Optim'] = Optim

    return MetaOpts


class Kriging_Regressor_UQLab(BaseEstimator, RegressorMixin):
    '''
    Scikit-learn wrapper for UQLab Kriging regression.

    Parameters
    ----------
    Trend: dict, Options to specify the Kriging trend.
    Corr: dict, Options to specify the correlation function.
    EstimMethod: str, Select the method to estimate Kriging parameters
                 options are: 'CV', 'ML'.
    CV: dict, Options relevant to the cross-validation estimation method.
    Optim: dict, Options related to the optimization in the estimation
           of Kriging parameters.


    '''
    def __init__(self, uq=None, random_state: int = 2652124,
                 Trend=default_params['Trend'], Corr=default_params['Corr'],
                 EstimMethod=default_params['EstimMethod'],
                 CV=default_params['CV'], Optim=default_params['Optim']):
        self.uq = uq
        self.random_state = random_state
        self.Trend = Trend
        self.Corr = Corr
        self.EstimMethod = EstimMethod
        self.CV = CV
        self.Optim = Optim
        self.__model = None

    def fit(self, X, y):
        if self.uq is None:
            mySession = sessions.cloud()
            self.uq = mySession.cli
            mySession.reset()
            mySession.timeout = 900
        self.uq.rng(self.random_state, 'twister')

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        MetaOpts = get_MetaOpts(self.X_, self.y_, self.Trend, self.Corr,
                                self.EstimMethod, self.CV, self.Optim)

        self.__model = self.uq.createModel(MetaOpts)
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
        model_params = deepcopy(self.__model['Kriging'])
        return model_params
