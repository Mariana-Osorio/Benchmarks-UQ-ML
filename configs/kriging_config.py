# Configuration for the UQpyLab Kriging model.
from models.utils import AttrDict


config_default = {
    "model_type": "uq",
    "model_impl": "uqpylab-sklearn",
    "model_name": "kriging",
    "model__Trend": AttrDict({
        'Type': 'ordinary',
        'Degree': 0,
        'TruncOptions': {
            'qNorm': 1
        }
    }),
    "model__Corr": AttrDict({
        'Family': 'Matern-5_2',
        'Type': 'Ellipsoidal',
        'Isotropic': True,
        'Nugget': 10**(-10),
    }),
    "model__EstimMethod": 'CV',
    "model__CV": AttrDict({
        'LeaveKOut': 1
    }),
    "model__Optim": AttrDict({
        'InitialValue': 1.0,
        'Bounds': [10**(-3), 10.0],
        'Display': 'none',
        'MaxIter': 20,
        'Tol': 10**(-4),
        'Method': 'HGA'
    })
}

config_random = {
    "model_type": "uq",
    "model_impl": "uqpylab-sklearn",
    "model_name": "kriging",
    "model__Corr__Family": ['Matern-5_2', 'Gaussian'],
    "model__Corr__Type": ['Ellipsoidal', 'Separable'],
    "model__Corr__Isotropic": [True, False],
    "model__Trend__Type": ['ordinary', 'linear', 'quadratic'],
    "model__EstimMethod": ['CV', 'ML'],
}
