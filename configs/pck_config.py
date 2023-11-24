# Configuration for the UQpyLab PCK (PCE-Kriging) model.

from models.utils import AttrDict

config_default = {
    "model_type": "uq",
    "model_impl": "uqpylab-sklearn",
    "model_name": "pck",
    "model__Mode": "Sequential",
    "model__Corr": AttrDict({
        'Family': 'Matern-5_2',
        'Type': 'Ellipsoidal',
        'Isotropic': True,
    }),
    "model__EstimMethod": 'CV',
}

config_random = {
    "model_type": "uq",
    "model_impl": "uqpylab-sklearn",
    "model_name": "pck",
    "model__Mode": ['Sequential', 'Optimal'],
    "model__Corr__Family": ['Matern-5_2', 'Gaussian'],
    "model__Corr__Type": ['Ellipsoidal', 'Separable'],
    "model__Corr__Isotropic": [True, False],
    "model__EstimMethod": ['CV', 'ML'],
}
