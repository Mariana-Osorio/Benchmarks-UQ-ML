# Configuration for the UQpyLab PCE model.

import numpy as np
from models.utils import AttrDict

config_default = {
    "model_type": "uq",
    "model_impl": "uqpylab-sklearn",
    "model_name": "pce",
    'model__Degree': list(range(2, 16)),
    'model__DegreeEarlyStop': True,
    'model__TruncOptions': AttrDict({
        'qNorm': np.arange(0.5, 1.01, 0.1).tolist(),
        'MaxInteraction': 2,
    }),
    'model__qNormEarlyStop': True,
    'model__Method': 'OMP',
    'model__OLS': AttrDict({
        'TargetAccuracy': 0.0
    }),
    'model__LARS': AttrDict({
        'LarsEarlyStop': 'default',
        'TargetAccuracy': 0.0,
        'KeepIterations': False,
        'HybridLoo': True,
        'ModifiedLoo': 1
    }),
    'model__OMP': AttrDict({
        'OmpEarlyStop': 'default',
        'TargetAccuracy': 0.0,
        'KeepIterations': False,
        'ModifiedLoo': 1
    }),
    'model__SP': AttrDict({
        'NNZ': [],
        'CVMethod': 'loo',
        'NumFolds': 5,
        'ModifiedLoo': 1
    }),
    'model__BCS': AttrDict({
        'NumFolds': 10,
        'ModifiedLoo': 0
    }),
    'model__Bootstrap': 'default'
}

config_random = {
    "model_type": "uq",
    "model_impl": "uqpylab-sklearn",
    "model_name": "pce",
    'model__Method': ['OMP', 'SP', 'LARS', 'BCS'],
    'model__TruncOptions__MaxInteraction': [1, 2, 'default'],
}
