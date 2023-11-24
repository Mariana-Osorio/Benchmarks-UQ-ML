# Configuration implementation of "Why do tree-based models still outperform
# deep learning on tabular data?"
from numpy import arange
from scipy.stats import logser

config_random = {
    "model_type": "ml",
    "model_impl": "sklearn",
    "model_name": "rf",
    "model__random_state": 2652124,
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__n_estimators": arange(50, 560, 100),
    "model__min_samples_split": [2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
    "model__min_samples_leaf": logser(0.95),
    "model__bootstrap": [True, False],
    "model__min_impurity_decrease": [0]*7+[0.01]+[0.02]+[0.05]
}

config_default = {
    "model_type": "ml",
    "model_impl": "sklearn",
    "model_name": "rf",
    "model__random_state": 2652124,
}
