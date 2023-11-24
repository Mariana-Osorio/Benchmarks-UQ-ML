# Configuration implementation of "Why do tree-based models still outperform
# deep learning on tabular data?"
from scipy.stats import loguniform, randint


config_random = {
    "model_type": "ml",
    "model_impl": "sklearn",
    "model_name": "xgb",
    "model__random_state": 2652124,
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "model__learning_rate": loguniform(1E-5, 0.7),
    "model__n_estimators": randint(100, 6001),
    "model__gamma": loguniform(1E-8, 7),
    "model__min_child_weight": [0.1, 1, 10, 100],
    "model__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1],
    "model__reg_alpha": loguniform(1E-8, 1E2),
    "model__reg_lambda": loguniform(1, 4),
}


config_default = {
    "model_type": "ml",
    "model_impl": "sklearn",
    "model_name": "xgb",
    "model__random_state": 2652124,
}
