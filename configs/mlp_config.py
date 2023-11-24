# Configuration implementation of "Why do tree-based models still outperform
# deep learning on tabular data?"
# from scipy.stats import loguniform
from configs.skorch_config import skorch_config_random, skorch_config_default

config_random = {
    "model_type": "ml",
    "model_impl": "skorch",
    "model_name": "mlp",
    "model__module__n_layers": [2, 4, 8, 16],
    "model__module__d_layers": [16, 32, 64, 128, 256, 512, 1024],
    "model__module__dropout": [0.0, 0.1, 0.2],
    "model__lr": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
    "model__lr_scheduler":  [True, False],
}

config_default = {
    "model_type": "ml",
    "model_impl": "skorch",
    "model_name": "mlp",
    "model__lr_scheduler": True,
    "model__module__n_layers": 8,
    "model__module__d_layers": 128,
    "model__module__dropout": 0.0,
    "model__lr": 5e-3,
}

config_random = dict(skorch_config_random, **config_random)

config_default = dict(skorch_config_default, **config_default)
