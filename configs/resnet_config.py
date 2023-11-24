# Configuration implementation of "Why do tree-based models still outperform
# deep learning on tabular data?"

from configs.skorch_config import skorch_config_random, skorch_config_default
from scipy.stats import randint, uniform, loguniform

config_random = {
    "model_type": "ml",
    "model_impl": "skorch",
    "model_name": "resnet",
    "model__module__normalization": ["batchnorm", "layernorm"],
    "model__module__n_layers": list(range(17)),
    "model__module__d": randint(64, 1025),
    "model__module__d_hidden_factor": uniform(1, 4),
    "model__module__hidden_dropout": uniform(0.0, 0.5),
    "model__module__residual_dropout": uniform(0.0, 0.5),
    "model__lr": loguniform(1e-5, 1e-2),
    "model__optimizer__weight_decay": loguniform(1e-8, 1e-3),
    "model__module__d_embedding": randint(64, 513),
    "model__lr_scheduler": [True, False]
}

config_default = {
    "model_type": "ml",
    "model_impl": "skorch",
    "model_name": "resnet",
    "model__lr_scheduler":  True,
    "model__module__activation": "reglu",
    "model__module__normalization":  "batchnorm",
    "model__module__n_layers": 8,
    "model__module__d": 256,
    "model__module__d_hidden_factor": 2,
    "model__module__hidden_dropout": 0.2,
    "model__module__residual_dropout": 0.2,
    "model__lr": 1e-3,
    "model__optimizer__weight_decay": 1e-7,
    "model__module__d_embedding": 128,
}

config_random = dict(skorch_config_random, **config_random)

config_default = dict(skorch_config_default, **config_default)
