# Configuration implementation of "Why do tree-based models still outperform
# deep learning on tabular data?"

from configs.skorch_config import skorch_config_random, skorch_config_default
from scipy.stats import randint, uniform, loguniform

config_random = {
    "model_type": "ml",
    "model_impl": "skorch",
    "model_name": "ft_transformer",
    "model__module__activation": "reglu",
    "model__module__token_bias": True,
    "model__module__prenormalization": True,
    "model__module__kv_compression": [True, False],
    "model__module__kv_compression_sharing": ["headwise", 'key-value'],
    "model__module__initialization": "kaiming",
    "model__module__n_layers": randint(1, 7),
    "model__module__n_heads": 8,
    "model__module__d_ffn_factor": uniform(2. / 3, 8. / 3),
    "model__module__ffn_dropout": uniform(0.0, 0.5),
    "model__module__attention_dropout": uniform(0.0, 0.5),
    "model__module__residual_dropout": uniform(0.0, 0.5),
    "model__lr": loguniform(1e-5, 1e-3),
    "model__optimizer__weight_decay": loguniform(1e-6, 1e-3),
    "d_token": randint(64, 513),
    "model__lr_scheduler": [True, False],
}

config_default = {
    "model_type": "ml",
    "model_impl": "skorch",
    "model_name": "ft_transformer",
    "model__lr_scheduler": False,
    "model__module__activation": "reglu",
    "model__module__token_bias": True,
    "model__module__prenormalization": True,
    "model__module__kv_compression": True,
    "model__module__kv_compression_sharing": "headwise",
    "model__module__initialization": "kaiming",
    "model__module__n_layers": 3,
    "model__module__n_heads": 8,
    "model__module__d_ffn_factor": 4. / 3,
    "model__module__ffn_dropout": 0.1,
    "model__module__attention_dropout": 0.2,
    "model__module__residual_dropout": 0.0,
    "model__lr": 1e-4,
    "model__optimizer__weight_decay": 1e-5,
    "d_token": 192,
}

config_random = dict(skorch_config_random, **config_random)

config_default = dict(skorch_config_default, **config_default)
