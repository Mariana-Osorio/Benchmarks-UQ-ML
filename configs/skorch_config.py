# Configuration implementation of "Why do tree-based models still outperform
# deep learning on tabular data?"

# Default config for all skorch model
# Can be overwritten in the config file of a model

skorch_config_default = {
    "model_type": "ml",
    "model_impl": "skorch",
    "model__device": "cpu",
    "model__use_checkpoints": False,
    "model__optimizer_name": "adamw",
    "model__batch_size": 512,
    "model__max_epochs": 1000,
}


skorch_config_random = {
    "model_type": "ml",
    "model_impl": "skorch",
    "model__device": "cpu",
    "model__use_checkpoints": [True, False],
    "model__optimizer_name": ["adamw", "adam", "sgd"],
    "model__batch_size": [256, 512, 1024],
    "model__max_epochs": [300, 600, 1000],
}
