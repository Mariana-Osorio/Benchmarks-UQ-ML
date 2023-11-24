# Competitor configuration settings
from sklearn.preprocessing import StandardScaler

# random forest
rf_config = {
    "name": "rf",
    "config": {
        "random": True,
        "pre_processing": None
    }
}

# xgboost
xgb_config = {
    "name": "xgb",
    "config": {
        "random": True,
        "pre_processing": None
    }
}

# mlp
mlp_config = {
    "name": "mlp",
    "config": {
        "random": True,
        "pre_processing": StandardScaler()
    }
}

# resnet
resnet_config = {
    "name": "resnet",
    "config": {
        "random": True,
        "pre_processing": StandardScaler()
    }
}

# ft_transformer
ft_transformer_config = {
    "name": "ft_transformer",
    "config": {
        "random": True,
        "pre_processing": StandardScaler()
    }
}

# kriging
kriging_config = {
    "name": "kriging",
    "config": {
        "random": True,
        "pre_processing": None
    }
}

# pce
pce_config = {
    "name": "pce",
    "config": {
        "random": True,
        "pre_processing": None
    }
}

# pck
pck_config = {
    "name": "pck",
    "config": {
        "random": True,
        "pre_processing": None
    }
}

# Competitor config
competitor_config = {
    "rf": rf_config,
    "xgb": xgb_config,
    "mlp": mlp_config,
    "resnet": resnet_config,
    "ft_transformer": ft_transformer_config,
    "kriging": kriging_config,
    "pce": pce_config,
    "pck": pck_config
}
