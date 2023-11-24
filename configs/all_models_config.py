from copy import deepcopy

# ML models
from sklearn.ensemble import RandomForestRegressor
from configs.rf_config import config_default as config_default_rf
from configs.rf_config import config_random as config_random_rf
from xgboost import XGBRegressor
from configs.xgb_config import config_default as config_default_xgb
from configs.xgb_config import config_random as config_random_xgb
from models.mlp import MLP_Regressor
from configs.mlp_config import config_default as config_default_mlp
from configs.mlp_config import config_random as config_random_mlp
from models.resnet import ResNet_Regressor
from configs.resnet_config import config_default as config_default_resnet
from configs.resnet_config import config_random as config_random_resnet
from models.ft_transformer import FT_Transformer_Regressor
from configs.ft_transformer_config import config_default as \
    config_default_ft_transformer
from configs.ft_transformer_config import config_random as \
    config_random_ft_transformer

# UQ models
from models.kriging import Kriging_Regressor_UQLab
from configs.kriging_config import config_default as config_default_kriging
from configs.kriging_config import config_random as config_random_kriging
from models.pce import PCE_Regressor_UQLab
from configs.pce_config import config_default as config_default_pce
from configs.pce_config import config_random as config_random_pce
from models.pck import PCK_Regressor_UQLab
from configs.pck_config import config_default as config_default_pck
from configs.pck_config import config_random as config_random_pck

total_config = {}
model_keyword_dic = {}

keyword = "rf"
total_config[keyword] = {
    "default": deepcopy(config_default_rf),
    "random": deepcopy(config_random_rf)
}
model_keyword_dic[config_default_rf["model_name"]] = \
    RandomForestRegressor

keyword = "xgb"
total_config[keyword] = {
    "default": deepcopy(config_default_xgb),
    "random": deepcopy(config_random_xgb)
}
model_keyword_dic[config_default_xgb["model_name"]] = \
    XGBRegressor

keyword = "mlp"
total_config[keyword] = {
    "default": deepcopy(config_default_mlp),
    "random": deepcopy(config_random_mlp)
}
model_keyword_dic[config_default_mlp["model_name"]] = \
    MLP_Regressor

keyword = "resnet"
total_config[keyword] = {
    "default": deepcopy(config_default_resnet),
    "random": deepcopy(config_random_resnet)
}
model_keyword_dic[config_default_resnet["model_name"]] = \
    ResNet_Regressor

keyword = "ft_transformer"
total_config[keyword] = {
    "default": deepcopy(config_default_ft_transformer),
    "random": deepcopy(config_random_ft_transformer)
}
model_keyword_dic[config_default_ft_transformer["model_name"]] =\
    FT_Transformer_Regressor

keyword = "kriging"
total_config[keyword] = {
    "default": deepcopy(config_default_kriging),
    "random": deepcopy(config_random_kriging)
}
model_keyword_dic[config_default_kriging["model_name"]] = \
    Kriging_Regressor_UQLab

keyword = "pce"
total_config[keyword] = {
    "default": deepcopy(config_default_pce),
    "random": deepcopy(config_random_pce)
}
model_keyword_dic[config_default_pce["model_name"]] = \
    PCE_Regressor_UQLab

keyword = "pck"
total_config[keyword] = {
    "default": deepcopy(config_default_pck),
    "random": deepcopy(config_random_pck)
}
model_keyword_dic[config_default_pck["model_name"]] = \
    PCK_Regressor_UQLab
