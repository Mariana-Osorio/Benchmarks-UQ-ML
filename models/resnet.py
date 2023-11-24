# Implementaton from the paper "Revisiting Deep Learning Models for
# Tabular Data" by Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov
# and Artem Babenko.
# https://github.com/yandex-research/tabular-dl-revisiting-models


import numpy as np
import random
import math
import typing as ty
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import skorch
from skorch.callbacks import Checkpoint, EarlyStopping, \
    LRScheduler, EpochScoring
from models.utils import optimizer_dict, get_activation_fn, \
    get_nonglu_activation_fn, InputShapeSetterResnet


class ResNet(nn.Module):
    def __init__(
            self,
            *,
            d_numerical: int,
            categories: ty.Optional[ty.List[int]],
            d_embedding: int,
            d: int,
            d_hidden_factor: float,
            n_layers: int,
            activation: str,
            normalization: str,
            hidden_dropout: float,
            residual_dropout: float,
            d_out: int,
            regression: bool,
            categorical_indicator
    ) -> None:
        super().__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)
        self.categorical_indicator = categorical_indicator
        self.regression = regression
        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(int(sum(categories)),
                                                    d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight,
                                     a=math.sqrt(5))

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d,
                            d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x) -> Tensor:
        if self.categorical_indicator is not None:
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long()  # TODO
        else:
            x_num = x
            x_cat = None
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(
                    x_cat + self.category_offsets[None]).view(
                        x_cat.size(0), -1))

        x = torch.cat(x, dim=-1)
        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x


default_params = {
    "lr": 1e-3,
    "max_epochs": 1000,
    "batch_size": 512,
    "device": "cpu",
    "use_checkpoints": False,
    "optimizer_name": "adamw",
    "lr_scheduler": True,
    "es_patience": 40,
    "lr_patience": 30,
    "module__activation": "reglu",
    "module__normalization": "batchnorm",
    "module__n_layers": 8,
    "module__d": 256,
    "module__d_hidden_factor": 2,
    "module__hidden_dropout":  0.2,
    "module__residual_dropout": 0.2,
    "optimizer__weight_decay": 1e-7,
    "module__d_embedding": 128,
}


class ResNet_Regressor(skorch.NeuralNetRegressor):
    def __init__(self,
                 module=ResNet,
                 criterion=torch.nn.MSELoss(),
                 lr=default_params["lr"],
                 max_epochs=default_params["max_epochs"],
                 batch_size=default_params["batch_size"],
                 callbacks='None',
                 predict_nonlinearity='auto',
                 warm_start=False,
                 verbose=0,
                 device=default_params["device"],
                 categorical_indicator=None,
                 categories=None,
                 use_checkpoints=default_params["use_checkpoints"],
                 optimizer_name=default_params["optimizer_name"],
                 lr_scheduler=default_params["lr_scheduler"],
                 es_patience=default_params["es_patience"],
                 lr_patience=default_params["lr_patience"],
                 random_state=2652124,
                 iterator_train__shuffle=True,
                 module__categories=None,
                 module__d_numerical=1,
                 module__d_out=1,
                 module__regression=True,
                 module__activation=default_params["module__activation"],
                 module__normalization=default_params["module__normalization"],
                 module__n_layers=default_params["module__n_layers"],
                 module__d=default_params["module__d"],
                 module__d_hidden_factor=default_params[
                     "module__d_hidden_factor"],
                 module__hidden_dropout=default_params[
                     "module__hidden_dropout"],
                 module__residual_dropout=default_params[
                     "module__residual_dropout"],
                 module__d_embedding=default_params["module__d_embedding"],
                 optimizer__weight_decay=default_params[
                     "optimizer__weight_decay"],
                 train_split=skorch.dataset.ValidSplit(0.05)):

        torch.manual_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)
        torch.use_deterministic_algorithms(True)

        self.module = module
        self.criterion = criterion
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.predict_nonlinearity = predict_nonlinearity
        self.warm_start = warm_start
        self.verbose = verbose
        self.device = device
        self.categorical_indicator = categorical_indicator
        self.categories = categories
        self.use_checkpoints = use_checkpoints
        self.optimizer_name = optimizer_name
        self.lr_scheduler = lr_scheduler
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.random_state = random_state
        self.iterator_train__shuffle = iterator_train__shuffle
        self.module__categories = module__categories
        self.module__d_numerical = module__d_numerical
        self.module__d_out = module__d_out
        self.module__regression = module__regression
        self.module__activation = module__activation
        self.module__normalization = module__normalization
        self.module__n_layers = module__n_layers
        self.module__d = module__d
        self.module__d_hidden_factor = module__d_hidden_factor
        self.module__hidden_dropout = module__hidden_dropout
        self.module__residual_dropout = module__residual_dropout
        self.module__d_embedding = module__d_embedding
        self.optimizer__weight_decay = optimizer__weight_decay
        self.train_split = train_split

        super().__init__(
            module=self.module,
            criterion=self.criterion,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            predict_nonlinearity=self.predict_nonlinearity,
            warm_start=self.warm_start,
            verbose=self.verbose,
            device=self.device,
            iterator_train__shuffle=self.iterator_train__shuffle,
            module__categories=self.module__categories,
            module__categorical_indicator=self.categorical_indicator,
            module__d_numerical=self.module__d_numerical,
            module__d_out=self.module__d_out,
            module__regression=self.module__regression,
            module__activation=self.module__activation,
            module__normalization=self.module__normalization,
            module__n_layers=self.module__n_layers,
            module__d=self.module__d,
            module__d_hidden_factor=self.module__d_hidden_factor,
            module__hidden_dropout=self.module__hidden_dropout,
            module__residual_dropout=self.module__residual_dropout,
            module__d_embedding=self.module__d_embedding,
            optimizer__weight_decay=self.optimizer__weight_decay,
            train_split=self.train_split,
        )

    @classmethod
    def from_params(cls, params):
        return cls(**params)

    def __sklearn_clone__(self):
        params = {key: val for key, val in
                  list(self.get_params().items())[:28]}
        return self.from_params(params)

    def set_callbacks(self):
        input_shape_setter = InputShapeSetterResnet(
            regression=True,
            categorical_indicator=self.categorical_indicator,
            categories=self.categories)
        early_stopping = EarlyStopping(monitor="valid_loss",
                                       patience=self.es_patience)
        epoch_scoring = EpochScoring(scoring='neg_root_mean_squared_error',
                                     name='train_accuracy', on_train=True)

        callbacks = [input_shape_setter, early_stopping, epoch_scoring]

        if self.lr_scheduler:
            lr_scheduler = LRScheduler(
                policy=ReduceLROnPlateau, patience=self.lr_patience,
                min_lr=2e-5, factor=0.2)
            callbacks.append(lr_scheduler)

        if self.use_checkpoints:
            checkpoint = Checkpoint(dirname="skorch_cp",
                                    f_params=r"params_{}.pt".format(id),
                                    f_optimizer=None, f_criterion=None)
            callbacks.append(checkpoint)
        return callbacks

    def fit(self, X, y):
        self.categorical_indicator = torch.BoolTensor(
            self.categorical_indicator) \
                if self.categorical_indicator is not None else None
        self.optimizer = optimizer_dict[self.optimizer_name]
        self.callbacks = self.set_callbacks()
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        if self.verbose > 0:
            print(f"Training {self.__class__.__name__}:")
            print(f"\tLayers: {self.module__n_layers}")
            print(f"\tLayer size: {self.module__d}")
            print(f"\tHidden factor: {self.module__d_hidden_factor}")
            print(f"\tActivation: {self.module__activation}")
            print(f"\tNormalization: {self.module__normalization}")
            print(f"\tHidden dropout: {self.module__hidden_dropout}")
            print(f"\tResidual dropout: {self.module__residual_dropout}")
            print(f"\tLearning rate: {self.lr}")
            print(f"\tLearning rate scheduler: {self.lr_scheduler}")
            print(f"\tBatch size: {self.batch_size}")
            print(f"\tOptimizer: {self.optimizer_name}")
            print(f"\tMax epochs: {self.max_epochs}")

        return super().fit(X, y)
