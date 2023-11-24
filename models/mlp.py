# Implementaton from the paper "Revisiting Deep Learning Models for
# Tabular Data" by Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov
# and Artem Babenko.
# https://github.com/yandex-research/tabular-dl-revisiting-models


import math
import numpy as np
import random
import typing as ty
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.utils import optimizer_dict
import skorch
from skorch.callbacks import Checkpoint, EarlyStopping, \
    LRScheduler, EpochScoring


default_params = {
    "lr": 5e-3,
    "max_epochs": 1000,
    "batch_size": 512,
    "device": "cpu",
    "use_checkpoints": False,
    "optimizer_name": "adamw",
    "lr_scheduler": True,
    "es_patience": 40,
    "lr_patience": 30,
    "module__n_layers": 8,
    "module__d_layers": 128,
    "module__dropout": 0.0,
}


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        n_layers: int,
        d_layers: int,  # CHANGED
        dropout: float,
        d_out: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        regression: bool,
        categorical_indicator
    ) -> None:
        super().__init__()

        self.regression = regression
        self.categorical_indicator = categorical_indicator  # Added

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories),
                                                    d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight,
                                     a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        d_layers = [d_layers for _ in range(n_layers)]  # CHANGED

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x):

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
                    x_cat + self.category_offsets[None]).view(x_cat.size(0),
                                                              -1)
            )
        x = torch.cat(x, dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x


class InputShapeSetterMLP(skorch.callbacks.Callback):
    def __init__(self, regression=False, batch_size=None,
                 categorical_indicator=None, categories=None):
        self.categorical_indicator = categorical_indicator
        self.regression = regression
        self.batch_size = batch_size
        self.categories = categories

    def on_train_begin(self, net, X, y):
        if self.categorical_indicator is None:
            d_in = X.shape[1]
            categories = None
        else:
            d_in = X.shape[1] - sum(self.categorical_indicator)
            if self.categories is None:
                categories = list((
                    X[:, self.categorical_indicator].max(0) + 1).astype(int))
            else:
                categories = self.categories
        net.set_params(module__d_in=d_in,
                       module__categories=categories,
                       module__d_out=2 if not self.regression else 1
                       )


class MLP_Regressor(skorch.NeuralNetRegressor):
    def __init__(self,
                 module=MLP,
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
                 module__d_in=1,
                 module__categories=None,
                 module__d_out=1,
                 module__regression=True,
                 module__n_layers=default_params["module__n_layers"],
                 module__d_layers=default_params["module__d_layers"],
                 module__dropout=default_params["module__dropout"],
                 module__d_embedding=128,
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
        self.module__d_in = module__d_in
        self.module__categories = module__categories
        self.module__d_out = module__d_out
        self.module__regression = module__regression
        self.module__n_layers = module__n_layers
        self.module__d_layers = module__d_layers
        self.module__dropout = module__dropout
        self.module__d_embedding = module__d_embedding
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
            module__d_in=self.module__d_in,
            module__categories=self.module__categories,
            module__d_out=self.module__d_out,
            module__regression=self.module__regression,
            module__n_layers=self.module__n_layers,
            module__d_layers=self.module__d_layers,
            module__dropout=self.module__dropout,
            module__d_embedding=self.module__d_embedding,
            module__categorical_indicator=self.categorical_indicator,
            train_split=self.train_split)

    @classmethod
    def from_params(cls, params):
        return cls(**params)

    def __sklearn_clone__(self):
        params = {key: val for key, val in
                  list(self.get_params().items())[:28]}
        return self.from_params(params)

    def set_callbacks(self):
        input_shape_setter = InputShapeSetterMLP(
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
            print(f"\tLayer size: {self.module__d_layers}")
            print(f"\tDropout: {self.module__dropout}")
            print(f"\tLearning rate: {self.lr}")
            print(f"\tLearning rate scheduler: {self.lr_scheduler}")
            print(f"\tBatch size: {self.batch_size}")
            print(f"\tOptimizer: {self.optimizer_name}")
            print(f"\tMax epochs: {self.max_epochs}")

        return super().fit(X, y)
