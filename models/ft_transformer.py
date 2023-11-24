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
import torch.nn.init as nn_init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor
import skorch
from skorch.callbacks import Checkpoint, EarlyStopping, \
    LRScheduler, EpochScoring
from models.utils import optimizer_dict, get_activation_fn, \
    get_nonglu_activation_fn, InputShapeSetterResnet


class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        # categories = None
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight,
                                     a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(
                    x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or
                                               m is not self.W_v):
                # gain is needed, W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key),
                              dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(nn.Module):
    """Transformer.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/
        representations/transformer
    - https://github.com/pytorch/fairseq/blob/
        1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/
        linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,
        regression: bool,
        categorical_indicator
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token,
                                   token_bias)
        n_tokens = self.tokenizer.n_tokens

        self.categorical_indicator = categorical_indicator
        self.regression = regression

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu')
                                             else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization \
            else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout,
                                   self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x) -> Tensor:
        if self.categorical_indicator is not None:
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long()  # TODO
        else:
            x_num = x
            x_cat = None
        # x_cat = None #FIXME
        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout,
                                       self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x


default_params = {
    "lr": 1e-4,
    "max_epochs": 1000,
    "batch_size": 512,
    "device": "cpu",
    "use_checkpoints": False,
    "optimizer_name": "adamw",
    "lr_scheduler": False,
    "es_patience": 40,
    "lr_patience": 30,
    "module__activation": "reglu",
    "module__token_bias": True,
    "module__prenormalization": True,
    "module__kv_compression": True,
    "module__kv_compression_sharing": "headwise",
    "module__initialization": "kaiming",
    "module__n_layers": 3,
    "module__n_heads": 8,
    "module__d_ffn_factor": 4./3,
    "module__attention_dropout": 0.2,
    "module__ffn_dropout": 0.1,
    "module__residual_dropout": 0.0,
    "d_token": 192,
    "optimizer__weight_decay": 1e-5,
}


class FT_Transformer_Regressor(skorch.NeuralNetRegressor):
    def __init__(self,
                 module=Transformer,
                 criterion=nn.MSELoss(),
                 lr=default_params["lr"],
                 max_epochs=default_params["max_epochs"],
                 batch_size=default_params["batch_size"],
                 callbacks='None',
                 predict_nonlinearity='auto',
                 warm_start=False,
                 verbose=0,
                 device=default_params["device"],
                 module__categorical_indicator=None,
                 module__categories=None,
                 use_checkpoints=default_params["use_checkpoints"],
                 optimizer_name=default_params["optimizer_name"],
                 lr_scheduler=default_params["lr_scheduler"],
                 es_patience=default_params["es_patience"],
                 lr_patience=default_params["lr_patience"],
                 random_state=2652124,
                 iterator_train__shuffle=True,
                 module__d_numerical=1,
                 module__d_out=1,
                 module__regression=True,
                 module__activation=default_params["module__activation"],
                 module__token_bias=default_params["module__token_bias"],
                 module__prenormalization=default_params[
                     "module__prenormalization"],
                 module__kv_compression=default_params[
                     "module__kv_compression"],
                 module__kv_compression_sharing=default_params[
                     "module__kv_compression_sharing"],
                 module__initialization=default_params[
                     "module__initialization"],
                 module__n_layers=default_params["module__n_layers"],
                 module__n_heads=default_params["module__n_heads"],
                 module__d_ffn_factor=default_params[
                     "module__d_ffn_factor"],
                 module__attention_dropout=default_params[
                     "module__attention_dropout"],
                 module__ffn_dropout=default_params["module__ffn_dropout"],
                 module__residual_dropout=default_params[
                     "module__residual_dropout"],
                 module__d_token=default_params["d_token"],
                 optimizer__weight_decay=default_params[
                     "optimizer__weight_decay"],
                 ):
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
        self.use_checkpoints = use_checkpoints
        self.optimizer_name = optimizer_name
        self.lr_scheduler = lr_scheduler
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.random_state = random_state
        self.iterator_train__shuffle = iterator_train__shuffle
        self.module__categories = module__categories
        self.module__categorical_indicator = module__categorical_indicator
        self.module__d_numerical = module__d_numerical
        self.module__d_out = module__d_out
        self.module__regression = module__regression
        self.module__activation = module__activation
        self.module__token_bias = module__token_bias
        self.module__prenormalization = module__prenormalization
        self.module__kv_compression = module__kv_compression
        self.module__kv_compression_sharing = module__kv_compression_sharing
        self.module__initialization = module__initialization
        self.module__n_layers = module__n_layers
        self.module__n_heads = module__n_heads
        self.module__d_ffn_factor = module__d_ffn_factor
        self.module__attention_dropout = module__attention_dropout
        self.module__ffn_dropout = module__ffn_dropout
        self.module__residual_dropout = module__residual_dropout
        self.module__d_token = module__d_token
        self.optimizer__weight_decay = optimizer__weight_decay

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
            module__categorical_indicator=self.module__categorical_indicator,
            module__d_numerical=self.module__d_numerical,
            module__d_out=self.module__d_out,
            module__regression=self.module__regression,
            module__activation=self.module__activation,
            module__token_bias=self.module__token_bias,
            module__prenormalization=self.module__prenormalization,
            module__kv_compression=self.module__kv_compression,
            module__kv_compression_sharing=self.module__kv_compression_sharing,
            module__initialization=self.module__initialization,
            module__n_layers=self.module__n_layers,
            module__n_heads=self.module__n_heads,
            module__d_ffn_factor=self.module__d_ffn_factor,
            module__attention_dropout=self.module__attention_dropout,
            module__ffn_dropout=self.module__ffn_dropout,
            module__residual_dropout=self.module__residual_dropout,
            module__d_token=self.module__d_token,
            optimizer__weight_decay=self.optimizer__weight_decay
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
            categorical_indicator=self.module__categorical_indicator,
            categories=self.module__categories)
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
        self.module__categorical_indicator = torch.BoolTensor(
            self.module__categorical_indicator) \
                if self.module__categorical_indicator is not None else None
        self.optimizer = optimizer_dict[self.optimizer_name]
        self.callbacks = self.set_callbacks()
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        if self.verbose > 0:
            print(f"Training {self.__class__.__name__}:")
            print(f"\tLayers: {self.module__n_layers}")
            print(f"\tHeads: {self.module__n_heads}")
            print(f"\tToken size: {self.module__d_token}")
            print(f"\tActivation: {self.module__activation}")
            print(f"\tAttention dropout: {self.module__attention_dropout}")
            print(f"\tFFN dropout: {self.module__ffn_dropout}")
            print(f"\tResidual dropout: {self.module__residual_dropout}")
            print(f"\tLearning rate: {self.lr}")
            print(f"\tLearning rate scheduler: {self.lr_scheduler}")
            print(f"\tBatch size: {self.batch_size}")
            print(f"\tOptimizer: {self.optimizer_name}")
            print(f"\tMax epochs: {self.max_epochs}")

        return super().fit(X, y)
