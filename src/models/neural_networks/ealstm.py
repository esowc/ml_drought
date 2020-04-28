import torch
from torch import nn

from pathlib import Path
from copy import copy
import xarray as xr

from typing import Dict, List, Optional, Tuple, Union

from .base import NNBase


class EARecurrentNetwork(NNBase):

    model_name = "ealstm"

    def __init__(
        self,
        hidden_size: int,
        dense_features: Optional[List[int]] = None,
        rnn_dropout: float = 0.25,
        data_folder: Path = Path("data"),
        batch_size: int = 1,
        experiment: str = "one_month_forecast",
        pred_months: Optional[List[int]] = None,
        include_latlons: bool = False,
        include_pred_month: bool = True,
        include_monthly_aggs: bool = True,
        include_yearly_aggs: bool = True,
        surrounding_pixels: Optional[int] = None,
        ignore_vars: Optional[List[str]] = None,
        static: Optional[str] = "features",
        static_embedding_size: Optional[int] = None,
        device: str = "cuda:0",
        predict_delta: bool = False,
        spatial_mask: Union[xr.DataArray, Path] = None,
        include_prev_y: bool = True,
        normalize_y: bool = True,
    ) -> None:
        super().__init__(
            data_folder,
            batch_size,
            experiment,
            pred_months,
            include_pred_month,
            include_latlons,
            include_monthly_aggs,
            include_yearly_aggs,
            surrounding_pixels,
            ignore_vars,
            static,
            device,
            predict_delta=predict_delta,
            spatial_mask=spatial_mask,
            include_prev_y=include_prev_y,
            normalize_y=normalize_y,
        )

        # to initialize and save the model
        self.hidden_size = hidden_size
        self.rnn_dropout = rnn_dropout
        self.input_dense = copy(
            dense_features
        )  # this is to make sure we can reload the model
        if dense_features is None:
            dense_features = []
        self.dense_features = dense_features
        if static_embedding_size is not None:
            assert (
                static is not None
            ), "Can't have a static embedding without input static information!"
        self.static_embedding_size = static_embedding_size

        self.features_per_month: Optional[int] = None
        self.current_size: Optional[int] = None
        self.yearly_agg_size: Optional[int] = None
        self.static_size: Optional[int] = None

    def save_model(self):

        assert self.model is not None, "Model must be trained before it can be saved!"

        model_dict = {
            "model": {
                "state_dict": self.model.state_dict(),
                "features_per_month": self.features_per_month,
                "current_size": self.current_size,
                "yearly_agg_size": self.yearly_agg_size,
                "static_size": self.static_size,
            },
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "rnn_dropout": self.rnn_dropout,
            "dense_features": self.input_dense,
            "include_pred_month": self.include_pred_month,
            "include_latlons": self.include_latlons,
            "surrounding_pixels": self.surrounding_pixels,
            "include_monthly_aggs": self.include_monthly_aggs,
            "include_yearly_aggs": self.include_yearly_aggs,
            "static_embedding_size": self.static_embedding_size,
            "experiment": self.experiment,
            "ignore_vars": self.ignore_vars,
            "static": self.static,
            "device": self.device,
            "spatial_mask": self.spatial_mask,
            "include_prev_y": self.include_prev_y,
            "normalize_y": self.normalize_y,
        }

        torch.save(model_dict, self.model_dir / "model.pt")

    def load(
        self,
        state_dict: Dict,
        features_per_month: int,
        current_size: Optional[int],
        yearly_agg_size: Optional[int],
        static_size: Optional[int],
    ) -> None:
        self.features_per_month = features_per_month
        self.current_size = current_size
        self.yearly_agg_size = yearly_agg_size
        self.static_size = static_size

        self.model: EALSTM = EALSTM(
            features_per_month=self.features_per_month,
            dense_features=self.dense_features,
            hidden_size=self.hidden_size,
            rnn_dropout=self.rnn_dropout,
            include_pred_month=self.include_pred_month,
            experiment=self.experiment,
            current_size=self.current_size,
            yearly_agg_size=self.yearly_agg_size,
            include_latlons=self.include_latlons,
            static_size=self.static_size,
            static_embedding_size=self.static_embedding_size,
            include_prev_y=self.include_prev_y,
        )
        self.model.to(torch.device(self.device))
        self.model.load_state_dict(state_dict)

    def _initialize_model(self, x_ref: Optional[Tuple[torch.Tensor, ...]]) -> nn.Module:
        if self.features_per_month is None:
            assert (
                x_ref is not None
            ), f"x_ref can't be None if features_per_month or current_size is not defined"
            self.features_per_month = x_ref[0].shape[-1]
        if self.experiment == "nowcast":
            if self.current_size is None:
                assert (
                    x_ref is not None
                ), f"x_ref can't be None if features_per_month or current_size is not defined"
                self.current_size = x_ref[3].shape[-1]
        if self.include_yearly_aggs:
            if self.yearly_agg_size is None:
                assert x_ref is not None
                self.yearly_agg_size = x_ref[4].shape[-1]
        if self.static:
            if self.static_size is None:
                if self.static == "features":
                    assert x_ref is not None
                    self.static_size = x_ref[5].shape[-1]
                elif self.static == "embeddings":
                    self.static_size = self.num_locations

        model = EALSTM(
            features_per_month=self.features_per_month,
            dense_features=self.dense_features,
            hidden_size=self.hidden_size,
            rnn_dropout=self.rnn_dropout,
            include_pred_month=self.include_pred_month,
            experiment=self.experiment,
            yearly_agg_size=self.yearly_agg_size,
            current_size=self.current_size,
            include_latlons=self.include_latlons,
            static_size=self.static_size,
            static_embedding_size=self.static_embedding_size,
            include_prev_y=self.include_prev_y,
        )

        return model.to(torch.device(self.device))


class EALSTM(nn.Module):
    def __init__(
        self,
        features_per_month,
        dense_features,
        hidden_size,
        rnn_dropout,
        include_latlons,
        include_pred_month,
        experiment,
        include_prev_y,
        yearly_agg_size=None,
        current_size=None,
        static_size=None,
        static_embedding_size=None,
    ):
        super().__init__()

        self.experiment = experiment
        self.include_pred_month = include_pred_month
        self.include_latlons = include_latlons
        self.include_yearly_agg = False
        self.include_static = False
        self.include_prev_y = include_prev_y

        assert (
            include_latlons
            or (yearly_agg_size is not None)
            or (static_size is not None)
        ), "Need at least one of {latlons, yearly mean, static} for the static input"
        ea_static_size = 0
        if self.include_latlons:
            ea_static_size += 2
        if yearly_agg_size is not None:
            self.include_yearly_agg = True
            ea_static_size += yearly_agg_size
        if static_size is not None:
            self.include_static = True
            ea_static_size += static_size
        if include_pred_month:
            ea_static_size += 12
        if self.include_prev_y:
            ea_static_size += 1

        self.use_static_embedding = False
        if static_embedding_size:
            assert (
                self.include_static is not None
            ), "Can't have a static embedding without a static input!"
            self.use_static_embedding = True
            self.static_embedding = nn.Linear(ea_static_size, static_embedding_size)

            ea_static_size = static_embedding_size

        self.dropout = nn.Dropout(rnn_dropout)
        self.rnn = OrgEALSTMCell(
            input_size_dyn=features_per_month,
            input_size_stat=ea_static_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.hidden_size = hidden_size
        self.rnn_dropout = nn.Dropout(rnn_dropout)

        dense_input_size = hidden_size
        if experiment == "nowcast":
            assert current_size is not None
            dense_input_size += current_size

        dense_features.insert(0, dense_input_size)
        if dense_features[-1] != 1:
            dense_features.append(1)

        # add linear layer with nonlinear activation functions
        dense_layers = []
        for i in range(1, len(dense_features)):
            dense_layers.append(
                nn.Linear(
                    # in = size of previous dense layer
                    in_features=dense_features[i - 1],
                    # out = size of current dense layer
                    out_features=dense_features[i],
                )
            )
            if i < len(dense_features) - 1:
                # add a ReLU to all layers except the final layer
                dense_layers.append(nn.ReLU())

        self.dense_layers = nn.ModuleList(dense_layers)

        self.initialize_weights()

    def initialize_weights(self):
        for dense_layer in self.dense_layers:
            # initialise weights for all linear layers
            if not isinstance(dense_layer, nn.ReLU):
                nn.init.kaiming_uniform_(dense_layer.weight.data)
                nn.init.constant_(dense_layer.bias.data, 0)

    def forward(
        self,
        x,
        pred_month=None,
        latlons=None,
        current=None,
        yearly_aggs=None,
        static=None,
        prev_y=None,
    ):

        assert (
            (yearly_aggs is not None) or (latlons is not None) or (static is not None)
        ), "latlons, yearly means and static can't all be None"

        static_x = []
        if self.include_latlons:
            assert latlons is not None
            static_x.append(latlons)
        if self.include_yearly_agg:
            assert yearly_aggs is not None
            static_x.append(yearly_aggs)
        if self.include_static:
            static_x.append(static)
        if self.include_pred_month:
            static_x.append(pred_month)
        if self.include_prev_y:
            static_x.append(prev_y)

        static_tensor = torch.cat(static_x, dim=-1)

        if self.use_static_embedding:
            static_tensor = self.static_embedding(static_tensor)

        hidden_state, cell_state = self.rnn(x, static_tensor)

        x = self.rnn_dropout(hidden_state[:, -1, :])

        if self.experiment == "nowcast":
            assert current is not None
            x = torch.cat((x, current), dim=-1)

        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
        return x


class EALSTMCell(nn.Module):
    """See below. Implemented using modules so it can be explained with shap
    """

    def __init__(
        self,
        input_size_dyn: int,
        input_size_stat: int,
        hidden_size: int,
        batch_first: bool = True,
    ):
        super().__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.forget_gate_i = nn.Linear(
            in_features=input_size_dyn, out_features=hidden_size, bias=False
        )
        self.forget_gate_h = nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=True
        )

        self.update_gate = nn.Sequential(
            *[
                nn.Linear(in_features=input_size_stat, out_features=hidden_size),
                nn.Sigmoid(),
            ]
        )

        self.update_candidates_i = nn.Linear(
            in_features=input_size_dyn, out_features=hidden_size, bias=False
        )
        self.update_candidates_h = nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=True
        )

        self.output_gate_i = nn.Linear(
            in_features=input_size_dyn, out_features=hidden_size, bias=False
        )
        self.output_gate_h = nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=True
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        self._reset_i(self.forget_gate_i)
        self._reset_i(self.update_candidates_i)
        self._reset_i(self.output_gate_i)
        self._reset_i(self.update_gate[0])
        nn.init.constant_(self.update_gate[0].bias.data, val=0)

        self._reset_h(self.forget_gate_h, self.hidden_size)
        self._reset_h(self.update_candidates_h, self.hidden_size)
        self._reset_h(self.output_gate_h, self.hidden_size)

    @staticmethod
    def _reset_i(layer):
        nn.init.orthogonal(layer.weight.data)

    @staticmethod
    def _reset_h(layer, hidden_size):
        weight_hh_data = torch.eye(hidden_size)
        layer.weight.data = weight_hh_data
        nn.init.constant_(layer.bias.data, val=0)

    def forward(self, x_d, x_s):
        """[summary]
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor, containing a batch of sequences of the dynamic features. Shape has to match
            the format specified with batch_first.
        x_s : torch.Tensor
            Tensor, containing a batch of static features.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # calculate input gate only once because inputs are static
        i = self.update_gate(x_s)

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            forget_state = self.sigmoid(
                self.forget_gate_i(x_d[t]) + self.forget_gate_h(h_0)
            )
            cell_candidates = self.tanh(
                self.update_candidates_i(x_d[t]) + self.update_candidates_h(h_0)
            )
            output_state = self.sigmoid(
                self.output_gate_i(x_d[t]) + self.output_gate_h(h_0)
            )

            c_1 = forget_state * c_0 + i * cell_candidates
            h_1 = output_state * self.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n


class OrgEALSTMCell(nn.Module):
    """Implementation of the Entity-Aware-LSTM (EA-LSTM)

    This code was copied from
    https://github.com/kratzert/ealstm_regional_modeling/blob/master/papercode/ealstm.py
    and is currently used just to test our implementation of the EALSTMCell

    Parameters
    ----------
    input_size_dyn : int
        Number of dynamic features, which are those, passed to the LSTM at each time step.
    input_size_stat : int
        Number of static features, which are those that are used to modulate the input gate.
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(
        self,
        input_size_dyn: int,
        input_size_stat: int,
        hidden_size: int,
        batch_first: bool = True,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(  # type: ignore
            torch.FloatTensor(  # type: ignore
                input_size_dyn, 3 * hidden_size
            )
        )  # type: ignore
        self.weight_hh = nn.Parameter(  # type: ignore
            torch.FloatTensor(  # type: ignore
                hidden_size, 3 * hidden_size
            )
        )  # type: ignore
        self.weight_sh = nn.Parameter(  # type: ignore
            torch.FloatTensor(  # type: ignore
                input_size_stat, hidden_size
            )
        )  # type: ignore
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))  # type: ignore
        self.bias_s = nn.Parameter(torch.FloatTensor(hidden_size))  # type: ignore

        # module activations for shap
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[: self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d, x_s):
        """[summary]
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor, containing a batch of sequences of the dynamic features. Shape has to match
            the format specified with batch_first.
        x_s : torch.Tensor
            Tensor, containing a batch of static features.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())

        # calculate input gate only once because inputs are static
        bias_s_batch = self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size())
        i = self.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(
                x_d[t], self.weight_ih
            )
            f, o, g = gates.chunk(3, 1)

            c_1 = self.sigmoid(f) * c_0 + i * self.tanh(g)
            h_1 = self.sigmoid(o) * self.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n
