import numpy as np
import torch


torch.set_default_dtype(torch.float32)
import torch.nn as nn
import math
from torch.nn import Parameter
import torch.nn.functional as F


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class MV_LSTM(nn.Module):
    def __init__(self, n_features, seq_length, nhidden, n_layers, dropout):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = nhidden  # number of hidden states in each LSTM layer
        self.n_layers = n_layers  # number of LSTM layers (stacked)
        self.dropout = torch.nn.Dropout(dropout)

        self.l_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True,
        )

        self.l_linear1 = nn.Linear(
            self.n_hidden * self.seq_len, self.n_hidden * self.seq_len
        )

        self.l_linear2 = nn.Linear(self.n_hidden * self.seq_len, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden = (hidden_state, cell_state)
        lstm_out, self.hidden = self.l_lstm(x, hidden)

        x = lstm_out.contiguous().view(batch_size, -1)
        x = self.dropout(x)
        x = self.l_linear1(x)
        x = self.dropout(x)
        x = self.l_linear2(x)
        return x.squeeze()


class Hiddenstate_LSTM(nn.Module):
    def __init__(self, n_features, use_static, nhidden, n_layers, dropout):
        super(Hiddenstate_LSTM, self).__init__()
        self.n_features = n_features
        # self.seq_len = seq_length
        if use_static:
            self.static_dim = 2
        else:
            self.static_dim = 0
        self.n_hidden = nhidden  # number of hidden states in each LSTM layer
        self.n_layers = n_layers  # number of LSTM layers (stacked)
        self.dropout = nn.Dropout(p=dropout)
        self.l_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True,
        )

        self.fcl = nn.Linear(self.n_hidden + self.static_dim, 1)

    def forward(self, x, static_data):
        lstm_out, (h_n, c_n) = self.l_lstm(x)

        final_hidden_state = h_n[-1]
        final_hidden_state = self.dropout(final_hidden_state)
        if self.static_dim != 0:
            final_hidden_state = torch.cat((final_hidden_state, static_data), dim=1)
        logits = self.fcl(final_hidden_state)  # .squeeze()
        return logits


class onedCNN(nn.Module):
    def __init__(self, n_features, seq_len, use_static, kernel_size, fcl_size, dropout):
        super(onedCNN, self).__init__()
        self.use_static = use_static
        # Define your convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=n_features, out_channels=2 * n_features, kernel_size=kernel_size
        )
        # (input_size - kernel_size + 2*padding)/stride + 1
        output_size = int(((seq_len - kernel_size + 2 * 0) / 1 + 1) // 1)

        # max_pool layer = (input_size - kernel_size)/stride + 1
        output_size = int(((output_size - 2) / 2 + 1) // 1)

        self.conv2 = nn.Conv1d(
            in_channels=2 * n_features,
            out_channels=4 * n_features,
            kernel_size=kernel_size,
            padding=1,
        )
        output_size = int(((output_size - kernel_size + 2 * 1) / 1 + 1) // 1)

        # after maxpool
        output_size = int(((output_size - 2) / 1 + 1) // 1)

        self.conv3 = nn.Conv1d(
            in_channels=4 * n_features,
            out_channels=8 * n_features,
            kernel_size=kernel_size,
            padding=1,
        )

        output_size = int(((output_size - kernel_size + 2 * 1) / 1 + 1) // 1)

        # after max pool
        output_size = int(((output_size - 2) / 1 + 1) // 1)

        fclayer_input = output_size * 8 * n_features
        if self.use_static:
            fclayer_input += 2
        self.fc1 = nn.Linear(fclayer_input, fcl_size)
        self.fc2 = nn.Linear(fcl_size, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, static):
        # Input has shape (batch_size, channels, height, width)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=1)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=1)
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # x.size(0) is the batch size
        if self.use_static:
            x = torch.cat((x, static))
        x = self.dropout(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()


class MV_GRU(nn.Module):
    def __init__(self, n_features, nhidden, n_layers, dropout, use_static):
        super(MV_GRU, self).__init__()
        self.n_features = n_features
        self.n_hidden = nhidden  # number of hidden states in each GRU layer
        self.n_layers = n_layers  # number of GRU layers (stacked)
        self.use_static = use_static
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True,
        )
        fcl_size = self.n_hidden
        if use_static:
            fcl_size += 2

        self.fcl = nn.Linear(self.n_hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, static):

        output, h_n = self.gru(x)

        final_hidden_state = h_n[-1]
        x = torch.cat((final_hidden_state, static), dim=1)
        x = self.dropout(x)
        logits = self.fcl(final_hidden_state)
        return logits


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        super(FilterLinear, self).__init__()
        """
        This filters which weights are updated.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.filter_square_matrix = filter_square_matrix.clone().detach()

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + "out_features="
            + str(self.out_features)
            + ", bias="
            + str(self.bias is not None)
            + ")"
        )


class GRUD(nn.Module):
    def __init__(
        self,
        input_size,
        nhidden,
        X_mean,
        output_last=True,
        dropout=0.5,
        use_static=False,
    ):
        super(GRUD, self).__init__()

        self.use_dems = use_static
        self.hidden_size = nhidden
        self.delta_size = input_size
        self.mask_size = input_size

        if use_static:
            self.static_dim = 2
        else:
            self.static_dim = 0

        self.identity = torch.eye(input_size)
        self.zeros = torch.zeros(input_size)
        self.X_mean = torch.Tensor(X_mean)

        self.zl = nn.Linear(
            input_size + self.hidden_size + self.mask_size, self.hidden_size
        )  # Update gate
        self.rl = nn.Linear(
            input_size + self.hidden_size + self.mask_size, self.hidden_size
        )  # Reset gate
        self.hl = nn.Linear(
            input_size + self.hidden_size + self.mask_size, self.hidden_size
        )  # Hidden state
        self.gamma_x_l = FilterLinear(
            self.delta_size, self.delta_size, self.identity
        )  # Input decay term
        self.gamma_h_l = nn.Linear(
            self.delta_size, self.hidden_size
        )  # Hidden state decay term
        self.binary_final_fcl = nn.Linear(self.hidden_size + self.static_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        self.output_last = output_last

    def step(self, x, x_last_obsv, x_mean, h, mask, delta_x, delta_h):

        ##delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
        #delta_x = torch.exp(-F.relu(self.gamma_x_l(delta)))
        ###delta_h = torch.exp(
        ##    -torch.max(torch.zeros(self.hidden_size), self.gamma_h_l(delta))
        ##)
        #delta_h = torch.exp(-F.relu(self.gamma_h_l(delta)))
        x = torch.nan_to_num(x, nan=-1.0)
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)

        h = delta_h * h

        combined = torch.cat((x, h, mask), 1).to(torch.float32)

        z = F.sigmoid(self.zl(combined))
        r = F.sigmoid(self.rl(combined))

        combined_r = torch.cat((x, r * h, mask), 1).to(torch.float32)  # reset_gate

        h_tilde = F.tanh(self.hl(combined_r))  # Candidate hidden state

        h = (1 - z) * h + z * h_tilde  # Update Hidden State

        h = self.dropout(h)
        return h

    def forward(self, input, static_data):

        batch_size = input.size(0)
        # type_size = input.size(1)  # types of data, e.g. deltas, mask
        step_size = input.size(2)
        # spatial_size = input.size(3)

        Hidden_State = self.initHidden(batch_size)

        X = input[:, 0, :, :]  # current observation

        X_last_obsv = input[:, 1, :, :]

        Mask = input[:, 2, :, :]

        Delta = input[:, 3, :, :]

        # Compute deltas outside loop
        delta_x_precomputed =  torch.exp(-F.relu(self.gamma_x_l(Delta))) # for all timesteps
        delta_h_precomputed = torch.exp(-F.relu(self.gamma_h_l(Delta))) # for all timesteps
        outputs = torch.zeros(batch_size, step_size, Hidden_State.shape[1])

        for i in range(step_size):
            Hidden_State = self.step(
                X[:,i,:],#torch.squeeze(X[:, i : i + 1, :], dim=1),
                X_last_obsv[:,i,:], #torch.squeeze(X_last_obsv[:, i : i + 1, :], dim=1),
                self.X_mean[:,i,:],#torch.squeeze(self.X_mean[:, i : i + 1, :], dim=1),
                Hidden_State,
                Mask[:,i,:],#torch.squeeze(Mask[:, i : i + 1, :], dim=1),
                #Delta[:,i,:]#torch.squeeze(Delta[:, i : i + 1, :], dim=1),
                delta_x_precomputed[:,i,:],
                delta_h_precomputed[:,i,:]
            )
            #if outputs is None:
            #    outputs = Hidden_State.unsqueeze(1)
            #else:
            #    outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            outputs[:, i, :] = Hidden_State
        input_to_class_layer = outputs[:, -1, :]
        if self.use_dems == True:
            input_to_class_layer = torch.cat((input_to_class_layer, static_data), dim=1)
            out = self.binary_final_fcl(input_to_class_layer)
            return out  # return logits
        else:
            out = self.binary_final_fcl(outputs[:, -1, :])
            return out

    def initHidden(self, batch_size):

        Hidden_State = torch.zeros(batch_size, self.hidden_size)
        return Hidden_State


class BloodTestTransformer(nn.Module):
    def __init__(self, num_features,d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1):
        
        super().__init__()

        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)

        return x