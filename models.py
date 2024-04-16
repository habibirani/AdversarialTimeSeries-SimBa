import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


modelA_desc = "CNN"
# Define the CNN model
class CNN1D(nn.Module):
    def __init__(self, n_features=3, WINDOW_SIZE=35, n_outputs=1, num_filters=64, kernel_size=3, pool_size=2, dropout_rate=0.5, fc1_output_size=128):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(n_features, num_filters,
                               kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters,
                               kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Dynamically compute the size of the input to the first fully connected layer
        self.fc1_input_size = self._get_conv_output(WINDOW_SIZE)
        self.fc1 = nn.Linear(self.fc1_input_size, fc1_output_size)
        self.fc2 = nn.Linear(fc1_output_size, n_outputs)

    def _get_conv_output(self, size):
        with torch.no_grad():
            # Dummy input to calculate convolution output size
            dummy_input = torch.zeros(1, n_features, size)
            output = self.conv1(dummy_input)
            output = self.conv2(output)
            output = self.pool(output)
            return np.prod(output.size()[1:])  # total number of features

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.dropout(self.pool(x))
        x = x.view(-1, self.fc1_input_size)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


modelB_desc = "LSTM"
class LSTMModel(nn.Module):
    def __init__(self, n_features=3, hidden_dim=128, n_layers=2, n_outputs=1, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # LSTM Layer
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True, dropout=dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        # No need to manually adjust for seq_len here, as x is expected to have the shape [batch_size, seq_len, n_features]
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Taking the output of the last time step

        return out


modelC_desc = "Transformers"
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = self.positional_encoding(max_len=96, d_model=hidden_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, 6)  # Output size corresponds to the number of classes

    def positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc = torch.zeros((max_len, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:x.size(1), :]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.fc(x[:, -1, :])  # Use only the output of the last time step for classification
        return x

     