import torch
import torch.nn as nn
import pdb


class SimpleLSTM(nn.Module):
    def __init__(self, batch_size, input_size=1, hidden_size=64, output_size=2, num_layers=2):
        super().__init__()
        # parameter
        self.hidden_layer_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        # functions
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_layer_size, num_layers=self.num_layers,
                            batch_first=True, dropout=0.3)
        self.linear = nn.Linear(in_features=self.hidden_layer_size, out_features=output_size, bias=True)
        self.hidden_cell = (torch.zeros(self.num_layers, self.batch_size, self.hidden_layer_size),
                            torch.zeros(self.num_layers, self.batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        lstm_out_last = lstm_out[:, -1, :]
        last_linear = self.linear(lstm_out_last)
        output = torch.nn.functional.softmax(last_linear, dim=1)
        return output
