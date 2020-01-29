import torch
import torch.nn as nn
#import torch.nn.functional as F
import pdb
import numpy as np


class testLSTM(nn.Module):
    def __init__(self, batch_size, input_size=1, hidden_size=64, output_size=2):
        super().__init__()
        # parameter
        self.hidden_layer_size = hidden_size
        self.batch_size = batch_size
        # functionals
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_layer_size, num_layers=2, batch_first=True, dropout=0)
        self.linear1 = nn.Linear(in_features= hidden_size, out_features=self.hidden_layer_size, bias=True)
        self.uni_lstm = nn.LSTM(input_size=hidden_size, hidden_size=self.hidden_layer_size, num_layers=1, batch_first=True, dropout=0)
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size),
                            torch.zeros(1, batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        bi_lstm_out, self.hidden_cell = self.bi_lstm(input_seq, self.hidden_cell)
        bi_lstm_out = self.linear1(bi_lstm_out)
        uni_lstm_out,self.hidden_cell2 = self.uni_lstm(bi_lstm_out,self.hidden_cell2)
        uni_lstm_out_last = uni_lstm_out[:,-1,:]
        last_linear = self.linear2(uni_lstm_out_last)
        #pdb.set_trace()
        return last_linear