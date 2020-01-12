import torch
import torch.nn as nn
#import torch.nn.functional as F
import pdb
import numpy as np


class BiLSTM(nn.Module):
    def __init__(self, batch_size,input_size=1, hidden_size=64, output_size=2):
        super().__init__()
        # parameter
        self.hidden_layer_size = hidden_size
        self.batch_size = batch_size
        # functionals
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_layer_size, num_layers=1, batch_first=True, dropout=0.5,bidirectional=True)
        self.linear1 = nn.Linear(in_features=2 * hidden_size, out_features=self.hidden_layer_size, bias=True)
        self.uni_lstm = nn.LSTM(input_size=hidden_size, hidden_size=self.hidden_layer_size, num_layers=1, dropout=0.5,batch_first=True)
        self.hidden_cell = (torch.zeros(2, batch_size, self.hidden_layer_size),
                            torch.zeros(2, batch_size, self.hidden_layer_size))
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size, bias= True)
        self.hidden_cell2 = (torch.zeros(1, batch_size, self.hidden_layer_size),
                             torch.zeros(1, batch_size, self.hidden_layer_size))
        #self.linear = nn.Linear(1*hidden_size * seq_len, output_size, bias=True)

    def forward(self, input_seq):
        bi_lstm_out, self.hidden_cell = self.bi_lstm(input_seq, self.hidden_cell)
        bi_lstm_out = self.linear1(bi_lstm_out)
        uni_lstm_out,self.hidden_cell2 = self.uni_lstm(bi_lstm_out,self.hidden_cell2)
        uni_lstm_out_last = uni_lstm_out[:,-1,:]
        last_linear = self.linear2(uni_lstm_out_last)
        #pdb.set_trace()
        output = torch.nn.functional.softmax(last_linear,dim=1)
        return output



