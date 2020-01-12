import torch
import torch.nn as nn
import torch.nn.functional as F
import BiDirectionalLSTM  as biLstm

input = torch.randn(5,1,1)
model = biLstm.BiLSTM(len(input))
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

print(len(input))
print(model)

hidden_layer_size = 1
hidden_vec = (torch.zeros(2, 1, hidden_layer_size),
              torch.zeros(2, 1, hidden_layer_size))
print(input[:,0,0])
model.hidden_cell = hidden_vec
output = model(input)

print(output)


