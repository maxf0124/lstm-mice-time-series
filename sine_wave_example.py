import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb
import testLSTM as mylstm

def generate_sine_data(a = 1, T= 400*np.pi, seq_len = 10000):
    t = np.linspace(0,T,seq_len)
    sine_curve = np.sin(a*t)
    return sine_curve

data = generate_sine_data()

def delayed_embedding(x, delay=25, dim=3):
    embed_data = []
    target = []
    for i in range((len(x) - dim * delay)-1):
        idx = i + np.arange(dim) * delay
        embed_i = x[idx]
        embed_data.append(embed_i)
        target.append(x[idx[-1]+1])
    return np.array(embed_data),np.array(target)

embed_data, target = delayed_embedding(data)
#pdb.set_trace()


def create_seq_data(in_seq,target,seq_len):
    #train_num = batch_num*batch_size
    #rand_idx = np.random.permutation(train_num)
    #inout_seq_permute = [inout_seq[i] for i in rand_idx]
    inout_seq_permute = in_seq
    batch_data = []
    batch_label = []
    for i in range(in_seq.shape[0]-seq_len-1):
        batch_i = inout_seq_permute[i:i+seq_len]
        label_i = target[i+seq_len+1]
        batch_data.append(batch_i)
        batch_label.append(label_i)
    return np.array(batch_data),np.array(batch_label)


#batch_num = 180
seq_len = 30
seq_X, seq_y = create_seq_data(in_seq=embed_data,target=target,seq_len=seq_len)
#pdb.set_trace()
batch_size=50
def generate_batch_data(x,y,batch_size):
    n = len(x)
    batch_num = n//batch_size
    idx = np.random.permutation(n)
    batch_x,batch_y = [],[]
    for i in range(batch_num):
        batch_i = x[idx[i*batch_size:(i+1)*batch_size],:,:]
        label_i = y[idx[i*batch_size:(i+1)*batch_size]]
        batch_x.append(batch_i)
        batch_y.append(label_i)
    return torch.tensor(batch_x,requires_grad=False).float(), torch.tensor(batch_y,requires_grad=False).float()

batch_x,batch_y = generate_batch_data(seq_X,seq_y,batch_size=batch_size)



hidden_layer_size = 64
lstm_model = mylstm.testLSTM(batch_size,input_size=3,hidden_size=hidden_layer_size,output_size=1)
lstm_model = lstm_model.float()
loss_function = nn.MSELoss()
print(lstm_model.parameters)
optimizer = torch.optim.Adam(lstm_model.parameters(),lr=0.001)
epoch = 100



for i in range(epoch):
    batch_err = 0
    for j in range(batch_x.shape[0]):
        seq, label = batch_x[j],batch_y[j]
        #seq = seq.detach()
        #label = label.detach()
        optimizer.zero_grad()
        lstm_model.hidden_cell=(torch.zeros(1, batch_size, hidden_layer_size),
                                torch.zeros(1, batch_size, hidden_layer_size))
        lstm_model.hidden_cell2 = (torch.zeros(1, batch_size, hidden_layer_size),
                                   torch.zeros(1, batch_size, hidden_layer_size))
        ypred = lstm_model(seq)

        #hidden.detach_()
        #pdb.set_trace()
        single_loss = loss_function(ypred.view((len(ypred),-1)),label.view(len(ypred),-1))
        single_loss.backward()
        optimizer.step()
        print('%5d th batch: error: %5.8f'%(j,single_loss.item()))
        batch_err = batch_err + single_loss.item()
    print(ypred)

    print('epoch:%5d, batch_error: %8.5f'%(i,batch_err))


