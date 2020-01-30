# import BiDirectionalLSTM as biLstm
import simpleLSTM
import pdb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pickle
import LoadMiceTimeSeriesData as loadMice

# ----------------------------Loading data--------------------------------------
path = '/data3/darpa/all_CCD_processed_data/tamu_expts_01-28.h5'
loader = loadMice.MiceLoader()
loader.load_ccd(path)

lineName = ['CC001']
segment_interval = 120
embedding = [3, 30]  # time delayed embedding: [embedding dimension, delay]
miceDict = loader.get_cc_data_in_dict('CC001', segment_interval, embedding)
# ----------------------------Prepare mice Data --------------------------------


def create_inout_seq(mice_dict):
    input_seq = []
    for i in mice_dict.keys():
        for j in range(len(mice_dict[i][0])):
            train_in, train_out = torch.from_numpy(np.array(mice_dict[i][0][j])).float(), \
                                  torch.from_numpy(np.array(mice_dict[i][1][j])).float()
            input_seq.append((train_in, train_out))
    return input_seq


def create_inout_seq_single(mice_dict, mice_name, interval):
    mice = mice_dict[mice_name]
    out_seq = []
    for i in range(len(mice[0])//interval):
        train_in, train_out = torch.from_numpy(np.array(mice[0][i*interval])), \
                              torch.from_numpy(np.array(mice[1][i*interval]))
        out_seq.append((train_in, train_out))
    return out_seq


# inout_seq = create_inout_seq(miceDict)
name = list(miceDict.keys())[0]
print('pick mice: %s' % name)
inout_seq = create_inout_seq_single(miceDict, mice_name=name, interval=segment_interval)
train_labels = [list(inout[1]) for inout in inout_seq]
first_sick_index = None
for i in range(len(train_labels)-1):
    if train_labels[i] != train_labels[i+1]:
        first_sick_index = i+1
        break
print('training number is %d, first positive index is %d' % (len(inout_seq), first_sick_index))

train_size = 10
healthy_start_idx = 4
sick_start_idx = first_sick_index+20
healthy_train = inout_seq[healthy_start_idx:healthy_start_idx+train_size]
sick_train = inout_seq[sick_start_idx:sick_start_idx+train_size]
inout_seq = healthy_train + sick_train

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
for i in range(train_size):
    ax1.plot(healthy_train[i][0][:, 0], label="negative_%d" % i)
    ax2.plot(sick_train[i][0][:, 0], label="positive_%d" % i)
ax1.legend()
ax2.legend()
ax1.set_xlabel('time')
ax1.set_ylabel('standardized temp')
ax2.set_xlabel('time')
ax2.set_ylabel('standardized temp')
plt.show()


batch_size = train_size*2
batch_num = 1
train_num = batch_size * batch_num
test_num = 1


def create_batch_data(inout_seq, batch_number, batch_volume):
    inout_seq_permute = inout_seq
    batch_data = []
    batch_label = []
    for i in np.arange(batch_number):
        pair_i = inout_seq_permute[i*batch_size:(i+1)*batch_volume]
        batch_i, label_i = [kk[0].tolist() for kk in pair_i], [ll[1].tolist() for ll in pair_i]
        # TODO: resize batch data for LSTM input
        batch_data.append(batch_i)
        batch_label.append(label_i)
    return torch.tensor(batch_data).float(), torch.tensor(batch_label).float()


batch_train, batch_train_label = create_batch_data(inout_seq, batch_num, batch_size)

# ----------------------------Run LSTM Model -----------------------------------
hidden_layer_size = 256
layers = 3
lstm_model = simpleLSTM.SimpleLSTM(batch_size, input_size=embedding[0], hidden_size=hidden_layer_size, output_size=2
                                   , num_layers=layers)
lstm_model = lstm_model.float()
loss_function = nn.BCELoss()
print(lstm_model.parameters)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
epoch = 1000

for i in range(epoch):
    batch_err = 0
    for j in np.arange(batch_num):
        seq, label = batch_train[j], batch_train_label[j]
        optimizer.zero_grad()
        lstm_model.hidden_cell = (torch.zeros(layers, batch_size, hidden_layer_size),
                                  torch.zeros(layers, batch_size, hidden_layer_size))
        ypred = lstm_model(seq)
        single_loss = loss_function(ypred, label)
        single_loss.backward()
        optimizer.step()
        print('%5d th batch ---- error %8.6f' % (j, single_loss.item()))
        batch_err = batch_err + single_loss.item()
    #print('epoch:%5d, batch_error: %8.6f' % (i, batch_err))
print(ypred)

"""
pred_label = []
true_label = []
for k in range(train_num, test_num+train_num):
    test_seq, truth_label = inout_seq[k]
    lstm_model.hidden_cell = (torch.zeros(2, 1, hidden_layer_size),
                              torch.zeros(2, 1, hidden_layer_size))
    lstm_model.batch_size = 1
    pred_label.append(lstm_model(test_seq.reshape(1, segment_interval, 1)).detach().numpy())
    true_label.append(truth_label.detach().numpy())

with open('pred_label.txt', 'wb') as fp:
    pickle.dump(pred_label, fp)

with open('true_label.txt', 'wb') as fp:
    pickle.dump(true_label, fp)

with open('pred_label.txt', 'rb') as fp:
    predicted = pickle.load(fp)
print(predicted)
for pred in predicted:
    if pred[0][1].item() > 0.5:
        print(pred[0][1])
print(true_label)
true_positive = 0
true_negative = 0
num_positive = 0
num_negative = 0
for i in range(test_num):
    pred_label = [1, 0] if predicted[i][0][0] > predicted[i][0][1] else [0, 1]
    if abs(true_label[i][0].item()-1) <= 0.001:
        num_negative += 1
        if pred_label[0] == 1:
            true_negative += 1
    else:
        num_positive += 1
        if pred_label[1] == 1:
            true_positive += 1
print('True Negative:%.3f,# of negative %d; True Positive:%.3f,# of negative %d' %
      (true_negative/num_negative, num_negative, true_positive/num_positive, num_positive))
"""
