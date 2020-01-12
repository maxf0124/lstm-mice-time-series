#import calcom
import BiDirectionalLSTM as biLstm
import pdb
import torch
import torch.nn as nn
import numpy as np
import pickle
import LoadMiceTimeSeriesData as loadMice



# ----------------------------Loading data--------------------------------------
path ='/data3/darpa/all_CCD_processed_data/tamu_expts_01-28.h5'
loader = loadMice.MiceLoader()
loader.load_ccd(path)

lineName = ['CC001']
segment_interval = 100
embedding = [5,30]
miceDict = loader.get_ccDataInDict('CC001',segment_interval,embedding)
# ----------------------------Prepare mice Data --------------------------------


def create_inout_seq(miceDict):
    input_seq = []
    for i in miceDict.keys():
        for j in range(len(miceDict[i][0])):
            train_in,train_out = torch.from_numpy(np.array(miceDict[i][0][j])).float(), torch.from_numpy(np.array(miceDict[i][1][j])).float()
            input_seq.append((train_in,train_out))
    return input_seq


inout_seq = create_inout_seq(miceDict)
#pdb.set_trace()
batch_size = 10
batch_num = 80
train_num = batch_size * batch_num
test_num =130


def create_batch_data(inout_seq,test_num,batch_num,batch_size):
    train_num = batch_num*batch_size
    rand_idx = np.random.permutation(train_num)
    inout_seq_permute = [inout_seq[i] for i in rand_idx]
    #inout_seq_permute = inout_seq
    batch_data = []
    batch_label = []
    for i in range(batch_num):
        pair_i = inout_seq_permute[i*batch_size:(i+1)*batch_size]
        batch_i,label_i = [k[0].tolist() for k in pair_i],[ll[1].tolist() for ll in pair_i]
        # TODO: resize batch data for LSTM input
        batch_data.append(batch_i)
        batch_label.append(label_i)
    return torch.tensor(batch_data).float(),torch.tensor(batch_label).float()


batch_train,batch_train_label = create_batch_data(inout_seq,test_num,batch_num,batch_size)
pdb.set_trace()

# ----------------------------Run LSTM Model -----------------------------------
hidden_layer_size = 128
lstm_model = biLstm.BiLSTM(batch_size,input_size=embedding[0],hidden_size=hidden_layer_size,output_size=2)
lstm_model = lstm_model.float()
loss_function = nn.BCELoss()
print(lstm_model.parameters)
optimizer = torch.optim.Adam(lstm_model.parameters(),lr=0.001)
epoch = 1000

#train_num = 150
#test_num = 75

for i in range(epoch):
    batch_err = 0
    for j in range(batch_num):
        seq, label = batch_train[j],batch_train_label[j]
        #seq = seq.detach()
        #label = label.detach()
        optimizer.zero_grad()
        lstm_model.hidden_cell=(torch.zeros(2, batch_size, hidden_layer_size),
                                torch.zeros(2, batch_size, hidden_layer_size))
        lstm_model.hidden_cell2 = (torch.zeros(1, batch_size, hidden_layer_size),
                                   torch.zeros(1, batch_size, hidden_layer_size))
        ypred = lstm_model(seq)
        #hidden.detach_()
        #pdb.set_trace()
        single_loss = loss_function(ypred,label)
        single_loss.backward()
        optimizer.step()
        print('%5d th batch: error%5.3f'%(j,single_loss.item()))
        batch_err = batch_err + single_loss.item()
    print(ypred)

    print('epoch:%5d, batch_error:%8.5f'%(i,batch_err))

pred_label = []
true_label = []
for k in range(train_num,test_num+train_num):
    test_seq, truth_label = inout_seq[k]
    lstm_model.hidden_cell = (torch.zeros(2, 1, hidden_layer_size),
                                torch.zeros(2, 1, hidden_layer_size))
    lstm_model.batch_size = 1
    pred_label.append(lstm_model(test_seq.reshape(1,segment_interval,1)).detach().numpy())
    true_label.append(truth_label.detach().numpy())

with open('pred_label.txt','wb') as fp:
    pickle.dump(pred_label,fp)

with open('true_label.txt', 'wb') as fp:
    pickle.dump(true_label,fp)

with open('pred_label.txt','rb') as fp:
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
    pred_label = [1,0] if predicted[i][0][0]>predicted[i][0][1] else [0,1]
    if abs(true_label[i][0].item()-1) <= 0.001:
        num_negative += 1
        if pred_label[0] == 1:
            true_negative += 1
    else:
        num_positive += 1
        if pred_label[1]==1:
            true_positive+=1
print('True Negative:%.3f,# of negative %d; True Positve:%.3f,# of negative %d'%(true_negative/num_negative,num_negative,true_positive/num_positive,num_positive))