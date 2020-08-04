'''
in .cha
start with *PAR are the interviewee's transcripts
'''
# coding: utf-8
import pandas as pd
import numpy as np
import torch
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow.contrib.keras as kr
from modeling import adCNN,batch_iter
from torch import nn
from torch import optim
from torch.autograd import Variable
import re
import os

NUM_CLASS=14
data_dir='AD_data'
# train_df = pd.read_csv('train_sample.csv', sep='\t',nrows=100)
train_df = pd.read_csv(os.path.join(data_dir,'train_set.csv'),nrows=100)
# lengthes=[len(train_df['transcript'][i]) for i in range(len(train_df))]
# print(min(lengthes),max(lengthes)) #13 12976

train_y=train_df['label']
train_x=train_df['transcript']
# train_x=[train_df['transcript'][i].split(' ') for i in range(len(train_df))]
print(len(train_x))
print(len(train_x[0]))
maxlen=1000
print('maxlen:',maxlen)
train_x_consis=[]
train_y_consis=[]
for i in range(len(train_x)):
    # print('i',i,len(train_x[i])-499)
    for j in range(0,len(train_x[i]),maxlen):
        end=min(len(train_x[i]),j+maxlen)
        print(j,end)
        train_x_consis.append(train_x[i][j:end])
        train_y_consis.append(train_y[i])
# print(len(train_x_consis[0]),len(train_x_consis[1]))
# print(len(train_y_consis))
# exit()
# data_id, label_id = [], []
# for i in range(len(train_x_consis)):
    # data_id.append([word_to_id[x] for x in train_x_consis[i] if x in word_to_id])
    # label_id.append(cat_to_id[train_y_consis[i]])
train_x = kr.preprocessing.sequence.pad_sequences(train_x_consis, maxlen)
train_y = kr.utils.to_categorical(train_y_consis, num_classes=NUM_CLASS) 
print(len(cat_to_id))
exit()

train_x,val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2, random_state = 0)

# f1_score(y_true, y_pred, average='macro')
def train(x_batch,y_batch):
    x = np.array(x_batch)
    y = np.array(y_batch)
    x = torch.LongTensor(x)
    y = torch.Tensor(y)
    # y = torch.LongTensor(y)
    x = Variable(x)
    y = Variable(y)
    out = model(x)
    loss = Loss(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy = np.mean((torch.argmax(out,1)==torch.argmax(y,1)).numpy())
    return accuracy

if __name__=='__main__':
    model = adCNN(maxlen-4)
    Loss = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    best_val_acc = 0
    for epoch in range(11):
        # if epoch%200==0:
        print('epoch:',epoch)
        batch_train = batch_iter(train_x, train_y,16)
        for x_batch, y_batch in batch_train:
            print('train batch')
            accuracy=train(x_batch,y_batch)
        #validate
        if (epoch)%10 == 0:
            print('validation')
            batch_val = batch_iter(val_x, val_y,16)
            # for x_batch, y_batch in batch_train:
            print('val batch')
            for x_batch, y_batch in batch_val:
                accuracy=train(x_batch,y_batch)
                if accuracy > best_val_acc:
                    torch.save(model.state_dict(),'model_params.pkl')
                    best_val_acc = accuracy
                print('accuracy',accuracy)

