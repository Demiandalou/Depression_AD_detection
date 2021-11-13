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
from modeling import adCNN,batch_iter,process_words
from torch import nn
from torch import optim
from torch.autograd import Variable
import re
import os
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=32


data_dir='AD_data'
# train_df = pd.read_csv('train_sample.csv', sep='\t',nrows=100)
train_df = pd.read_csv(os.path.join(data_dir,'train.csv'))
# lengthes=[len(train_df['transcript'][i]) for i in range(len(train_df))]
# print(min(lengthes),max(lengthes)) #13 12976
# print('avg length:',np.mean(lengthes))
# exit()
train_y=train_df['label']
train_x=[train_df['transcript'][i].strip(' ').split(' ') for i in range(len(train_df))]
for i in range(len(train_x)):
    for j in range(len(train_x[i])):
        elem=train_x[i][j]
        if int(elem)>=10000:
            train_x[i][j]=str(int(elem)-1000)
        if int(elem)>=11000:
            print('aaa')
            exit()

print(len(train_x))
print(len(train_x[0]))
maxlen=500
print('maxlen:',maxlen)

train_x,train_y,val_x,val_y=process_words(train_x,train_y,maxlen)
# print(train_x[0],train_y[0])
# print(val_x[0],val_y[0])

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
    optimizer = optim.Adam(model.parameters(),lr=0.5)
    best_val_acc = 0
    for epoch in range(11):
        # if epoch%200==0:
        print('epoch:',epoch)
        batch_train = batch_iter(train_x, train_y,TRAIN_BATCH_SIZE)
        print('training')
        for x_batch, y_batch in batch_train:
            accuracy=train(x_batch,y_batch)
        #validate
        if (epoch)%5 == 0:
            print('validating')
            batch_val = batch_iter(val_x, val_y,VAL_BATCH_SIZE)
            # for x_batch, y_batch in batch_train:
            right=0
            for x_batch, y_batch in batch_val:
                accuracy=train(x_batch,y_batch)
                right+=len(x_batch)*accuracy
            accuracy=right/len(val_x)
            if accuracy > best_val_acc:
                torch.save(model.state_dict(),'model_params.pkl')
                best_val_acc = accuracy
            print('accuracy',accuracy)

