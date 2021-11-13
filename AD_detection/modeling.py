# coding: utf-8
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import tensorflow.contrib.keras as kr
from sklearn.model_selection import train_test_split
import os
NUM_CLASS=2
def batch_iter(x, y, batch_size=64):
  data_len = len(x)
  num_batch = int((data_len - 1) / batch_size) + 1
  
  indices = np.random.permutation(np.arange(data_len))
  x_shuffle = x[indices]
  y_shuffle = y[indices]
  
  for i in range(num_batch):
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, data_len)
    yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def process_words(train_x,train_y,maxlen):
  # https://zhuanlan.zhihu.com/p/53277723
  # https://blog.csdn.net/qq_28840013/article/details/89681499
  # https://blog.csdn.net/huacha__/article/details/84068653
  train_x_consis=[]
  train_y_consis=[]
  for i in range(len(train_x)):
      # print('i',i,len(train_x[i])-499)
      for j in range(0,len(train_x[i]),maxlen):
          end=min(len(train_x[i]),j+maxlen)
          # print(j,end)
          train_x_consis.append(train_x[i][j:end])
          train_y_consis.append(train_y[i])

  train_x = kr.preprocessing.sequence.pad_sequences(train_x_consis, maxlen)
  train_y = kr.utils.to_categorical(train_y_consis, num_classes=NUM_CLASS) 

  train_x,val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2, random_state = 0)

  return train_x,train_y,val_x,val_y

class adCNN(nn.Module):
  def __init__(self,cur_shape):
    super(adCNN, self).__init__()
    self.embedding = nn.Embedding(10000,64)
    self.conv = nn.Conv1d(64,256,5)
    self.cur_shape=cur_shape
    self.f1 = nn.Sequential(nn.Linear(256*cur_shape,128),
                nn.ReLU())
    self.f2 = nn.Sequential(nn.Linear(128, NUM_CLASS),
                nn.Softmax())
  def forward(self, x):
    x = self.embedding(x)
    x = x.detach().numpy()
    x = np.transpose(x,[0,2,1])
    x = torch.Tensor(x)
    x = Variable(x)
    x = self.conv(x)
    x = x.view(-1, 256*self.cur_shape)
    # print(x.shape)
    x = self.f1(x)
    return self.f2(x)

