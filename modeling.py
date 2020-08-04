# coding: utf-8
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


# def preprocess(train_set):
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

# def process_words():


class adCNN(nn.Module):
  def __init__(self,cur_shape):
    super(adCNN, self).__init__()
    self.embedding = nn.Embedding(10000,64)
    self.conv = nn.Conv1d(64,256,5)
    self.cur_shape=cur_shape
    self.f1 = nn.Sequential(nn.Linear(256*cur_shape,128),
                nn.ReLU())
    self.f2 = nn.Sequential(nn.Linear(128, 14),
                nn.Softmax())
  def forward(self, x):
    x = self.embedding(x)
    x = x.detach().numpy()
    x = np.transpose(x,[0,2,1])
    x = torch.Tensor(x)
    x = Variable(x)
    x = self.conv(x)
    # x = x.view(-1,256*596)
    x = x.view(-1, 256*self.cur_shape)
    # x = x.view(2563584,1)
    print(x.shape)
    # x = x.view(x.size(0), -1)
    x = self.f1(x)
    return self.f2(x)

