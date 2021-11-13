# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:20:47 2020

@author: long
"""

import torch
from torch import nn
import torch.nn.init as init 

class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
                                      
        super(CausalConv1d, self).__init__()
        
        # attributes:
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size-1)*dilation
        
        # modules:
        self.conv1d = nn.utils.weight_norm(torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=stride,
                                      padding=(kernel_size-1)*dilation,
                                      dilation=dilation), name = 'weight')

    def forward(self, seq):

        conv1d_out = self.conv1d(seq)
        # remove k-1 values from the end:
        return conv1d_out[:,:,:-(self.padding)]

class Single_CCNN(nn.Module):

    def __init__(self,USE_GLOVE,pretrained_emb, token_size):
        super(Single_CCNN, self).__init__()
        
        
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=300
        )
        if USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            #self.embedding.weight.requires_grad = False
        self.conv_unit = nn.Sequential(
            CausalConv1d(300 ,128, kernel_size=5, stride=1, dilation=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=8),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=16),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=32),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=128),
            # nn.ReLU()
        )
        
        
        # flatten
        # fully connected (fc) unit
        self.fc_unit = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def init_parameter(self):
        for name,para in self.named_parameters():
            if "conv1d" in name and "weight" in name:
                init.kaiming_normal_(para)
            if "fc_unit" in name and "weight" in name:
                init.xavier_normal_(para)


    def forward(self, inputs):
        # input (b,len,dim)

        inputs = self.embedding(inputs)

        inputs = torch.transpose(inputs,1,2)  #for 1dCNN input 
        out = self.conv_unit(inputs)  # (b,128,len)
        out = torch.transpose(out, 1, 2) #(b,len,128)
        length = out.shape[1]
        #embedding_v = out[: ,length//2, :]  #get center vector
        embedding_v = out[:,-1,:]
        
       
        
        logits = self.fc_unit(embedding_v)

        return logits


## multi_layer LSTM

if __name__ == "__main__":
    input_t = torch.randn(16,1000,300)
    print(input_t.shape)
    net = CFNN()
    out = net(input_t)
    init.constant_(net[0].weight,0)
    print(out.shape)