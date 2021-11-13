import torch
from torch import nn, optim

import nltk
# nltk.download('punkt')

from torch.utils.data import DataLoader
import audioLoader
import videoLoader
import transcriptLoader
import metaInfoLoader
import models
from models import CFNN
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == '__main__':
    print("start running")
 
    #read features from daicwoz dataset into lists
    daic_train, daic_test = prepare_sample()       
    print('train length:', len(daic_train), 'test length:', len(daic_test))

    #make datasets with given batch size
    batchsz_train = 8
    batchsz_test = 5    
    daic_train = DataLoader(daic_train, batch_size=batchsz_train, shuffle=True, drop_last=True)
    daic_test = DataLoader(daic_test, batch_size=batchsz_test, shuffle=True, drop_last=True)
    V, A, L, label = iter(daic_train).next()

    