# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:57:14 2020

@author: long
"""


import numpy as np
from numpy import genfromtxt
import torch
#import models
import wordEmbedding
import os
from gensim.models import Word2Vec,KeyedVectors
import pprint
import gensim
from glove import Glove
from glove import Corpus
import tensorflow as tf

def process_transcript(session):
    session=str(session)
    if not os.path.isfile('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT.pt'):        
        transcript = genfromtxt('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_TRANSCRIPT.csv', usecols = (2,3), encoding = "UTF-8", dtype = str, delimiter = '\t', skip_header = 1)
        patient_indice = np.where(transcript == 'Participant' or transcript == 'Ellie')
        # patient_indice = np.where(transcript == 'Participant')
        transcript = transcript[patient_indice[0],1]
        #use fastText to do word embedding
        embeddings = wordEmbedding.embed(transcript)
        embeddings = embeddings.double()
        torch.save(embeddings, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT.pt') 
    else:
        embeddings = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT.pt')
        # print('embedding',embeddings)
        # exit()
    
    return embeddings
    
def w2v_process_transp(session):
    session=str(session)
    transcript = genfromtxt('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_TRANSCRIPT.csv', usecols = (2,3), encoding = "UTF-8", dtype = str, delimiter = '\t', skip_header = 1)
    txt_list=[]
    for t in transcript:
        tmplist=[]
        tmplist.append(t[0])
        tmplist+=t[1].split(' ')
        txt_list.append(tmplist)
    model= Word2Vec(txt_list,min_count=1,size=300,workers=4)
    
    model.wv.save_word2vec_format('word2vec.bin')
    model = KeyedVectors.load_word2vec_format('word2vec.bin')
    # weights = torch.FloatTensor(model.vectors)
    weights = torch.DoubleTensor(model.vectors)
    # embedding = torch.nn.Embedding.from_pretrained(weights)        
    # embedding(input)
    return weights

def glove_vec(session):
    session=str(session)
    transcript = genfromtxt('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_TRANSCRIPT.csv', usecols = (2,3), encoding = "UTF-8", dtype = str, delimiter = '\t', skip_header = 1)
    txt_list=[]
    for t in transcript:
        tmplist=[]
        tmplist.append(t[0])
        tmplist+=t[1].split(' ')
        txt_list.append(tmplist)
    corpus_model = Corpus()
    corpus_model.fit(txt_list, window=10)
    #corpus_model.save('corpus.model')
    # print('Dict size: %s' % len(corpus_model.dictionary))
    # print('Collocations: %s' % corpus_model.matrix.nnz)
    glove = Glove(no_components=300, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=10,
          no_threads=1, verbose=False)
    glove.add_dictionary(corpus_model.dictionary)
    # print(type(glove.word_vectors))
    # weights = tf.convert_to_tensor(glove.word_vectors, dtype=tf.float32)
    weights = torch.from_numpy(glove.word_vectors)
    # print(type(weights))
    return weights

def main():
    # embeddings = process_transcript(301)
    embeddings = glove_vec(301)
    # embeddings = w2v_process_transp(301)
    print("embeddings: ", embeddings)
    print("embeddings shape: ", embeddings.shape)


if __name__ == '__main__':
    main()
