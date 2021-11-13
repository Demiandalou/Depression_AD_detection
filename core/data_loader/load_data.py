# --------------------------------------------------------
# Produce training dataset
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.data.dataset as Data  
from torch.utils.data import DataLoader,Subset
import numpy as np 
import sys
sys.path.append("..")
from data_utils import *


#[item,item_label,segment_question,segment_answer,segment_face,segment_voice]

class SplitedDataset(Data.Dataset):

    def __init__(self,train_or_test):
        self.train_or_test = train_or_test 
        train_index = np.load("/mnt/sdc1/daicwoz/data_pro/train_title2label.npy",allow_pickle=True)
        test_index = np.load("/mnt/sdc1/daicwoz/data_pro/test_title2label.npy",allow_pickle=True)
        train_index = list(train_index.keys())  #[indexs]
        test_index = list(test_index.keys())    #[indexs]
        
        
        
        if self.train_or_test:
            self.face_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_face.npy",allow_pickle=True)
            
            self.voice_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_voice.npy",allow_pickle=True)
            
        else:
            self.face_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_face_test.npy",allow_pickle=True)
            self.voice_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_voice_test.npy",allow_pickle=True)
            
        
        self.text_feat_train_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_0_4.npy",allow_pickle=True)
        self.text_feat_test_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_0_4_test.npy",allow_pickle=True)
        
        self.attri_feat_list = self.text_feat_train_list.tolist()+self.text_feat_test_list.tolist()
        self.attri_feat_list = np.array(self.attri_feat_list)
        self.attri_feat_list = np.concatenate((self.attri_feat_list[:,2],self.attri_feat_list[:,3]),axis=0)
        self.token_to_ix, self.pretrained_emb = tokenize(self.attri_feat_list, True)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        if self.train_or_test:
            self.data_text = self.text_feat_train_list[:,2:4]  # get question and answer
            self.ans_to_ix = self.text_feat_train_list[:,1]
        else:
            self.data_text = self.text_feat_test_list[:,2:4]   # get question and answer
            self.ans_to_ix = self.text_feat_test_list[:,1]

        

    def __getitem__(self,idx):

        face_feat_iter = np.zeros(1)
        voice_feat_iter = np.zeros(1)
        text_feat_iter = np.zeros(1)

        text_iter = proc_ques(self.data_text[idx],self.token_to_ix,14,45)
        

        face_feat_iter = self.face_feat_list[idx]
        voice_feat_iter = self.voice_feat_list[idx]
        # print("face:",self.face_feat_list.__len__())
        ans_iter = self.ans_to_ix[idx]
        
        return torch.from_numpy(face_feat_iter).type(torch.float), \
               torch.from_numpy(voice_feat_iter).type(torch.float), \
               torch.from_numpy(text_iter).type(torch.long), \
               torch.tensor(ans_iter,dtype=torch.float)

    def __len__(self):
        if self.train_or_test:
            return self.text_feat_train_list.__len__()
        else:
            return self.text_feat_test_list.__len__()
       

    
    def test(self):
        # print(self.face_feat_list.__len__())
        # print(self.voice_feat_list.__len__())
        # print(self.text_feat_list.__len__())
        pass 
        

class Singe_Language(Data.Dataset):
    def __init__(self,train_or_test):
       
        # {300:0 or 1}
        train_dict = np.load("/mnt/sdc1/daicwoz/data_pro/train_title2label.npy",allow_pickle=True).tolist()
        test_dict = np.load("/mnt/sdc1/daicwoz/data_pro/test_title2label.npy",allow_pickle=True).tolist()
        
        train_index = list(train_dict.keys())  #[indexs]
        test_index = list(test_dict.keys())    #[indexs]

        # {300:[[q],[a]]}
        text_data = np.load("/mnt/sdc1/daicwoz/data_pro/single/qa_list.npy",allow_pickle=True).tolist()
        
        # [ [[q],[a]],[label]], [...]]
        self.data_list = []

        if train_or_test:
            for item in train_index:
                if item in [367,396,451,458,480]:
                    continue
                else:
                    try:
                        temp_data = text_data[item]
                        self.data_list.append([temp_data,train_dict[item]])
                    except:
                        print("except",item)
        else:
            for item in test_index:
                if item in [367,396,451,458,480]:
                    continue
                else:
                    try:
                        temp_data = text_data[item]
                        self.data_list.append([temp_data,test_dict[item]])
                    except:
                        print("except",item) 
    
        ## tokenize
        self.text_list_all = []
        for i in text_data.values():
            self.text_list_all.extend(i[0])
            self.text_list_all.extend(i[1])
        # input ["sentence","sentence"]
        self.token_to_ix, self.pretrained_emb = tokenize(self.text_list_all, True)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

    def __getitem__(self,idx):

        data = self.data_list[idx]  #[ [[[q],[a]],label] , [...] ]

        # method one, only answer

        answer_list = data[0][1]

        ## q and a 
        # answer_list = data[0]
        # text_iter = proc_ques_q_and_ans_only(answer_list,self.token_to_ix,3000)
        # text_iter = proc_ques_answer_only(answer_list,self.token_to_ix,1500)
        text_iter = answer_only_with_padding(answer_list,self.token_to_ix,70,3000)
        ans_iter = data[1]
        
        return torch.from_numpy(text_iter).type(torch.long), \
               torch.tensor(ans_iter,dtype=torch.float)

    def __len__(self):
        return self.data_list.__len__()
    

if __name__ == "__main__":
    train_dataset = Singe_Language(train_or_test=True)
    
    train_data_iter = DataLoader(train_dataset,batch_size=1,shuffle=False)
    
    for i in train_data_iter:
        print(i[0].shape)

    print(train_dataset[2][0].shape)