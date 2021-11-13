# /mnt/ssd/hpb/depression/data_pro/alignment
# 390_alignment_concat_feature.csv  w/o ellie
# ['face_id_end', 'face_id_start', 'question', 'speaker', 'start_time',
    #    'stop_time', 'value', 'voice_id_end', 'voice_id_start']

# 390_alignment_feature.csv
# ['start_time', 'stop_time', 'speaker', 'value', 'face_id_start',
    #    'face_id_end', 'voice_id_start', 'voice_id_end'],

# Devault, naive bayes, 74.4%
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import os
import re
import en_vectors_web_lg, random,json
import torch.utils.data.dataset as Data  
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data_dir='/mnt/sdc1/daicwoz/data_pro/alignment'
valence_dict=np.load('valence_dict.npy',allow_pickle=True).tolist()

TAALES_df=pd.read_csv('TAALES_results.csv')
Filenames=TAALES_df['Filename']
TAALES_df=TAALES_df.drop(['Filename'],axis=1).drop(['Word Count'],axis=1)
pca = PCA(n_components=20)
TAALES_df = pd.DataFrame(pca.fit_transform(TAALES_df))
TAALES_features=TAALES_df.columns[:30]
TAALES_df['Filename']=Filenames


def tokenize(stat_ques_list, use_glove):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
        'SS' : 2,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
        pretrained_emb.append(spacy_tool("SS").vector)

    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb

def answer_only_with_padding(ques, token_to_ix, sentence_length,max_length):
    """
        add padding to each sentences to make sure every sentence
        has the same length
        ques ["sent","sent"]
    """
    ques_ix = []
    if_large=False
    for item in ques:
        word_a = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            item.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        temp_list = []
        for word in word_a:
            if word in token_to_ix:
                temp_list.append(token_to_ix[word])
            else:
                temp_list.append(0)
        if temp_list.__len__()>sentence_length:
            ## delete 
            temp_list = temp_list[:sentence_length]
        else:
            ## add 
            temp_list.extend([0 for i in range(sentence_length-temp_list.__len__())])
        ques_ix.extend(temp_list)
        ques_ix.append(2) #<ss>

               
       
    
    if ques_ix.__len__()<max_length:
        for i in range(max_length-ques_ix.__len__()):
            ques_ix.append(0)
    else:
        ques_ix = ques_ix[:max_length]
    return np.array(ques_ix)

def extract_devault_features(item):
    df=pd.read_csv(os.path.join(data_dir,str(item)+'_alignment_feature.csv'))
    # print(df.head())
    # print(df.columns)
    speak_rate=[]
    onset_time_first_seg=[]
    onset_time_nonfirst_seg=[]

    min_valence=[]
    mean_valence=[]
    max_valence=[]

    filled_tokens=['uh', 'um', 'mm']
    num_filled=[]
    filled_rate=[]
    
    length_seg=[]
    
    total_num_seg=0
    total_len_seg=0
    for i in range(len(df)):
        if df['speaker'][i]=='Participant':
            total_num_seg+=1
            tmplen=len(df['value'][i].split(' '))
            total_len_seg+=tmplen
            speak_rate.append(tmplen/(df['stop_time'][i]-df['start_time'][i]))
            length_seg.append(tmplen)

            cnt=0
            valence=[]
            transp=df['value'][i]
            transp = re.sub(
                    r"([.,'!?\"\'(<>)*#:;])",
                    '',
                    transp.lower()
                ).replace('-', ' ').replace('/', ' ').split()   
            for w in transp:
                for t in filled_tokens:
                    if t in w:
                        cnt+=1
                if w in valence_dict:
                    valence.append(valence_dict[w])
            
            num_filled.append(cnt)
            filled_rate.append(cnt/tmplen)
            if valence:
                min_valence.append(min(valence))
                mean_valence.append(np.mean(valence))
                max_valence.append(max(valence))

            if i>=1:
    # for i in range(1,len(df)):
        # if df['speaker'][i]=='Participant':
                if df['speaker'][i-1]!='Participant':# prev is ellie, first seg
                    onset_time_first_seg.append(df['start_time'][i]-df['stop_time'][i-1])
                else:
                    onset_time_nonfirst_seg.append(df['start_time'][i]-df['stop_time'][i-1])


    feature_list=[np.mean(speak_rate), # (a)
        np.mean(onset_time_first_seg),# (b)
        np.mean(onset_time_nonfirst_seg),# (c)
        np.mean(length_seg),#(d)
        np.mean(min_valence),# (e)
        np.mean(mean_valence),# (f)
        np.mean(max_valence),# (g)
        np.mean(num_filled),# (h)
        np.mean(filled_rate),#(i)
        total_num_seg, # (j)
        total_len_seg # (k)
        ]
    return feature_list

def extract_taales_features(item):
    # print(item)
    feature_list=[]
    item_df=TAALES_df[TAALES_df['Filename']==item]
    item_df=item_df[TAALES_features]
    item_df.reset_index(inplace=True,drop=True)
    for i in item_df.columns:
        feature_list.append(item_df[i][0])
    return feature_list
    # return list(np.array(item_df).reshape(len(TAALES_df.columns),1))

# if __name__== "__main__":
# only context-free features
class Devault_language(Data.Dataset):
    def __init__(self,train_or_test):
        self.all_features=[]
        self.labels=[]
        self.itemnum=[]
        train_dict = np.load("/mnt/sdc1/daicwoz/data_pro/train_title2label.npy",allow_pickle=True).tolist()
        test_dict = np.load("/mnt/sdc1/daicwoz/data_pro/test_title2label.npy",allow_pickle=True).tolist()
        train_index = list(train_dict.keys())  #[indexs]
        test_index = list(test_dict.keys())    #[indexs]
        if train_or_test:
            for item in train_index:
                if item in [367,396,451,458,480]:
                        continue
                feature_list=extract_devault_features(item)
                self.all_features.append(feature_list)
                # print(item)
                self.labels.append(train_dict[item])
                self.itemnum.append(item)
        else:
            for item in test_index:
                # print(len(test_index))
                # exit()
                # print(item)
                if item in [367,396,451,458,480]:
                    continue
                feature_list=extract_devault_features(item)
                self.all_features.append(feature_list)
                self.labels.append(test_dict[item])
                self.itemnum.append(item)
        # print(self.all_features[:2])
        scaler=StandardScaler()
        self.all_features=scaler.fit_transform(self.all_features)
        # print(self.all_features[:2])
        # exit()
    def __getitem__(self,idx):
        data=np.array(self.all_features[idx])
        return torch.from_numpy(data).type(torch.long),\
                torch.tensor(self.labels[idx],dtype=torch.float)

    def __len__(self):
        return self.all_features.__len__()
    
    def get_all(self):
        return self.all_features,self.labels ,self.itemnum

def recover_qa(text_data_idx):
    questions=[]
    answers=[]
    for i in text_data_idx[1]:
        questions.append(i[0])
        answers.append(i[1])
    temp_data=[questions,answers]
    return temp_data

# context-free features and original features
class Devault_Single_Language(Data.Dataset):
    def __init__(self,train_or_test,args,data_aug=False):
        self.data_aug=data_aug
        self.args=args
        # {300:0 or 1}
        train_dict = np.load("/mnt/sdc1/daicwoz/data_pro/train_title2label.npy",allow_pickle=True).tolist()
        test_dict = np.load("/mnt/sdc1/daicwoz/data_pro/test_title2label.npy",allow_pickle=True).tolist()
        
        train_index = list(train_dict.keys())  #[indexs]
        test_index = list(test_dict.keys())    #[indexs]

        # {300:[[q],[a]]}
        # if train_or_test:
        #     text_data = np.load("/mnt/ssd/hpb/depression/data_pro/split/split_feature_qa.npy",allow_pickle=True).tolist()
        # else:
        #     text_data = np.load("/mnt/ssd/hpb/depression/data_pro/split/split_feature_qa_test.npy",allow_pickle=True).tolist()
        text_data = np.load("/mnt/sdc1/daicwoz/data_pro/single/qa_list.npy",allow_pickle=True).tolist()
        

        # [ [[q],[a]],[label]], [...]]
        self.data_list = []
        self.train_items=[]
        self.devault_features=[]
        self.taales_features=[]
        if train_or_test:
            for item in train_index:
                if item in [367,396,451,458,480]:
                    continue
                else:
                    # try:
                    # temp_data = recover_qa(text_data[item])
                    temp_data = text_data[item]
                    self.data_list.append([temp_data,train_dict[item]])

                    feature_list=extract_devault_features(item)
                    self.devault_features.append(feature_list)
                    if args.TAALES:
                        taales_feature_list=extract_taales_features(item)
                        self.taales_features.append(taales_feature_list)
                    if data_aug:
                        for i in range(9):
                            # text_data_idx=text_data[item]
                            # random.shuffle(text_data_idx[1])
                            # temp_data = recover_qa(text_data[item])
                            temp_data = text_data[item]
                            # print(temp_data[1][:4])
                            # exit()
                            random.shuffle(temp_data[1])
                            self.data_list.append([temp_data,train_dict[item]])
                            self.devault_features.append(feature_list)
                            if args.TAALES:
                                self.taales_features.append(taales_feature_list)

                    self.train_items.append(item)
                    # except:
                        # print("except",item)
        else:
            for item in test_index:
                if item in [367,396,451,458,480]:
                    continue
                else:
                    # try:
                    # temp_data = recover_qa(text_data[item])
                    temp_data = text_data[item]
                    self.data_list.append([temp_data,test_dict[item]])
                    feature_list=extract_devault_features(item)
                    self.devault_features.append(feature_list)
                    if args.TAALES:
                        taales_feature_list=extract_taales_features(item)
                        self.taales_features.append(taales_feature_list)
                    # if data_aug:
                    #     for i in range(2):
                    #         text_data_idx=text_data[item]
                    #         random.shuffle(text_data_idx[1])
                    #         temp_data = recover_qa(text_data[item])
                    #         self.data_list.append([temp_data,test_dict[item]])
                    #         self.devault_features.append(feature_list)
                    # except:
                        # print("except",item) 

        scaler=StandardScaler()
        self.devault_features=scaler.fit_transform(self.devault_features)
        if args.TAALES:
            print(len(self.devault_features))
            print(len(self.devault_features[0]))
            print(len(self.taales_features))
            print(len(self.taales_features[0]))
            scaler=StandardScaler()
            self.taales_features=scaler.fit_transform(self.taales_features)
        
        ## tokenize
        self.text_list_all = []
        for i in text_data.values():
            # print(i)
            # i=recover_qa(i)
            self.text_list_all.extend(i[0])
            self.text_list_all.extend(i[1])
            # for j in i[1]:
                # self.text_list_all.extend(j)
                
        # input ["sentence","sentence"]
        self.token_to_ix, self.pretrained_emb = tokenize(self.text_list_all, True)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)
    def train_item(self):
        return self.train_items
    def __getitem__(self,idx):
        data = self.data_list[idx]  #[ [[[q],[a]],label] , [...] ]
        devaultf = np.array(self.devault_features[idx])
        # method one, only answer
        answer_list = data[0][1]
        ## q and a 
        # answer_list = data[0]
        # text_iter = proc_ques_q_and_ans_only(answer_list,self.token_to_ix,3000)
        # text_iter = proc_ques_answer_only(answer_list,self.token_to_ix,1500)
        
        text_iter = answer_only_with_padding(answer_list,self.token_to_ix,70,3000)
        ans_iter = data[1]
        if not self.args.TAALES:
            return torch.from_numpy(text_iter).type(torch.long), \
               torch.from_numpy(devaultf).type(torch.float),\
               torch.tensor(ans_iter,dtype=torch.float)
        else:
            taalesf=np.array(self.taales_features[idx])
            return torch.from_numpy(text_iter).type(torch.long), \
               torch.from_numpy(devaultf).type(torch.float),\
               torch.from_numpy(taalesf).type(torch.float),\
               torch.tensor(ans_iter,dtype=torch.float)

    def __len__(self):
        return self.data_list.__len__()
    
    
class Devault_Single_CCNN(nn.Module):

    def __init__(self,USE_GLOVE,pretrained_emb, token_size,args):
        super(Devault_Single_CCNN, self).__init__()
        
        
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
        if args.freeze_conv:
            for param in self.parameters():
                param.requires_grad = False
        # flatten
        # fully connected (fc) unit
        if args.TAALES:
            fc_unit_input=128+11+len(TAALES_features)
        else:
            fc_unit_input=128+11
        print('FC input dim=',fc_unit_input)
        

        self.fc_unit = nn.Sequential(
            nn.Linear(fc_unit_input,64),
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

    def taales_forward(self, inputs,de_feature,taalesf):
        inputs = self.embedding(inputs)
        inputs = torch.transpose(inputs,1,2)  #for 1dCNN input 
        out = self.conv_unit(inputs)  # (b,128,len)
        out = torch.transpose(out, 1, 2) #(b,len,128)
        length = out.shape[1]
        embedding_v = out[:,-1,:]
        
        combo=torch.cat([embedding_v, de_feature,taalesf], 1)
        logits = self.fc_unit(combo)
        # print(logits)
        return logits
    def forward(self, inputs,de_feature):
        # input (b,len,dim)

        inputs = self.embedding(inputs)

        inputs = torch.transpose(inputs,1,2)  #for 1dCNN input 
        out = self.conv_unit(inputs)  # (b,128,len)
        out = torch.transpose(out, 1, 2) #(b,len,128)
        length = out.shape[1]
        #embedding_v = out[: ,length//2, :]  #get center vector
        embedding_v = out[:,-1,:]
        
        # print(de_feature.shape)
        # print(embedding_v.shape)
        # exit()
        combo=torch.cat([embedding_v, de_feature], 1)
        # print('shape combo',combo.shape)
        logits = self.fc_unit(combo)
        # logits = self.fc_unit(embedding_v)
        return logits

class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
                                      
        super(CausalConv1d, self).__init__()
        
        # attributes:
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilationwt
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

class CCNN_conv(nn.Module):

    def __init__(self,USE_GLOVE,pretrained_emb, token_size,args):
        super(CCNN_conv, self).__init__()
        self.args=args
        
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
        if args.freeze_conv:
            for param in self.parameters():
                param.requires_grad = False
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


    def forward(self, inputs,de_feature):
        # input (b,len,dim)
        inputs = self.embedding(inputs)
        inputs = torch.transpose(inputs,1,2)  #for 1dCNN input 
        out = self.conv_unit(inputs)  # (b,128,len)
        out = torch.transpose(out, 1, 2) #(b,len,128)
        length = out.shape[1]
        #embedding_v = out[: ,length//2, :]  #get center vector
        embedding_v = out[:,-1,:]
        
        # if args.freeze_conv:
        #     combo=torch.cat([embedding_v, de_feature], 1)
        #     logits = self.fc_unit(combo)
        #     return logits
        logits = self.fc_unit(embedding_v)
        return logits,embedding_v


class CCNN_fc(nn.Module):

    def __init__(self,args):
        super(CCNN_fc, self).__init__()
        
        # flatten
        # fully connected (fc) unit
        self.fc_unit = nn.Sequential(
            nn.Linear(128+11,64),
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

    def forward(self, embedding_v,de_feature):
        combo=torch.cat([embedding_v, de_feature], 1)

        # print('shape combo',combo.shape)
        logits = self.fc_unit(combo)
        # logits = self.fc_unit(embedding_v)
        return logits

