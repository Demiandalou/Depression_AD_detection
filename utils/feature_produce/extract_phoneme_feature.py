import pandas as pd 
import nltk 
# nltk.download('punkt')
import numpy as np

from nltk.corpus import stopwords
from nltk import word_tokenize 
import string 
import os
import re
# from functools import lru_cache
from sklearn.decomposition import PCA
from itertools import product as iterprod
import itertools
try:
    arpabet = nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    arpabet = nltk.corpus.cmudict.dict()

TAALES_df=pd.read_csv('TAALES_results.csv')
Filenames=TAALES_df['Filename']
TAALES_df=TAALES_df.drop(['Filename'],axis=1).drop(['Word Count'],axis=1)
pca = PCA(n_components=20)
TAALES_df = pd.DataFrame(pca.fit_transform(TAALES_df))
TAALES_features=TAALES_df.columns[:30]
# TAALES_features=TAALES_df.columns
TAALES_df['Filename']=Filenames

# print('improt done')
train_dict = np.load("/mnt/sdc1/daicwoz/data_pro/train_title2label.npy",allow_pickle=True).tolist()
test_dict = np.load("/mnt/sdc1/daicwoz/data_pro/test_title2label.npy",allow_pickle=True).tolist()

train_index = list(train_dict.keys())  #[indexs]
test_index = list(test_dict.keys())    #[indexs]
item_index=train_index+test_index

# save_dir='/mnt/sdc1/daicwoz/data_pro/single'
save_dir='/home/mengyixuan/workspace/models/extracted_features'
data_dir='/mnt/sdc1/daicwoz/data_pro/alignment'
# @lru_cache()
def wordbreak(s):
    s = s.lower()
    if s in arpabet:
        return arpabet[s]
    middle = len(s)/2
    partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in arpabet and wordbreak(suf) is not None:
            return [x+y for x,y in iterprod(arpabet[pre], wordbreak(suf))]
    return None

# get mean phonemes per word
def phoneme_features(text):
    phoneme_num=[]
    for w in text:
        if '_' in w or '<' in w or '>' in w or '[' in w or ']' in w:
            continue
        # print(w)
        phonemes = list(itertools.chain.from_iterable(wordbreak(w)))
        phoneme_num.append(len(phonemes))
    return np.array([np.mean(phoneme_num)])
def extract_taales_features(item):
    # print(item)
    feature_list=[]
    item_df=TAALES_df[TAALES_df['Filename']==item]
    item_df=item_df[TAALES_features]
    item_df.reset_index(inplace=True,drop=True)
    for i in item_df.columns:
        feature_list.append(item_df[i][0])
    return np.array(feature_list)
    # return list(np.array(item_df).reshape(len(TAALES_df.columns),1))

feature_dict={}
# for item in item_index:
#     par_text=''
#     df=pd.read_csv(os.path.join(data_dir,str(item)+'_alignment_feature.csv'))
#     for i in range(len(df)):
#         if df['speaker'][i]=='Participant':
#             par_text+=df['value'][i]+' '
#     par_text = re.sub(
#             r"([.,'!?\"\'()*#:;])",
#             '',
#             par_text.lower()
#         ).replace('-', ' ').replace('/', ' ').split()    
#     features=phoneme_features(par_text)
#     # print(features,[item])
#     # break
#     feature_dict[item]=features
# np.save(os.path.join(save_dir,'Phoneme_feature.npy'), feature_dict)
    
if __name__=='__main__':
    for item in item_index:
        features=extract_taales_features(item)
        feature_dict[item]=features
    np.save(os.path.join(save_dir,'TAALES_feature.npy'), feature_dict)



#     from devault_feature import *
#     train_dataset = Devault_language(train_or_test=True)  #106
#     test_dataset = Devault_language(train_or_test=False)  #30
#     train_x,train_y,train_y_item=train_dataset.get_all()
#     test_x,test_y,test_y_item=test_dataset.get_all()

#     for i in range(len(train_x)):
#         feature_dict[train_y_item[i]]=train_x[i]
#     # print(len(test_y_item))
#     for i in range(len(test_x)):
#         feature_dict[test_y_item[i]]=test_x[i]
#     np.save(os.path.join(save_dir,'context_free_feature.npy'), feature_dict)
