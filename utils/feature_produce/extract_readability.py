# ARI = 4.71 * (characters/words) + 0.5 * (words/sentence) -21.43, the higher the richer
# Flesch-Kincaid readability # the lower the richer
# FKGL = 0.39 * (total words/ total sentences) + 11.8 (total syllables/ total words) -15.59
import fkscore
import pandas as pd 
import nltk 
# nltk.download('punkt')
import numpy as np

from nltk.corpus import stopwords
from nltk import word_tokenize 
import string 
import os
import re
# text = '...blah blah blah...'
# f = fkscore.fkscore(text)
train_dict = np.load("/mnt/sdc1/daicwoz/data_pro/train_title2label.npy",allow_pickle=True).tolist()
test_dict = np.load("/mnt/sdc1/daicwoz/data_pro/test_title2label.npy",allow_pickle=True).tolist()

train_index = list(train_dict.keys())  #[indexs]
test_index = list(test_dict.keys())    #[indexs]
item_index=train_index+test_index

# save_dir='/mnt/sdc1/daicwoz/data_pro/single'
save_dir='/home/mengyixuan/workspace/models/extracted_features'
data_dir='/mnt/sdc1/daicwoz/data_pro/alignment'

def cal_ARI(text):
    characters=len(text.replace(' ','').replace('.',''))
    words=len(text.split())
    sentence=len(text)-len(text.replace('.',''))
    print('sentence',sentence)
    ARI=4.71 * (characters/words) + 0.5 * (words/sentence) -21.43
    return ARI
    
def readability_features(text):
    # print(text)
    # exit()
    ARI=cal_ARI(text)
    print(ARI)
    f=fkscore.fkscore(text)
    # print(f.stats)
    FK=f.score['readability']
    # print(f.score['readability'])
    return np.array([ARI, FK])


feature_dict={}
for item in item_index:
    par_text=''
    df=pd.read_csv(os.path.join(data_dir,str(item)+'_alignment_feature.csv'))
    for i in range(len(df)):
        if df['speaker'][i]=='Participant':
            par_text+=df['value'][i]+'. '
    par_text = re.sub(
            r"([,'!?\"\'()*#:;])",
            '',
            par_text.lower()
        ).replace('-', ' ').replace('/', ' ')   
    features=readability_features(par_text)
    feature_dict[item]=features
    print(features)
    # break

np.save(os.path.join(save_dir,'readability_feature.npy'), feature_dict)

# Your text: I am intelligent. je sui bien. I am intelligent. j ...(show all text)

# Flesch Reading Ease score: 76.9 (text scale)
# Flesch Reading Ease scored your text: fairly easy to read.

# Gunning Fog: 7.9 (text scale)
# Gunning Fog scored your text: fairly easy to read.

# Flesch-Kincaid Grade Level: 3.3
# Grade level: Third Grade.

# The Coleman-Liau Index: 5
# Grade level: Fifth Grade

# The SMOG Index: 4.4
# Grade level: Fourth Grade

# Automated Readability Index: -1.9
# Grade level: 3-5 yrs. old (Preschool)

# Linsear Write Formula : 1
# Grade level: First Grade.
