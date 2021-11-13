import fkscore
import pandas as pd 
import nltk 
import numpy as np

from nltk.corpus import stopwords
from nltk import word_tokenize 
import string 
import os
import re
import cophi

# Measures that use part of the frequency spectrum:

# Honoré’s H
# Sichel’s S
# Michéa’s M

# Brunet’s Measure (BM)
# there's a measure that use sample size and vocabulary size, called 'Brunet’s W' in cophi
# but it seems problematic, with value about 136153183, may not be the measure in the paper?

train_dict = np.load("/mnt/sdc1/daicwoz/data_pro/train_title2label.npy",allow_pickle=True).tolist()
test_dict = np.load("/mnt/sdc1/daicwoz/data_pro/test_title2label.npy",allow_pickle=True).tolist()

train_index = list(train_dict.keys())  #[indexs]
test_index = list(test_dict.keys())    #[indexs]
item_index=train_index+test_index

# save_dir='/mnt/sdc1/daicwoz/data_pro/single'
save_dir='/home/mengyixuan/workspace/models/extracted_features'
data_dir='/mnt/sdc1/daicwoz/data_pro/alignment'

def vocb_rich_features(text):
    f=open('tmp.txt','w')
    f.write(text)
    f.close()
    dickens = cophi.document(filepath="tmp.txt",
                         title="dickens-bleak",
                         lowercase=True,
                         n=2,
                         token_pattern=r"\p{L}+\p{P}?\p{L}+",
                         maximum=1000)
    Honore_Statistic = dickens.complexity(measure='honore_h')
    # 1489.9080013490884
    Sichel_Measure = dickens.complexity(measure='sichel_s')
    # 0.16666666666666666
    # >>> dickens.complexity(measure='brunet_w')
    # 136153183.3958158
    Michea_M = dickens.complexity(measure='michea_m')
    # 6.0
    os.remove('tmp.txt')
    return np.array([Honore_Statistic, Sichel_Measure, Michea_M])

feature_dict={}
for item in item_index:
    par_text=''
    df=pd.read_csv(os.path.join(data_dir,str(item)+'_alignment_feature.csv'))
    for i in range(len(df)):
        if df['speaker'][i]=='Participant':
            par_text+=df['value'][i]+' '
    par_text = re.sub(
            r"([.,'!?\"\'()*#:;])",
            '',
            par_text.lower()
        ).replace('-', ' ').replace('/', ' ')
    features=vocb_rich_features(par_text)
    feature_dict[item]=features
    print(features)
    # break
np.save(os.path.join(save_dir,'vocab_richness_feature.npy'), feature_dict)
