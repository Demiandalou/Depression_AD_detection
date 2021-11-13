import pandas as pd 
import nltk 
# nltk.download('punkt')
import numpy as np

from nltk.corpus import stopwords
from nltk import word_tokenize 
import string 
import os
import re
from collections import Counter
# print('improt done')
train_dict = np.load("/mnt/sdc1/daicwoz/data_pro/train_title2label.npy",allow_pickle=True).tolist()
test_dict = np.load("/mnt/sdc1/daicwoz/data_pro/test_title2label.npy",allow_pickle=True).tolist()

train_index = list(train_dict.keys())  #[indexs]
test_index = list(test_dict.keys())    #[indexs]
item_index=train_index+test_index

# save_dir='/mnt/sdc1/daicwoz/data_pro/single'
save_dir='/home/mengyixuan/workspace/models/extracted_features'
data_dir='/mnt/sdc1/daicwoz/data_pro/alignment'
stop = stopwords.words("english") + list(string.punctuation)

def pos_features(text):
    print('pos')
    # ttt=nltk.pos_tag([i for i in word_tokenize(str(text).lower())]) # without removing stopwords
    # word_tag_fq_stop = nltk.FreqDist(ttt) # (11) word frequency rate without excluding stop words.
    word_list=[i for i in word_tokenize(str(text).lower()) if i not in stop]
    tag_list_stop = nltk.pos_tag(word_list)
    # print(tag_list_stop[:5]) # [("'okay", 'POS'), ("'how", 'PRP'), ("'bout", 'IN'),
    word_tag_fq = nltk.FreqDist(tag_list_stop) # (10) word frequency rate
    tag_list = word_tag_fq.most_common() # [(("'you", 'POS'), 53), (("'and", 'CD'), 36), (("'know", 'POS'), 32),
    tags=[i[0][1] for i in tag_list]
    tags_dict=dict(Counter(tags))
    # print(tags_dict) # {'POS': 178, 'CD': 116, 'IN': 18, 'WP': 2, 'VBP': 26, 'NNS': 72, 'TO': 1, 'NN': 51, 'RB': 12, 'VBG': 26, 'NNP': 33, 'JJ': 80, "''": 47, 'VB': 27, 'VBZ': 20, 'VBD': 16, 'PRP': 23, 'CC': 11, 'MD': 21, 'FW': 25, 'JJS': 4, 
    # (1) number of pronouns
    num_pronoun=0
    for t in ['NNP','NNPS','PRP','WP']:
        if t in tags_dict:
            num_pronoun += tags_dict[t]
    print('num of pronouns',num_pronoun)
    # (2) pronoun-noun ratio
    num_noun=0
    for t in ['NN','NNS']:
        if t in tags_dict:
            num_noun += tags_dict[t]
    pron_noun_ratio = num_pronoun/num_noun
    print('pronoun-noun ratio',num_pronoun,'/',num_noun,'=',pron_noun_ratio)
    # (3) number of adverbs
    num_adv=0
    for t in ['RB','RBR','RBS']:
        if t in tags_dict:
            num_adv += tags_dict[t]
    print('num of adverb',num_adv)
    # (4) number of nouns
    print('num of nouns',num_noun)
    # (5) number of verbs
    num_verb=0
    for t in ['VB','VBD','VBG','VBN','VBP','VBZ']:
        if t in tags_dict:
            num_verb += tags_dict[t]
    print('num of verb',num_verb)
    # (6) pro-noun frequency rate,(7) noun frequency rate, (8) verb frequency rate, (9) adverb frequency rate
    pron_freq_rate = num_pronoun/len(word_list)
    noun_freq_rate = num_noun/len(word_list)
    verb_freq_rate = num_verb/len(word_list)
    adv_freq_rate = num_adv/len(word_list)

    # (10) word frequency rate
    word_dict = dict(Counter(word_list))
    cnt=0
    for i in word_dict:
        if word_dict[i]>=10:
            cnt+=word_dict[i]
            # print('wo stop',i,word_dict[i])
    print('word freq rate:',cnt,'/',len(word_list))
    word_freq_rate=cnt/len(word_list)
    
    # without removing stopwords
    word_list_wstop=[i for i in word_tokenize(str(text).lower())]
    word_dict_wstop = dict(Counter(word_list))
    cnt=0
    for i in word_dict_wstop:
        if word_dict_wstop[i]>=10:
            cnt+=word_dict_wstop[i]
            # print('w stop',i,word_dict[i])
    print('word freq rate wstop:',cnt,'/',len(word_list_wstop))
    word_freq_rate_wstop=cnt/len(word_list_wstop)
    return np.array([
        num_pronoun,pron_noun_ratio,num_adv,num_noun,num_verb,
        pron_freq_rate,noun_freq_rate,verb_freq_rate,adv_freq_rate,
        word_freq_rate,word_freq_rate_wstop
    ])
    # exit()


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
        ).replace('-', ' ').replace('/', ' ').split()    
    features=pos_features(par_text)
    feature_dict[item]=features
np.save(os.path.join(save_dir,'POSfeature.npy'), feature_dict)
    

# (1) number of pronouns, (2) pronoun-noun ratio, (3) number of adverbs, (4) number of nouns, (5) number of verbs, (6) pro-noun
# frequency rate, (7) noun frequency rate, (8) verb frequency rate, (9) adverb frequency rate, (10) word frequency rate,
# and (11) word frequency rate without excluding stop words.
