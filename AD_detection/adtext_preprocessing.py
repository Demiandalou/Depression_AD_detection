# coding: utf-8
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords as StopWords

wnl = WordNetLemmatizer() 
NUM_CLASS=2
data_dir='AD_data'
resfile=os.path.join(data_dir,'train_set.csv')
# whole_text=
# mystring=re.sub('[^A-Za-z\u4e00-\u9fa5 ]+', '', mystring)
AD_dict={'Pitt/Dementia/cookie':1, 'Pitt/Control/fluency':0, 'Holland':1, 'Pitt/Control/cookie':0, \
        'PerLA':1, 'Jalvingh':0, 'Madarin_Lu':1, 'Lanzi/Group2':0, 'Hopkins':0, 'Pitt/Dementia/recall':1, \
        'Pitt/Dementia/fluency':1, 'Lanzi/Treatment':0, 'Lanzi/Group1':0, 'Ye':0, 'Taiwanese_Lu':1, \
        'Kempler':1, 'Pitt/Dementia/sentence':1, 'DePaul':0}
chinese=['Madarin_Lu','Taiwanese_Lu','Ye']
all_setname=[]

def clear_transcript(corpus):
    cleared=''
    cnt=0
    for line in corpus.readlines():
        if line[:5]=='*PAR:':
            line=line.rstrip('\n').lstrip('*PAR:\t')
            line=re.sub('[^A-Za-z\u4e00-\u9fa5 ]+', '', line)
            if cleared!='' and cleared[-1]!=' ':
                cleared+=' '
            cleared+=line
            cnt+=1

        #     print(line)
        # if cnt==10:
        #     print(cleared)
        #     exit()
    return cleared

def clear_perAtranscript(corpus,par):
    cleared=''
    cnt=0
    for line in corpus.readlines():
        if line[:5]=='*'+par+':':
            line=line.rstrip('\n').lstrip('*'+par+':\t')
            line=re.sub('[^A-Za-z\u4e00-\u9fa5 ]+', '', line)
            if cleared!='' and cleared[-1]!=' ':
                cleared+=' '
            cleared+=line
            cnt+=1
    return cleared
def chi_stopwords():
    stopwords = [line.strip() for line in open(os.path.join('AD_data','chi_stopwords.txt'),encoding='UTF-8').readlines()]
    return stopwords
def eng_stopwords():
    stopwords = [line.strip() for line in open(os.path.join('AD_data','eng_stopwords.txt'),encoding='UTF-8').readlines()]
    return stopwords

def split_stop(sentence,isChinese):
  if isChinese==1:
    # split words
    sentence_seg = jieba.cut(sentence)
    sentence = ' '.join(sentence_seg)
    sentence=sentence.split(' ')
    # print(sentence)
    stopwords = chi_stopwords()
    res = ''
    for word in sentence:
        if word not in stopwords:
            res += word
            res += " "
  else:
    # lemmatize
    sentence=sentence.split(' ')
    for word in sentence:
    #   if word!=' ':
        # try:
        word=word.replace(' ','')
        word = word.lower() 
        word=wnl.lemmatize(word)
    # remove stop words
    res= [w for w in sentence if w not in StopWords.words('english')]
    # stopwords = eng_stopwords()
    # res = ''
    # for word in sentence:
        # if word not in stopwords:
            # res += word
            # res += " "
    res = " ".join(res)
  return res

if __name__ == "__main__":
    transcript=[]
    label=[]
    isChinese=[]
    cnt=0
    all_word=set()
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if '.cha' in filename:
                set_name=root[8:]
                hasAD=AD_dict[set_name]
                # print(root,filename)
                corpus=open(os.path.join(root,filename),'r')
                if set_name=='PerLA':
                    par=filename[-7:-4]
                    cleared=clear_perAtranscript(corpus,par)
                else:
                    cleared=clear_transcript(corpus)
                
                if len(cleared)==0:
                    print(root,filename)
                elif set_name in chinese:
                    all_setname.append(set_name)
                    transcript.append(cleared)
                    label.append(hasAD)
                    isChinese.append(1)
                else:
                    all_setname.append(set_name)
                    transcript.append(cleared)
                    label.append(hasAD)
                    isChinese.append(0)
                cnt+=1
        #         if cnt>=10:
        #             break
        # if cnt>=10:
        #     break
    
    for i in range(len(transcript)):
        if isChinese[i]==1:
            transcript[i]=transcript[i].replace(' ','')
        transcript[i]=split_stop(transcript[i],isChinese[i])
        # print(transcript[i])
    #Vectorize
    vector = TfidfVectorizer()
    tf_data = vector.fit_transform(transcript)
    vecDict=vector.vocabulary_
    # print(vecDict)
    for i in range(len(transcript)):
        script=transcript[i].split(' ')
        transcript[i]=''
        for w in script:
            if w in vecDict:
                transcript[i]+=str(vecDict[w])
                transcript[i]+=' '
    # for tr in transcript:
        # print(tr)
    lengthes=[len(transcript[i]) for i in range(len(transcript))]
    print(min(lengthes),max(lengthes),np.mean(lengthes))

    maxlen=500
    resdf=pd.DataFrame()
    resdf['transcript']=transcript
    resdf['label']=label
    resdf['corpus']=all_setname
    resdf['isChinese']=isChinese
    
    # train_x_chi=resdf.drop(resdf[resdf.isChinese==0].index)
    
    resdf.to_csv(os.path.join(data_dir,'val.csv'),index=False,encoding='utf-8')

