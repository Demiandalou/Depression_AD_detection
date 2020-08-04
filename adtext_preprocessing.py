# coding: utf-8
import pandas as pd
import numpy as np
import os
import re

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

                # cnt+=1
                # if cnt==3:
                #     exit()
    # print(list(set(all_setname)))
    resdf=pd.DataFrame()
    resdf['transcript']=transcript
    resdf['label']=label
    resdf['corpus']=all_setname
    resdf['isChinese']=isChinese
    resdf.to_csv(resfile,index=False,encoding='utf-8')

