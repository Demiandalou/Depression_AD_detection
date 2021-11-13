# --------------------------------------------------------
# Tools for data pre-processing
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------


import sys
sys.path.append("..")
import en_vectors_web_lg, random, re, json
import numpy as np 

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


def proc_ques(ques, token_to_ix, max_token_question,max_token_answer):
    ques_ix = np.zeros(max_token_question+max_token_answer+1, np.int64)

    word_q = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques[0].lower()
    ).replace('-', ' ').replace('/', ' ').split()

    word_a = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques[1].lower()
    ).replace('-', ' ').replace('/', ' ').split()

    word_q.append("SS")
    

    for ix, word in enumerate(word_q):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token_question+1:
            break

    for ix, word in enumerate(word_a):
        if word in token_to_ix:
            ques_ix[max_token_question+1+ix] = token_to_ix[word]
        else:
            ques_ix[max_token_question+1+ix] = token_to_ix['UNK']

        if max_token_question+2+ix == max_token_question+max_token_answer+1:
            break

    return ques_ix


def proc_ques_answer_only(ques, token_to_ix, max_length):
    """
        transform ansewr matrix to one sentence token_id 
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

        
        for word in word_a:
            if word in token_to_ix:
                ques_ix.append(token_to_ix[word])
            else:
                ques_ix.append(0)
        ques_ix.append(2) #<ss>

               
       
    
    if ques_ix.__len__()<max_length:
        for i in range(max_length-ques_ix.__len__()):
            ques_ix.append(0)
    else:
        ques_ix = ques_ix[:max_length]
    return np.array(ques_ix)


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


def proc_ques_q_and_ans_only(ques, token_to_ix, max_length):
    """
        transform ansewr matrix to one sentence token_id 
        ques ["sent","sent"]
    """

    ques_q = ques[0]
    ques_a = ques[1]
    ques_ix = []
    
    for item in range(ques_q.__len__()):
        word_q = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques_q[item].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        
        for word in word_q:
            if word in token_to_ix:
                ques_ix.append(token_to_ix[word])
            else:
                ques_ix.append(0)
        ques_ix.append(2) #<ss>

        word_a = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques_a[item].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        
        for word in word_a:
            if word in token_to_ix:
                ques_ix.append(token_to_ix[word])
            else:
                ques_ix.append(0)
        ques_ix.append(3) #<ss>

               
       
    
    if ques_ix.__len__()<max_length:
        for i in range(max_length-ques_ix.__len__()):
            ques_ix.append(0)
    else:
        ques_ix = ques_ix[:max_length]
    return np.array(ques_ix)








def proc_face_voice_feature(feature, padding_size):
    if feature.__len__() >padding_size:
        feature = feature[:padding_size]

    feature = np.pad(
        feature,
        ((0, padding_size - feature.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return feature



# if __name__ == "__main__":
#     pass 