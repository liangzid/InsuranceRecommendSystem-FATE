import jieba.analyse as ja
from collections import defaultdict
import math
import operator
import jieba.posseg as pseg

import argparse

# get clustering tool
from sklearn.cluster import AgglomerativeClustering as aggcluster
from sklearn.cluster import KMeans as kmeans

from gensim import models

import numpy as np
import pprint
import pickle
import os

# from statistics import fromList2contents as fl2c

# from tokenizer_and_model import MyTokenizer, get_model, GetDataset
from torch.utils.data import DataLoader,Dataset
from sklearn.cluster import KMeans

import numpy as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader,Dataset
from transformers import BertModel,BertTokenizer, ElectraTokenizer, ElectraModel
import torch
from tqdm import tqdm
import time
import pickle
import os
import json

class MyTokenizer():
    def __init__(self, max_sentence_length=64):
        self.path=os.path.dirname(os.path.abspath(__file__))
        self.max_sentence_length=max_sentence_length
        self.tokenizer=ElectraTokenizer.from_pretrained(self.path+"/"+'chinese-electra-180g-small-discriminator')
        self.tokenizer.add_special_tokens({"additional_special_tokens":["[DOMAIN]","[NAME]",
             "[NAME_ATBTE]",
             "[AT_VALUE]",
             "[POSITIVE]",
             "[NEGATIVE]",
             "[MAYBE_YOU_LIKE]",
             "[DOYOU_LIKE_IT?]",
             "[DOYOU_LIKE_IT?]"]})
        
    def __len__(self):
        return len(self.tokenizer)

    def single_convert_text_into_indextokens_and_segment_id(self,text):
        tokeniz_text = self.tokenizer.tokenize(text)
        indextokens = self.tokenizer.convert_tokens_to_ids(tokeniz_text)
        # indextokens.append(self.tokenizer.convert_tokens_to_ids('[EndOfResponse]'))
        input_mask = [1] * len(indextokens)

        if self.max_sentence_length<len(indextokens):
            indextokens=indextokens[:self.max_sentence_length]
            segment_id=[0]*self.max_sentence_length
            input_mask=input_mask[:self.max_sentence_length]
        else:
            pad_indextokens = [0]*(self.max_sentence_length-len(indextokens))
            indextokens.extend(pad_indextokens)
            input_mask_pad = [0]*(self.max_sentence_length-len(input_mask))
            input_mask.extend(input_mask_pad)
            segment_id = [0]*self.max_sentence_length

        indextokens=torch.tensor(indextokens,dtype=torch.long)
        segment_id=torch.tensor(segment_id,dtype=torch.long)
        input_mask=torch.tensor(input_mask,dtype=torch.long)

        return indextokens,segment_id,input_mask

    def single_convert_text_into_tokeniz_textes(self,text):
        tokenize_text=self.tokenizer.tokenize(text)
        return tokenize_text

    def convert_text_into_indextokens_and_segment_id(self,text1,text2,spl='[SEP]'):
        tokeniz_text1 = self.tokenizer.tokenize(text1)
        indextokens1 = self.tokenizer.convert_tokens_to_ids(tokeniz_text1)

        length_first=self.max_sentence_length//2
        input_mask1 = [1] * len(indextokens1)

        if length_first<len(indextokens1):
            indextokens1=indextokens1[:length_first]
            input_mask1=input_mask1[:length_first]
        else:
            pad_indextokens1 = [0]*(length_first-len(indextokens1))
            indextokens1.extend(pad_indextokens1)
            input_mask_pad1 = [0]*(length_first-len(input_mask1))
            input_mask1.extend(input_mask_pad1)

        text2=spl+text2
        tokeniz_text2 = self.tokenizer.tokenize(text2)
        indextokens2 = self.tokenizer.convert_tokens_to_ids(tokeniz_text2)

        length_second=self.max_sentence_length-length_first
        length2_begin=length_first
        
        input_mask2 = [1] * len(indextokens2)

        if length_second<len(indextokens2):
            indextokens2=indextokens2[:length_second]
            input_mask2=input_mask2[:length_second]
        else:
            pad_indextokens2 = [0]*(length_second-len(indextokens2))
            indextokens2.extend(pad_indextokens2)
            input_mask_pad2 = [0]*(length_second-len(input_mask2))
            input_mask2.extend(input_mask_pad2)

            
        indextokens1.extend(indextokens2)
        input_mask1.extend(input_mask2)
        indextokens=torch.tensor(indextokens1,dtype=torch.long)
        # segment_id=torch.tensor(segment_id,dtype=torch.long)
        input_mask=torch.tensor(input_mask1,dtype=torch.long)

        return indextokens,None,input_mask

    def convert_ids_to_tokens(self, index):
        return self.tokenizer.convert_ids_to_tokens(index)
    def convert_prediction_result2_sentence(self,output_prediction):
        output_index=output_prediction.cpu()[0].numpy().tolist()
        # print(output_index)
        return self.tokenizer.convert_ids_to_tokens(output_index)
        
    def indextoken2wordtoken(self,index):
        return self.tokenizer.convert_ids_to_tokens(index)

    def convert_tokens_to_ids(self,token):
        return self.tokenizer.convert_tokens_to_ids(token)


def get_model():
    path=os.path.dirname(os.path.abspath(__file__))
    tokenizer=MyTokenizer()
    model=ElectraModel.from_pretrained(path+"/"+"chinese-electra-180g-small-discriminator")
    model.resize_token_embeddings(len(tokenizer))
    # model=DataParallel(model,device_ids=[0,1,2])
    return model

class GetDataset(Dataset):

    def __init__(self,tokenizer,data_list):
        self.max_sentence_length=64
        self.tokenizer=tokenizer

        # get tagging label for each dataset.
        self.dataset=[]
        for i, sent in enumerate(data_list):
            sent="[CLS]"+sent
            sou_t=self.tokenizer.single_convert_text_into_tokeniz_textes(sent)
            index_tokens, segment_id,input_mask=self.tokenizer.single_convert_text_into_indextokens_and_segment_id(sent)
            self.dataset.append((index_tokens,segment_id,input_mask))

    def __getitem__(self,i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)



def get_embeddings_for_sentlist(sentence_list):
    tokenizer=MyTokenizer()
    model=get_model()

    # construct loader.
    dataloader=DataLoader(dataset=GetDataset(tokenizer,sentence_list), batch_size=1, shuffle=False)
    
    embedding_list=[]
    # model=model.to("cuda")
    model.eval()
    for index, (indextokens,_,attention_mask) in enumerate(dataloader):
        # # 使用tokenizer转化为需要的token结构
        # # indextokens, _, attention_mask=tokenizer.single_convert_text_into_indextokens_and_segment_id(sentence)
        # indextokens=indextokens.to("cuda")
        # attention_mask=attention_mask.to("cuda")

        # # indextokens=indextokens.unsqueeze(0)
        # # attention_mask=attention_mask.unsqueeze(0)
        # # print(indextokens.shape)
        # # print(attention_mask.shape)
        # # print(indextokens)
        # # print(attention_mask)
        
        # 使用model进行正向传播, 得到所需的embedding
        embedding=model(input_ids=indextokens, attention_mask=attention_mask).last_hidden_state
        embedding=embedding[0][0][:]
        # print(embedding[0][0][:])
        # 将embedding的类型转化为numpy数组类型
        embedding=embedding.cpu().detach().numpy()
        # print(embedding)
        xnorm=np.linalg.norm(embedding)
        embedding_list.append(embedding/xnorm)
    print("Done for inference.")
    embeddings=np.array(embedding_list)
    # similarity_result=np.zeros((len(sentence_list), len(sentence_list)),dtype=np.double)
    # similarity_result=np.dot(embeddings, embeddings.T)
    return embeddings


class NLUmodel:
    def __init__(self,loading_path=None,need_train=None):
        self.loading_path=loading_path
        self.need_train=need_train

    def train(self,dataloader,lr):
        pass
        
    def inference(self,text=None):

        embedding=None

        return embedding


## NLU with Chinese-Electra-Models.
class electraNLU(NLUmodel):

    def __init__(self,loading_path=None,need_train=None):
        self.loading_path=loading_path
        self.need_train=need_train

    def train(self,dataloader,lr):
        pass
        
    def inference(self,text=None):

        embedding=None

        return embedding

## NLU with TF_IDF
class TFIDF(NLUmodel):

    def __init__(self,loading_path=None,need_train=None):
        self.loading_path=loading_path
        self.need_train=need_train

    def train(self,dataloader,lr):
        pass

    def inference(self,text=None):
        embedding=None
        return embedding


    def tf_idf(self,documentlist):

        sentlist=documentlist

        # frequency of every words
        word_fre=defaultdict(int)
        for sent in sentlist:
            for word in sent:
                word_fre[word]+=1

        # calculate TF
        word_tf={}
        for i in word_fre.keys():
            word_tf[i]=word_fre[i]/(sum(word_fre.values()))

        # calculate IDF
        num_dialogs=len(sentlist)
        word_idf={}
        word_doc=defaultdict(int)
        for i in word_fre:
            for j in sentlist:
                if i in j:
                    word_doc[i]+=1

        for i in word_fre:
            word_idf[i]=math.log(num_dialogs/(word_doc[i]+1.0))

        # calculate TF*IDF
        word_tf_idf={}
        for i in word_fre:
            word_tf_idf[i]=word_tf[i]*word_idf[i]

        # sort the values of word_tf_idf
        sorted_words=sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)

        return sorted_words

def makeSegment(sentence_string):
    adiedai=pseg.lcut(text)
    return list(adiedai)

def readInsuranceFile(path):
    with open(path,'r',encoding='utf8') as f:
        data=f.readlines()

    result=""
    for line in data:
        result+=line[:-1]
    return result

def setup_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str, required=False)
    parser.add_argument("--query",  type=str, required=False,default="我想去看意外险")
    parser.add_argument("--save_data_path", default='/home/szhang/liangzi_need_smile/fatee/isr/BackEnd/data/ncf_training_data.pk', type=str, required=False)
    return parser.parse_args()

if __name__=="__main__":
    args=setup_args()
    path="/home/szhang/liangzi_need_smile/fatee/isr/BackEnd/data/"
    baoxian1=readInsuranceFile(path+"/baoxian1.txt")
    baoxian2=readInsuranceFile(path+"/baoxian2.txt")

    # sentlist=[makeSegment(baoxian1),makeSegment(baoxian2)]
    # query="坐飞机的保险可以找得到吗？"
    # query=makeSegment(query)
    # sentlist.append(query)
    
    sentlist=[baoxian1,baoxian2]
    query="坐飞机的保险可以找得到吗？"
    query="重大疾病保险？"
    # query=makeSegment(query)
    sentlist.append(query)

    embeddings=get_embeddings_for_sentlist(sentlist)
    with open(args.save_data_path,'wb') as f:
        pickle.dump(embeddings,f)

    print(embeddings.shape)

    print("save done.")
    
    # print(embeddings)
    print(np.dot(embeddings[0],embeddings[2]),
          np.dot(embeddings[1],embeddings[2]))

    # print(embeddings)
