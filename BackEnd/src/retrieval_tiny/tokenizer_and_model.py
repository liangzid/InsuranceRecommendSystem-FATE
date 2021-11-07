"""
Something of Models and Tokenizers.

Zi Liang.
2021.04.12
"""


from torch.nn import DataParallel
from torch.utils.data import DataLoader,Dataset
from transformers import BertModel,BertTokenizer, ElectraTokenizer, ElectraModel
import torch
from tqdm import tqdm
import time
import pickle
import os
import json
import random

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



