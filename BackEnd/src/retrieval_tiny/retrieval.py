"""
封装以自己使用的基于中文预训练模型的检索模块.

Zi Liang

"""

from torch.utils.data import DataLoader,Dataset
from sklearn.cluster import KMeans
import numpy as np
from retrieval_tiny.tokenizer_and_model import MyTokenizer, get_model, GetDataset

def clustering_retrieval(sentence_list):
    tokenizer=MyTokenizer()
    model=get_model()
    
    embedding_list=[]
    # model.to("cuda")
    for index, sentence in enumerate(sentence_list):
        # 使用tokenizer转化为需要的token结构
        indextokens, _, attention_mask=tokenizer.single_convert_text_into_indextokens_and_segment_id(sentence)
        # indextokens.to("cuda")
        # attention_mask.to("cuda")
        # 使用model进行正向传播, 得到所需的embedding
        embedding=model(input_ids=indextokens, attention_mask=attention_mask)[1]
        # 将embedding的类型转化为numpy数组类型
        embedding=embedding.cpu()[0].numpy().tolist()
        
        embedding_list.append(embedding_list)

    # 聚类
    embeddings=np.array(embedding_list)
    ...

    return -1

def retrieval_similarity_matrix(sentence_list):
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
    similarity_result=np.zeros((len(sentence_list), len(sentence_list)),dtype=np.double)
    similarity_result=np.dot(embeddings, embeddings.T)
    return similarity_result, embeddings

def get_embedd_sentlist(sentence_list):
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
        xnorm=np.linalg.norm(embedding)
        embedding_list.append(embedding/xnorm)
    embeddings=np.array(embedding_list)
    return embeddings


def __test_retrieval_sim_matrix():
    
    sentence_list=["今 天 的 天 气 很 好",
                   "明 天 的 天 气 很 差",
                   "今 天 的 天 气 不 好",
                   "今 天 的 天 气 好"]

    sim_result,embeddings=retrieval_similarity_matrix(sentence_list)

    print(sim_result)
    print(embeddings)

if __name__=="__main__":
    __test_retrieval_sim_matrix()

    
