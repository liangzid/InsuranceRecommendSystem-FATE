from tokenizer_and_model import MyTokenizer, get_model, GetDataset
from torch.utils.data import DataLoader,Dataset
from sklearn.cluster import KMeans
import numpy as np

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







