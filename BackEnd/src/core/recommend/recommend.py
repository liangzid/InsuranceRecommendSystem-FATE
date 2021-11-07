import gensim.downloader as loader
import numpy as np
import faiss
import pickle
import jieba

class retrievalModel:
    """retrievaling candidates with FAISS. See below for demo information."""

    def __init__(self,d=200,candidate_load_path=None):
        self.msl=128
        self.d=d
        self.candidate_mat=self._loadCandidates()

        # construct indexes.
        self.INDEX=faiss.IndexFlatL2(200)

        self.INDEX.add(self.candidate_mat)
        print("index constructed done.")

    def searchIndex(self,query,k=3):
        D,I=self.INDEX.search(query,k)
        return I

    def _loadCandidates(self):
        with open(candidate_load_path,'rb') as f:
            candidate_mat=pickle.load(f)
        return candidate_mat

    def _update(self,candidate_load_path=None):
        self.candidate_mat=self._loadCandidates()

        # construct indexes.
        self.INDEX=faiss.IndexFlatL2(200)

        self.INDEX.add(self.candidate_mat)
        print("index reconstructed done.")

    def Index2WordEmbeddingBatch(self,I):
        shapes=I.shape
        if len(shapes==1):
            subtems=self.inducted_templates[I]
            k=shapes[0]
            embeds=np.zeros((k,self.msl,200))
            for i in range(k):
                embeds[k,:,:]=self.getSentWordEmbed(subtems[i],msl=self.msl)
        elif len(shapes==2):
            batch,k=shapes
            overall_embeds=np.zeros((batch,k,self.msl,200))
            for b in range(batch):
                subtems=self.inducted_templates[I[b]]
                embeds=np.zeros((k,self.msl,200))
                for i in range(k):
                    embeds[k,:,:]=self.getSentWordEmbed(subtems[i],msl=self.msl)
                overall_embeds[b,:,:,:]=embeds
            return overall_embeds
                

    def getwv(self,word):
        if word in self.corpus:
            return self.corpus[word]
        else:
            return np.zeros(200)

    def getSentWordEmbed(self,sentence,msl=128):
        sentence=self.deleteChar(sentence,"[")
        sentence=self.deleteChar(sentence,"]")
        embedding=np.zeros((msl,200))
        ss=sentence.split()
        for i in range(msl):
            if i<len(ss):
                embedding[i,:]=self.getwv(ss[i])
        return embedding

    def getAverageSentWordEmbed(self,sentencels):
        print(sentencels)
        embedding=np.zeros(200,dtype=np.float32)
        for w in sentencels:
            w=self.deleteChar(w,"[")
            w=self.deleteChar(w,"]")
            embed=self.getwv(w)
            embedding+=embed
        # print(embedding)
        if len(sentencels)!=0:
            return embedding/len(sentencels)
        else:
            return embedding

    def getAverageSentVector(self,sentence):
        sentence=self.deleteChar(sentence,"[")
        sentence=self.deleteChar(sentence,"]")
        embedding=np.zeros(200)
        ss=sentence.split()
        for word in ss:
            embedding+=self.getwv(word)
        return embedding

    def deleteChar(self,sent,c):
        if c in sent:
            ss=sent.split(c)
            result=""
            for ele in ss:
                result+=ele
            return result
        else:
            return sent
        
    def getCandidateMatrix(self,inducted_templates):
        v_list=[]
        for tem in inducted_templates:
            v_list.append(self.getAverageSentWordEmbed(tem))
        return np.array(v_list)


if __name__=="__main__":

    mymodel=retrievalModel()
    s1="今天天气很好，所以我想买合适的意外险，最好便宜一点的。"
    s2="天地不仁以万物为刍狗。圣人不仁以百姓为刍狗。这是意外险。"
    s3="一切有为法，如雾亦如电。所以这是重大疾病险。"
