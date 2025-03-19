# retriever.py

from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever

from data import data

# 문서(요약) 리스트 생성
doc_list = [item['요약'] for item in data]

# BM25 Retriever 초기화
bm25_retriever = BM25Retriever.from_texts(
    doc_list, metadatas=[{"source": i} for i in range(len(data))]
)
bm25_retriever.k = 1

# SentenceTransformer 기반 FAISS Retriever 초기화
embedding = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": i} for i in range(len(data))]
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# BM25와 FAISS retriever를 앙상블 방식으로 결합
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

def search(query: str):
    """
    주어진 쿼리에 대해 앙상블 retriever를 사용해 관련 문서를 검색합니다.
    """
    ensemble_docs = ensemble_retriever.invoke(query)
    return ensemble_docs
