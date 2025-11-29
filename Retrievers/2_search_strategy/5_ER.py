import shutil 
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

doc_list = [
    "I like to eat apples and bananas.",              
    "The Apple iPhone 15 has a great camera.",        
    "Apples are high in fiber.",                      
    "Apple Inc. stock price rose today.",             
    "Fuji and Gala are types of apples."              
]

docs = [Document(page_content=d) for d in doc_list]

bm_retriever = BM25Retriever.from_documents(docs)
bm_retriever.k = 2

emb = HuggingFaceEmbeddings(model_name = 'BAAI/bge-small-en-v1.5')
vector_store = Chroma(
    embedding_function=emb,
    persist_directory=r'Retrievers/2_search_strategy/5_ER_VS',
    collection_name='col_1'
)
vector_store.add_documents(docs)

chroma_retriever = vector_store.as_retriever(search_kwargs = {'k': 2})

ensemble_ret = EnsembleRetriever(
    retrievers=[bm_retriever, chroma_retriever],
    weights=[0.5, 0.5] 
)

query = "gala"

print("--- CHROMA ONLY RESULTS ---")
C_docs = chroma_retriever.invoke(query)
for doc in C_docs:
    print(f"- {doc.page_content}")

print("\n--- ENSEMBLE RESULTS ---")
ER_docs = ensemble_ret.invoke(query)
for doc in ER_docs:
    print(f"- {doc.page_content}")

