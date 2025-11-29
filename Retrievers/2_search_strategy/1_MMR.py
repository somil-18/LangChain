from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# source documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    embedding_function=emb, 
    persist_directory=r'Retrievers/2_search_strategy/1_chroma_Db',
    collection_name='collec_1'
)

vector_store.add_documents(docs)

# Enable MMR in the retriever
retriever = vector_store.as_retriever(
    search_type="mmr", # this enables MMR
    search_kwargs={
        "k": 3, 
        "lambda_mult": 0.5,
        
    }  
)

'''
1) fetch_k: 5 ---> retriever first finds the 5 most relevant documents, the total number of documents that the system will initially fetch from the vector store based only on relevance

2) k:3 ---> final number of documents that the MMR algorithm will select from the fetch_k pool

3) lambda_mult ---> A float value (between 0.0 and 1.0) that determines the balance between relevance to the query and diversity from the already selected documents
1.0: Pure Relevance. The retriever acts like a standard similarity search (diversity is ignored).
0.0: Pure Diversity. The retriever will pick documents that are as different as possible, even if they are only mildly relevant.
0.5 - 0.8 (Typical Range): Balance. This is the sweet spot, favoring relevance while ensuring diversity.
'''

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

