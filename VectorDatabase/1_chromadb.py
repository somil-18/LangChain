from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# create IPL document objects
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"name": "Virat Kohli", "team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"name": "Rohit Sharma", "team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"name": "MS Dhoni", "team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"name": "Jasprit Bumrah", "team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"name": "Ravindra Jadeja", "team": "Chennai Super Kings"}
    )
]


# setup embeddings and vector store
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# it creates the Chroma vector database object
vector_store = Chroma(
    embedding_function=emb, # embedding model
    persist_directory=r'VectorDatabase/chroma_wala_Db', # WHERE to store the database on disk, if not given then Db is temporary (RAM)
    collection_name='sample' # collection
)


# add documents ---> stored as embedding + metadata + text
vector_store.add_documents(docs)
print("\n📌 Documents inserted successfully!")
print("\n📌 Current vector database entries:")
print(vector_store.get(include=['documents', 'metadatas']))
'''
1) documents - the stored text inside each document
2) metadatas - the metadata dictionary
3) embeddings - the vector embeddings
4) ids - unique db document ids
5) uris - file URIs (only when documents are from files)
'''

# similarity search
query = "Who among these are a bowler?"
print(f"\n🔍 Similarity search for query: {query}")
res = vector_store.similarity_search(query=query, k=2)
print(res)

# similarity search with scores
print("\n🔍 Similarity search with scores:")
res_score = vector_store.similarity_search_with_score(query=query, k=2)
print(res_score)

# metadata-based filtering
print("\n🔍 Filtering team = Chennai Super Kings")
filtered = vector_store.similarity_search_with_score(
    query="",
    filter={'team': 'Chennai Super Kings'}
)
print(filtered)

# update document (auto-detect, NOT hard-coded ID)
print("\n Updating document for Virat Kohli ...")
all_docs = vector_store.get(include=['documents', 'metadatas'])

virat_id = None
for i, meta in enumerate(all_docs['metadatas']):
    if meta.get('name') == "Virat Kohli":
        virat_id = all_docs['ids'][i]
        break

if virat_id:
    updated_doc = Document(
        page_content="Virat Kohli is the highest run-scorer in IPL history. Known for his fierce captaincy and unmatched consistency, he remains one of the most reliable T20 batsmen ever.",
        metadata={"name": "Virat Kohli", "team": "Royal Challengers Bangalore"}
    )
    vector_store.update_document(document_id=virat_id, document=updated_doc)
    print("✔ Virat Kohli document updated successfully!")
else:
    print("❌ Virat document not found. Update skipped.")

print("\n🧾 DB after update:")
print(vector_store.get(include=['documents', 'metadatas']))


# delete a document (auto-detect by name)
print("\n🗑 Deleting document for Rohit Sharma ...")
all_docs = vector_store.get(include=['documents', 'metadatas'])

rohit_id = None
for i, meta in enumerate(all_docs['metadatas']):
    if meta.get('name') == "Rohit Sharma":
        rohit_id = all_docs['ids'][i]
        break

if rohit_id:
    vector_store.delete(ids=[rohit_id])
    print("✔ Rohit Sharma document deleted!")
else:
    print("❌ Rohit document not found. Delete skipped.")

print("\n🧾 Final DB state:")
print(vector_store.get(include=['documents', 'metadatas']))

