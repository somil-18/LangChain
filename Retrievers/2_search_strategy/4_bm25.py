from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

docs = [
    Document(page_content="Flask is a Python web framework.", metadata={"tech": "Flask"}),
    Document(page_content="Django is a full-stack Python framework.", metadata={"tech": "Django"}),
    Document(page_content="FastAPI is built for speed.", metadata={"tech": "FastAPI"})
]

retriever = BM25Retriever.from_documents(
    docs,
    k=2
)

query = 'python'

result = retriever.invoke(query)

for r in result:
    print(f"Content: {r.page_content} | Metadata: {r.metadata}")

