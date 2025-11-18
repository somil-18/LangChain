from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))

# can also do the same for documents as earlier done with google api