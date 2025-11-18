from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', dimensions=32)

documents = [
    "Delhi is the captial of India",
    "Shimla is the captial of HP",
    "Ayodhya is the captial of UP"
]

result = embedding.embed_documents(documents)

print(str(result))