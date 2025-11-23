from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

loader1 = TextLoader(r"DocumentLoader\crick_poem.txt", encoding="UTF-8")

docs = loader1.load()            

prompt = PromptTemplate(
    template=(
        "Summarize the following poem clearly and concisely:\n\n"
        "{poem}"
    ),
    input_variables=["poem"]
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=TOKEN,
    max_new_tokens=150,
    temperature=0.4
)

chat = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | chat | parser

result = chain.invoke({"poem": docs[0].page_content})
print(result)

