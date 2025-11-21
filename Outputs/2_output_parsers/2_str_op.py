from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=150,
    temperature=0.3,
    huggingfacehub_api_token=TOKEN
)

chat = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write in detail about {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write 5 line summary of this text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

# concept of chaining
chain = template1 | chat | parser | template2 | chat | parser

result = chain.invoke({"topic": "black hole"})
print(result)

