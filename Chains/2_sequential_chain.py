from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

prompt1 = PromptTemplate(
    template = "Write a detailed explanation about {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Give 5 line summary of this text \n {text}",
    input_variables = ['text']
)

TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task = 'conversational',
    max_new_tokens=150,
    temperature=0.4,
    huggingfacehub_api_token=TOKEN
)

chat = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | chat | parser | prompt2 | chat | parser

result = chain.invoke({'topic': 'corruption in India'})

print(result)

