'''
- Why to parse string with str output parser as we can already parse string with .content?

- Work flow of this code:-

topic --> LLM --> detailed report --> LLM --> 5 line summary of detailed report
'''

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

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

prompt1 = template1.invoke({'topic': 'black hole'})
result1 = chat.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})
result2 = chat.invoke(prompt2)

print(result2.content)

