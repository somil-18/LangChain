from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke and also print the exact joke before explaining \n {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation',
    max_new_tokens=150,
    temperature=0.4,
    huggingfacehub_api_token=TOKEN
)

chat = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# method 1
chain = RunnableSequence(prompt1, chat, parser, prompt2, chat, parser)

# method 2
# chain = prompt1 | chat | parser | prompt2 | chat | parser

result = chain.invoke({'topic': 'India'})
print(result)

