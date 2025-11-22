# It turns a standard Python function (custom code) into a component that can fit inside a LangChain chain

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
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

joke_chain = RunnableSequence(prompt, chat, parser)

def word_counter(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_length': RunnableLambda(word_counter)
})

chain = RunnableSequence(joke_chain, parallel_chain)

result = chain.invoke({'topic': 'India'})
print(result)

