# It takes the input it receives and passes it to the next step exactly as it is. It creates no changes

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke \n {text}',
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

joke_chain = RunnableSequence(prompt1, chat, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, chat, parser)
})

chain = RunnableSequence(joke_chain, parallel_chain)

result = chain.invoke({'topic': 'India'})
print(result)

