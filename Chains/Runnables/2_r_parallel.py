from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

prompt1 = PromptTemplate(
    template='Write a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='write a linkedin post about \n {topic}',
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

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, chat, parser),
    'linkedin': RunnableSequence(prompt2, chat, parser)
})

print(parallel_chain.invoke({'topic': 'AI'}))

