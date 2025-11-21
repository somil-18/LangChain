# StructuredOutputParser is an output parser in LangChain that helps extract JSON data from LLM responses based on pre-defined schema

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

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

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give me 3 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    # partial_variables are not getting filled during runtime but before runtime
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | chat | parser

result = chain.invoke({'topic': 'AI'})

print(result)

'''
- No validations are there in StructuredOutputParser
- for e.g. i want age to int but somewhere i got 'age': '35 years' so there no validations 
- It's solution is PydanticOutputParser
'''

