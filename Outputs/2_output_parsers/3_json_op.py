from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

from langchain_core.output_parsers import JsonOutputParser

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

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a ficitional person \n {format_instruction}",
    input_variables=[],
    # partial_variables are not getting filled during runtime but before runtime
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# concept of chaining
chain = template | chat | parser

result = chain.invoke({})

print(result)
print(type(result)) # dict

'''
- We can't decide the structure of the JSON format for e.g. i need 5 facts about ML and I need facts as keys
- So basically we can't decide the structure of JSON format we want
- It's solution is StructuredOutputParser
'''

