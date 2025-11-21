from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.3,
    huggingfacehub_api_token=TOKEN
)

chat = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str
    age: int = Field(gt=18)
    city: str

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "You MUST output ONLY a valid JSON object that matches this schema.\n"
        "Do NOT explain anything. Do NOT add any extra text.\n"
        "{format_instruction}\n\n"
        "Generate the name, age and city of a fictional {place} person."
    ),
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | chat | parser

final_result = chain.invoke({"place": "indian"})

print(final_result)
print(final_result.city)
