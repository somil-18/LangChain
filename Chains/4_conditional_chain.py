from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv
import os


load_dotenv()

TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    huggingfacehub_api_token = TOKEN,
    task = 'text-generation',
    max_new_tokens = 150,
    temperature = 0.4
)

chat = ChatHuggingFace(llm=llm)


parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template = "You MUST output ONLY a valid JSON object that matches this schema.\n"
        "Do NOT explain anything. Do NOT add any extra text.\n"
        "{format_instruction}\n\n"
        "Classify the following text's sentiment as positive or negative \n {text}" ,
    input_variables=['text'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)


classifier_chain = prompt1 | chat | parser2


prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', prompt2 | chat | parser),
    (lambda x: x.sentiment == 'Negative', prompt3 | chat | parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)


chain = classifier_chain | branch_chain


result = chain.invoke({'text': 'This phone is very badand procesoor is very slow even slower then me'})

print(result)

