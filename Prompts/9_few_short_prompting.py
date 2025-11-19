from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
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

# EXAMPLES
examples = [
    {"question": "What is the capital of India?", "answer": "The capital of India is New Delhi."},
    {"question": "What is the capital of Spain?", "answer": "The capital of Spain is Madrid."},
    {"question": "What is the capital of China?", "answer": "The capital of China is Beijing."}
]

# TEMPLATE FOR EACH EXAMPLE
example_prompt = PromptTemplate(
    template="Question: {question}\nAnswer: {answer}",
    input_variables=["question", "answer"]
)

# FEW-SHOT TEMPLATE
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a helpful assistant. Answer the following question based on the examples.", # this is the instruction block placed BEFORE the examples, what you want the model to follow.
    suffix="Question: {question}\nAnswer:", # This part appears after the examples
    input_variables=["question"]
)

# LLM CHAIN
pipeline = few_shot_prompt | chat

result = pipeline.invoke({"question": "What is the capital of Japan?"})
print(result.content)

