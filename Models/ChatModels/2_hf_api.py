from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

'''
HuggingFaceEndpoint --> creates a connection to a specific model hosted on HuggingFace Hub
ChatHuggingFace --> wraps that connection so it behaves like a chat model, handling conversational input/output
'''

load_dotenv()

TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational", # this model expects a conversation style input and will respond like a chatbot.
    max_new_tokens=150,
    temperature=0.3,
    huggingfacehub_api_token=TOKEN
)

chat = ChatHuggingFace(llm=llm)

result = chat.invoke("What is the capital of India?")
print(result.content)

'''
** for line 22 and line 24 **

- We send a single user message
- It wraps it inside a chat format internally:
    [{"role": "user", "content": "What is the capital of India?"}]
- The instruct model responds like a ChatGPT-style assistant.
'''
