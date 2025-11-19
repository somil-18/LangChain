from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
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

chat_history = []

while True:
    user_input = input("User: ")
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    result = chat.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ", result.content)

print(chat_history)

# chat_history is a list of messages but there are no lables like is this Human Message or AI Message or Sytem Message?


