from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

'''
chat_template = ChatPromptTemplate(
    [
        SystemMessage(content="You are a helpful {domain} expert"),
        HumanMessage(content="Explain in simple terms, what is {topic}?")
    ]
)

this method is invalid with the new LangChain versions or maybe with old versions also, don't know
'''

chat_template = ChatPromptTemplate(
    [
        ('system', 'You are a helpful {domain} expert'),
        ('human', 'Explain in simple terms, what is {topic}?')
    ]
)

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'legspin'})

print(prompt)
