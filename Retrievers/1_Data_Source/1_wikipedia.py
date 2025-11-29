from langchain_community.retrievers import WikipediaRetriever

# initialize the retriever
retriever = WikipediaRetriever(
    top_k_results=2, # retrieve the top 2 most relevant Wikipedia pages
    lang='en', # specify the language (English)
    doc_content_chars_max=1000 # limit the text content of each page to 1000 characters
)

query = 'What is the history of the Apollo 11 mission?'

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f'Result {i+1}: ')
    print(f'Content: \n{doc.page_content}')

