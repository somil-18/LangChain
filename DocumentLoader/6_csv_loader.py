# it treats each row as a langchain document object

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(r'DocumentLoader/s.csv')

docs = loader.load()

print(len(docs))

print(docs[0])

'''
- there are so many document loaders for different use cases
- We can also make our own custom documentloader!
'''