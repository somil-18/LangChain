'''
1) "**/*" ---> all .txt files in all sub-folders
2) "*.pdf" ---> all .pdf in the root directory
3) "data/*.csv" ---> all .csv files in data folder
4) "**/*" --> all files (any type, all folders)

** ---> recursive search through sub-folders
'''

from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    path='DocumentLoader',
    glob='*.txt',
    loader_cls=TextLoader
)

docs = loader.load()

print(len(docs))
print('\n')

print(docs[1].page_content)
print('\n')

print(docs[0].metadata)
print('\n')

