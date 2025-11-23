from langchain_community.document_loaders import TextLoader

# Loaders for two separate text files
loader1 = TextLoader(r'DocumentLoader\crick_poem.txt', encoding='UTF-8')
loader2 = TextLoader(r'DocumentLoader\india_poem.txt', encoding='UTF-8')

# Each .load() returns a list containing ONE Document object
docs1 = loader1.load()    
docs2 = loader2.load()    

# Combine both lists into one list of Document objects
docs = docs1 + docs2      

# Print the entire list of Document objects
print(docs)
print('\n')

# Check the type of 'docs' (should be a Python list)
print(type(docs))
print('\n')

# Print the first Document object (cricket poem)
print(docs[0])
print('\n')

# Print the second Document object (India poem)
print(docs[1])
print('\n')

# Check the type of individual item (should be Document)
print(type(docs[0]))
print('\n')

# Metadata of first Document (shows file path)
print(docs[0].metadata)
print('\n')

# Content of first Document (actual text from crick_poem.txt)
print(docs[0].page_content)
print('\n')

# Metadata of second Document (shows file path)
print(docs[1].metadata)
print('\n')

# Content of second Document (actual text from india_poem.txt)
print(docs[1].page_content)
print('\n')
