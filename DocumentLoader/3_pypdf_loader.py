from langchain_community.document_loaders import PyPDFLoader

# Load a PDF file (each page becomes a separate Document object)
loader = PyPDFLoader(r'DocumentLoader/dl.pdf')

# Load and convert PDF into a list of Document objects
docs = loader.load()

# Print the type of 'docs' (should be a Python list)
print(type(docs))
print('\n')

# Print how many Document objects were created (usually one per page)
print(len(docs))
print('\n')

# Print the first Document object (represents page 1)
print(docs[0])
print('\n')

# Print the second Document object (represents page 2)
print(docs[1])
print('\n')

# Metadata of page 1 (contains page number, source file, etc.)
print(docs[0].metadata)
print('\n')

# Actual text content extracted from page 1
print(docs[0].page_content)
print('\n')

# Metadata of page 2
print(docs[1].metadata)
print('\n')

# Actual text content extracted from page 2
print(docs[1].page_content)
print('\n')

'''
- It works perfectly for simple, text-based PDFs, meaning:
    PDFs that actually contain digital text
    PDFs exported from Word/Docs
    PDFs with selectable text
    PDFs with straightforward formatting

- But not for complex documents
'''

