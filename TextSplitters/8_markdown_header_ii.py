'''
- MarkdownHeaderTextSplitter does not have 'split_documents' method, it only has 'split_text'
- we can't simply pass the list Document objects to it
- we have to send the Document objects one by one
'''

from langchain_core.documents import Document
from langchain_classic.text_splitter import MarkdownHeaderTextSplitter

existing_docs = [
    Document(page_content="# Intro \n Hi \n ## Greet \n How are you", metadata={"source": "file1.md"}),
    Document(page_content="# Mid \n Body", metadata={"source": "file2.md"}),
    Document(page_content="# Outro \n Bye", metadata={"source": "file3.md"})
]

headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2")
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# the loop and merge logic
final_docs = []
for d in existing_docs:
    '''
    - split the individual documents
    - 'split_text' takes string (raw text)
    - it returns a List of Document Object
    '''
    splits = md_splitter.split_text(d.page_content)

    # without this, original metadata of Document objects will not be captured
    for s in splits:
        s.metadata.update(d.metadata)

    final_docs.extend(splits)

print(len(existing_docs))
print(len(final_docs))

print(final_docs)

