from langchain_classic.text_splitter import MarkdownHeaderTextSplitter

markdown_document = """
# Backend Development
This section covers server-side logic.

## Database
We use PostgreSQL for relational data.

## API
We use Flask for our API endpoints.

# Frontend Development
This section covers the user interface.

## Frameworks
React is our primary library.
"""

# define headers we wana split
header_splitters = [
    ('#', 'Header1'),
    ('##', 'Header2')
]

# initialize the splitter
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=header_splitters)

# split the text, it takes string (raw text)
split = splitter.split_text(markdown_document)

print(split)
