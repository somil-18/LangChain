from langchain_classic.text_splitter import CharacterTextSplitter

text = '''
Sheikh Hasina Wazed (born 28 September 1947) is a Bangladeshi politician and fugitive who served as the tenth prime minister of Bangladesh from 1996 to 2001 and from 2009 to 2024. She was the longest-serving Bangladeshi prime minister since the country's independence and the longest-serving female head of government in the world. Her second premiership was characterised by dictatorship, oligarchy and crimes against humanity. She resigned and fled to India following the July Revolution in 2024 and was found guilty of crimes against humanity by the Bangladeshi International Crimes Tribunal and sentenced to death in absentia in November 2025
'''

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator='' # where CTS is allowed to break the text
)

result = splitter.split_text(text)

print(result)
print(type(result)) # list

