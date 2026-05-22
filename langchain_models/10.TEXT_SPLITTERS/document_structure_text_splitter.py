# Some documents have an inherent structure, such as HTML, Markdown, or JSON files.
# In these cases, it’s beneficial to split the document based on its structure, as it often naturally groups semantically related text. 
# Key benefits of structure-based splitting:
#     Preserves the logical organization of the document
#     Maintains context within each chunk
#     Can be more effective for downstream tasks like retrieval or summarization
# Examples of structure-based splitting:
#     Markdown: Split based on headers (e.g., #, ##, ###)
#     HTML: Split using tags
#     JSON: Split by object or array elements
#     Code: Split by functions, classes, or logical blocks

# RecursiveCharacterTextSplitter includes prebuilt lists of separators that are useful for splitting text in a specific programming language.
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter as RCTS
)

# Supported languages are stored in the langchain_text_splitters.Language enum.
# To view the full list of supported languages:

# print([e.value for e in Language])

# To view the list of separators for a given language, pass a value from this enum into
# RecursiveCharacterTextSplitter.get_separators_for_language
print("List of separators for Python:\t", RCTS.get_separators_for_language(Language.PYTHON))


# Python Code Splitter Example:

# python_code = """
# def hellow_world():
#     print("Hello, World!")

# # Call the function
# hellow_world()
# """

# py_splitter = RCTS.from_language(
#     Language.PYTHON, 
#     chunk_size=30, 
#     chunk_overlap=0
# )

# py_chunks = py_splitter.split_text(python_code)
# print("Python code chunks:\n", py_chunks)

# py_docs = py_splitter.create_documents([python_code])
# print("Python code documents:\n", py_docs)

# Markdown Splitter Example:

print("List of separators for Markdown:\t", RCTS.get_separators_for_language(Language.MARKDOWN))

markdown_text = """
# 🦜️🔗 LangChain

⚡ Building applications with LLMs through composability ⚡

## What is LangChain?

# Hopefully this code block isn't split
LangChain is a framework for...

As an open-source project in a rapidly developing field, we are extremely open to contributions.
"""

md_splitter = RCTS.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
md_chunks = md_splitter.split_text(markdown_text)
print("Markdown chunks:\n", md_chunks)
md_docs = md_splitter.create_documents([markdown_text])
print("Markdown documents:\n", md_docs)