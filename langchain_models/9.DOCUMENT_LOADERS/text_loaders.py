
# TextLoader is used to load text files.
# It takes the file path as input and returns the content of the file as a string. 
# It also has an option to split the content into chunks based on the specified chunk size and chunk overlap. This is useful when we want to process large text files that cannot be loaded into memory at once. We can specify the chunk size and chunk overlap to control how the text is split into chunks.
# In this example, we are going to load a text file and analyze the document object returned by the TextLoader.

# document_object = {
#     "page_content": "This is the content of the text file.",
#     "metadata": {
#         "source": "path/to/text/file.txt",
#     }
# }


from langchain_community.document_loaders import TextLoader
import os

# Get the directory of the current script and construct the path to report.txt
script_dir = os.path.dirname(os.path.abspath(__file__))
report_path = os.path.join(script_dir, 'example_data', 'report.txt')



# We are loading the report.txt file using Textloader. The encoding is set to utf-8 to ensure that the text is properly decoded. The TextLoader will return a list of Document objects, where each Document object contains the content of the text file and its metadata. Since we are loading a single text file, we will get a list with a single Document object. We can access the content of the text file using the page_content attribute of the Document object, and we can access the metadata using the metadata attribute of the Document object.
loader = TextLoader(report_path, encoding='utf-8')

# load() method is used to load the text file and return a list of Document objects. Each Document object contains the content of the text file and its metadata.
documents = loader.load()

# Print the content of the text file and its metadata.
print(documents)

# The type of the documents variable is a list of Document objects.
print(type((documents)))

# Since we are loading a single text file, we will get a list with a single Document object. 
print("Length of documents:", len(documents))       # 1
print("Type of the first document:", type(documents[0]))  # Document

# We can access the content of the text file using the page_content attribute of the Document object, and we can access the metadata using the metadata attribute of the Document object.
print("Content of the document:", documents[0].page_content)
print("Metadata of the document:", documents[0].metadata)

# We can use this document object to process the text file further using other components of Langchain, such as LLMs, chains, etc. For example, we can use the content of the document to generate a summary using an LLM, or we can use the metadata to keep track of the source of the information when we are processing multiple documents.

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

str_parser = StrOutputParser() 

prompt = PromptTemplate(
    template="Summarize the following report in less than 100 words: \n {report}",
    input_variables=["report"]
)

summary = prompt | llm | str_parser

result = summary.invoke({"report": documents[0].page_content})

print("============== Summary of the report:=================\n", result)


# # ========== EXAMPLES OF USING RunnableLambda WITH DOCUMENT LOADERS ==========

# # Example 1: Extract word count using RunnableLambda
# extract_word_count = RunnableLambda(lambda doc: {"text": doc.page_content, "word_count": len(doc.page_content.split())})

# word_count_result = extract_word_count.invoke(documents[0])
# print("\n--- Example 1: Extract Word Count ---")
# print(f"Word count: {word_count_result['word_count']}")


# # Example 2: Transform document to uppercase using RunnableLambda
# uppercase_transformer = RunnableLambda(lambda doc: doc.page_content.upper())

# uppercase_result = uppercase_transformer.invoke(documents[0])
# print("\n--- Example 2: Transform to Uppercase ---")
# print(f"Uppercase text (first 100 chars): {uppercase_result[:100]}")


# # Example 3: Pipeline with RunnableLambda - Extract first N words
# extract_first_n_words = RunnableLambda(lambda doc: " ".join(doc.page_content.split()[:50]))

# # Chain: Load document -> Extract first 50 words -> Summarize
# pipeline = RunnableLambda(lambda doc: doc.page_content) | RunnableLambda(lambda text: " ".join(text.split()[:50])) | prompt | llm | str_parser

# print("\n--- Example 3: Pipeline - First 50 words summarized ---")
# # pipeline.invoke(documents[0])  # Uncomment to run


# # Example 4: Log metadata and filter documents
# log_and_filter = RunnableLambda(
#     lambda doc: (
#         print(f"Processing document from: {doc.metadata['source']}"),
#         doc
#     )[1]
# )

# print("\n--- Example 4: Log and Filter ---")
# filtered_doc = log_and_filter.invoke(documents[0])


# # Example 5: Conditional processing with RunnableLambda
# check_length = RunnableLambda(
#     lambda doc: {
#         "content": doc.page_content,
#         "length": len(doc.page_content),
#         "is_long": len(doc.page_content) > 1000,
#         "category": "Long document" if len(doc.page_content) > 1000 else "Short document"
#     }
# )

# print("\n--- Example 5: Conditional Document Processing ---")
# categorized = check_length.invoke(documents[0])
# print(f"Document category: {categorized['category']}")
# print(f"Is long (>1000 chars): {categorized['is_long']}")