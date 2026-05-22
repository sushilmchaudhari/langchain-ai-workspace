# PyPDF loader is used to load PDF files. 
# It uses the PyPDF library to read the PDF file and extract the text from it. 
# The PyPDFLoader will return a list of Document objects, where each Document object contains the content of the PDF file and its metadata. 
# Since we are loading a single PDF file, we will get a list with a single Document object. We can access the content of the PDF file using the page_content attribute of the Document object, and we can access the metadata using the metadata attribute of the Document object. 

# [
#     Document(page_content="Text content of Page 1 from the PDF file.", metadata={"page": 1, "source": "path/to/pdf/file.pdf"})
#     Document(page_content="Text content of Page 2 from the PDF file.", metadata={"page": 2, "source": "path/to/pdf/file.pdf"})
#     ...
#     Document(page_content="Text content of Page n from the PDF file.", metadata={"page": n, "source": "path/to/pdf/file.pdf"})
# ]

from langchain_community.document_loaders import PyPDFLoader
import os

# Get the directory of the current script and construct the path to report.pdf
script_dir = os.path.dirname(os.path.abspath(__file__))
report_path = os.path.join(script_dir, 'example_data', 'report.pdf')

loader = PyPDFLoader(report_path)

docs = loader.load()

# print(docs)

print(type(docs))

print("Length of documents:", len(docs))       # Number of pages in the PDF file

print("Type of the first document:", type(docs[0]))  # Document

print("First Document Object:", docs[0])
print("Content of the first document:", docs[0].page_content)
print("Metadata of the first document:", docs[0].metadata)

