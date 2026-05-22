# CSV Loader is used to load CSV files.
# It uses the csv library to read the CSV file and extract the text from it.
# The CSVLoader will return a list of Document objects, where each Document object contains the content of the CSV file and its metadata. 
# List of document objects will be returned where each document object will contain the content of one row of the CSV file and its metadata.
# We can access the content of the CSV file using the page_content attribute of the Document object, and we can access the metadata using the metadata attribute of the Document object.

from langchain_community.document_loaders import CSVLoader
import os

# Get the directory of the current script and construct the path to report.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
report_path = os.path.join(script_dir, 'example_data', 'report.csv')

loader = CSVLoader(report_path)
docs = loader.load()

print("Length of documents:", len(docs))       # Number of rows in the CSV file

print("Content of the first document:", docs[0].page_content)  # Content of the first row
print("Metadata of the first document:", docs[0].metadata)  # Metadata of the first row

# for i in range(len(docs)):
#     print(f"Content of document {i+1}:", docs[i].page_content)
#     print(f"Metadata of document {i+1}:", docs[i].metadata)

for doc in loader.lazy_load():
    print("Content of the document:", doc.page_content)
    print("Metadata of the document:", doc.metadata)    