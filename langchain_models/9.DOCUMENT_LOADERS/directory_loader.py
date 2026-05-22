# Directory Loader is used to load all the files in a directory. 
# It is a convenient way to load multiple files at once. 
# We can specify the directory path and the file extension to load only the files with the specified extension. 
# The DirectoryLoader will return a list of Document objects, where each Document object contains the content of the file and its metadata. 
# We can access the content of the file using the page_content attribute of the Document object, and we can access the metadata using the metadata attribute of the Document object.

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
import os

# Get the directory of the current script and construct the path to the directory containing the files to be loaded
script_dir = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(script_dir, 'example_data')

# We are loading all the pdf files in the example_data directory using DirectoryLoader. The glob parameter is used to specify the file extension to load only the pdf files.

# pdf_loader = DirectoryLoader(
#     directory_path, 
#     glob="*.pdf",
#     show_progress=True,
#     loader_cls=PyPDFLoader,
# )
#
# pdf_docs = pdf_loader.load()

# print("Length of documents:\n", len(pdf_docs))       # Number of pdf files in the directory

# print(pdf_docs[0].metadata)

# We are loading all the txt files in the example_data directory using DirectoryLoader. The glob parameter is used to specify the file extension to load only the txt files.

txt_loader = DirectoryLoader(
    directory_path, 
    glob="*.txt",
    show_progress=True,
    loader_cls=TextLoader,
)

txt_docs = txt_loader.load()

print("Length of documents:\n", len(txt_docs))       # Number of txt files in the directory

print(txt_docs[0].metadata)
print(txt_docs[1].metadata)
