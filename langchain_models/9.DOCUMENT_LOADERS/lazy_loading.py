
# Lazy loading is a technique where the documents are not loaded into memory until they are actually needed. 
# This can be useful when dealing with large documents or when we want to load documents on demand. 

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os
import time

# Get the directory of the current script and construct the path to the directory containing the files to be loaded
script_dir = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(script_dir, 'example_data')

# We are loading all the pdf files in the example_data directory using DirectoryLoader. The glob parameter is used to specify the file extension to load only the pdf files.

pdf_loader = DirectoryLoader(
    directory_path, 
    glob="*.pdf",
    show_progress=True,
    loader_cls=PyPDFLoader,
)

# Normal loading of the pdf files using load() method. This will load all the pdf files in the directory into memory at once and return a list of Document objects.

# Measure total time for a loop over normal loaded documents.
start_total = time.perf_counter()
print("Time Start: Load(): ", start_total)
pdf_docs = pdf_loader.load()
for i, doc in enumerate(pdf_docs, start=1):
    start_iter = time.perf_counter()

    # print metadata of the document
    print(f"Metadata of document {i}:", doc.metadata)

    end_iter = time.perf_counter()    

end_total = time.perf_counter()
print(f"Total loop time load: {end_total - start_total:.6f} seconds")

# Measure total time for a loop over lazily loaded documents.
start_total = time.perf_counter()
print("Time Start: Lazy Load(): ", start_total)
pdf_docs = pdf_loader.lazy_load()
for i, doc in enumerate(pdf_docs, start=1):
    start_iter = time.perf_counter()

    # print metadata of the document
    print(f"Metadata of document {i}:", doc.metadata)

    end_iter = time.perf_counter()    

end_total = time.perf_counter()
print(f"Total loop time lazy load: {end_total - start_total:.6f} seconds")

