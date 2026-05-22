# RecursiveCharacterTextSplitter is a text splitter that splits text into chunks based on a list of separators. 
# It tries to split the text using the first separator in the list, and if the resulting chunks are too long, it will try to split the text using the next separator in the list, and so on until it finds a separator that can split the text into chunks of the desired size. 
# If none of the separators can split the text into chunks of the desired size, it will split the text into chunks of the specified size using the last separator in the list. 
# The RecursiveCharacterTextSplitter is useful when we want to split text into chunks based on natural language boundaries, such as paragraphs, sentences, or words, while also ensuring that the chunks are not too long for processing by language models.
# List of sepators = ["\n\n", "\n", " ", ""]


from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load example document
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "state_of_the_union.txt")
with open(file_path, encoding="utf-8") as f:
    state_of_the_union = f.read()

# print("Type of the document:", type(state_of_the_union))  # str

text_splitter1 = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.    
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    
)

text_splitter2 = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.    
    separators="\n\n",  # Split on double newlines first
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    
)

text_splitter3 = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0
)

# Create documents using the create_documents method of the text splitter. 
# The create_documents method takes a list of strings as input and returns a list of Document objects, where each Document object contains a chunk of the original text and its metadata.
texts1 = text_splitter1.create_documents([state_of_the_union])
print("Length of texts: ", len(texts1))  # Number of chunks created from the text
print("texts is: \n", texts1)
print("texts[0]:", texts1[0])
print("texts[1]:", texts1[1])


texts2 = text_splitter2.split_text(state_of_the_union)
print("Length of texts: ", len(texts2))  # Number of chunks created from the text
print("texts is: \n", texts2)

text3 = """My name is Sushil
I am 39 yrs old

I live in Bengaluru
How are you
"""

texts3 = text_splitter3.split_text(text3)
print("Length of texts3:", len(texts3))  # Number of chunks created from the text
print("texts3 is: \n", texts3)


