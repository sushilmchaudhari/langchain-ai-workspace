# Maximal Marginal Relevance (MMR) Retriever is a technique used in information retrieval to select a subset of documents that are both relevant to a query and diverse from each other. 
# The MMR algorithm balances relevance and diversity by selecting documents that are relevant to the query while also being different from the already selected documents. 
# This can help improve the quality of search results by providing a more diverse set of relevant documents. 

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Create an instance of the OpenAIEmbeddings class to generate vector embeddings for our documents.

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# List of document objects to be added to the vector store. Each document object contains the content of the document and its metadata. The content of the document is stored in the page_content attribute, and the metadata is stored in the metadata attribute.

documents = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)

# Enable MMR in the retriever
retriever = vector_store.as_retriever(
    search_type="mmr",                   # This search_type or search strategy specifies that we want to use Maximal Marginal Relevance for retrieving documents.
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = balance between relevance and diversity (0.5 means equal weight). The value of lambda_mult can be adjusted to give more weight to relevance (closer to 1) or diversity (closer to 0). 1 means only relevance, 0 means only diversity.
)

results = retriever.invoke("What is LangChain?")

for i, doc in enumerate(results):
    print(f"Relevant document {i+1}:\n", doc.page_content)