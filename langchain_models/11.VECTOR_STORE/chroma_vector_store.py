# Chroma DB Vector Store is a vector database that allows you to store and query high-dimensional vectors efficiently. 
# It is designed to work with large datasets and provides fast similarity search capabilities. 
# In this example, we will see how to use Chroma DB Vector Store with LangChain to store and query vector embeddings.

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

# Create an instance of the OpenAIEmbeddings class to generate vector embeddings for our documents. 
# We will use the "text-embedding-3-small" model to generate 384-dimensional embeddings for our text data.
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Create an instance of the Chroma vector store. We will specify the collection name as "my_collection" and the embedding model we created earlier.
vector_store = Chroma(
    embedding_function=embedding_model,  # Function to generate embeddings for queries
    collection_name="my_collection",
    persist_directory="./chroma_db"  # Directory to store the Chroma database files
)

# List of document objects to be added to the vector store. Each document object contains the content of the document and its metadata. The content of the document is stored in the page_content attribute, and the metadata is stored in the metadata attribute.

docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
]
 
# Add the documents to the Chroma vector store. 
# The add_documents method takes a list of Document objects and adds them to the vector store. It generates embeddings for the content of each document using the embedding model and stores them in the Chroma database along with the metadata.

# vector_documents = vector_store.add_documents(docs)
# print("Vector documents added:\n", vector_documents)

# View the number of documents in the vector store. The get method returns the , and the count method returns the number of documents in the collection.

# # vector_data = vector_store.get(include=["metadatas", "documents", "embeddings"])
vector_data = vector_store.get(include=["documents"])
print("Documents in the vector store:\n", vector_data)
print("Number of documents in the vector store:", len(vector_data["documents"]))

# # Search for similar documents in the vector store using a query. 
# # The similarity_search method takes a query string and returns a list of documents that are similar to the query based on their vector embeddings. 
# # We can specify the number of similar documents to return using the k parameter.

# query = "Who is the best captain in IPL history?"
# similar_docs = vector_store.similarity_search(
#     query=query, 
#     k=2 # Number of similar documents to return
# )
# print("Similar documents to the query:\n", similar_docs)

# # Search for similar documents with similarity scores. The similarity_search_with_score method returns a list of tuples, where each tuple contains a document and its similarity score to the query.

# similar_docs_with_scores = vector_store.similarity_search_with_score(
#     query=query, 
#     k=2 # Number of similar documents to return
# )
# print("======= Similar documents with scores:\n", similar_docs_with_scores)

# # Search with metadata filter. The similarity_search method also accepts a filter parameter that allows us to filter the documents based on their metadata before performing the similarity search. For example, we can filter the documents to only include those that belong to a specific team.

# metadata_filter = {"team": "Mumbai Indians"}
# similar_docs_filtered = vector_store.similarity_search(
#     query="",     
#     filter=metadata_filter,
#     k=8,
# )

# print("======= Similar documents with metadata filter:\n", similar_docs_filtered)

# Deleting the documents from the vector store. 
# The delete method takes a list of document IDs and deletes the corresponding documents from the vector store. We can get the document IDs from the vector_data we retrieved earlier.

# vector_store.delete(
#     ids=['79af17bc-f42e-4c87-b8f7-765384a329e7', '20bd5e74-33a4-491f-8d3a-f7f00ba28693', '2e81a38f-8074-4266-b6f3-5fa1a1dfa027', 'e19ad049-9cdf-4c56-8710-33e34e708195', 'a59e46d1-1a6b-4bd7-8593-0d2c7319f999', '58eecb3b-95be-4f24-bf2f-6a27df4c1fa7', 'efc77757-8ae8-4dc6-8315-e5802858b7a9', '007acc34-031d-41e2-9490-0c05e80bec8f', '2d112ed3-ebfa-4a88-9e7b-0a548afb8fef', 'ba3ccd0e-6d4b-436c-8595-1ea8db62fc7f', '1311d0fe-de1b-46a2-a456-9e2f13c08c95', '4c6b75ad-3d84-4b89-a4b8-d525d69a09f1', '78b42609-b0df-4faf-b3c6-615068b09a6b', '11141989-aad7-4811-823b-eb0c957dfc06', '90f580bf-44aa-476e-b807-77fa47ca13b3'
#     ]
# )

vector_data = vector_store.get(include=["documents"])
print("==================AFTER DELETE===============================")
print("Documents in the vector store:\n", vector_data)
print("Number of documents in the vector store:", len(vector_data["documents"]))

# Updating the documents in the vector store. 
# The update method takes a list of Document objects and updates the corresponding documents in the vector store based on their IDs. We can get the document IDs from the vector_data we retrieved earlier.

updated_doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons and has been a key player for the team.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(
    document_id='e15a31e9-837d-4716-b476-dcc97bc520fd', 
    document=updated_doc1 
)

vector_data = vector_store.get(include=["documents"])
print("=================AFTER UPDATE============================")
print("Documents in the vector store:\n", vector_data)
print("Number of documents in the vector store:", len(vector_data["documents"]))

