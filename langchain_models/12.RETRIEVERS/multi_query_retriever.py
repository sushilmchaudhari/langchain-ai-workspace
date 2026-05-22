# The Multi Query Retriever is a technique used in information retrieval to enhance the retrieval process by using multiple queries instead of a single query. 
# This approach can help improve the relevance of the retrieved documents by capturing different aspects of the user's information need. 
# The Multi Query Retriever works by generating multiple queries based on the original query and then retrieving documents for each of these queries. The retrieved documents are then combined and ranked based on their relevance to the original query. 
# This can help provide a more comprehensive set of relevant documents to the user.
# This technique is particularly useful when the original query is ambiguous or when the user's information need is complex and cannot be captured by a single query. 
# This can be especially beneficial in scenarios like question-answering systems, where the user's query may have multiple interpretations or when the information need is multifaceted. This can be implemented in various applications, such as search engines, chatbots, and recommendation systems, to enhance the retrieval of relevant information and improve user satisfaction.
# This technnique uses llm to generate multiple queries from the original query and then retrieves documents for each of these generated queries. The retrieved documents are then combined and ranked based on their relevance to the original query, providing a more comprehensive set of relevant documents to the user. 
# By using multiple queries, the retriever can capture different aspects of the user's information need and provide more accurate and relevant results.  


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
try:
    # LangChain 1.x
    from langchain_classic.retrievers import MultiQueryRetriever
except ImportError:
    try:
        # Older LangChain versions that re-export retrievers
        from langchain.retrievers import MultiQueryRetriever
    except ImportError:
        # Older layouts where MultiQueryRetriever lives in a submodule
        from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()   

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Create FAISS vector store
vector_store = FAISS.from_documents(documents=all_docs, embedding=embedding_model)

# Create retrievers
similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo") # Using LLM to generate multiple queries from the original query. The retriever will then retrieve documents fr each of these generated queries and combine the results.
)

# Query:

query = "How to improve energy levels and maintain balance?"

# Retrieve relevant documents using the multi-query retriever. The invoke method takes a query string and returns a list of relevant documents based on the multiple queries generated from the original query.

similarity_search_results = similarity_retriever.invoke(query)
print("Relevant documents retrieved using Similarity Retriever:\n")
for i, doc in enumerate(similarity_search_results):
    print(f"Relevant document {i+1}:\n", doc.page_content)
    print("Metadata:", doc.metadata)

multiquery_results = multiquery_retriever.invoke(query)

print("Relevant documents retrieved using Multi Query Retriever:\n")
for i, doc1 in enumerate(multiquery_results):
    print(f"Relevant document {i+1}:\n", doc1.page_content)
    print("Metadata:", doc1.metadata)