# Contextual Compression Retriever is a technique used in information retrieval to improve the relevance of retrieved documents by compressing the context of the query.
# The idea is to identify and retain only the most relevant parts of the query context while discarding less relevant information. This can help enhance the retrieval process by focusing on the key aspects of the query, leading to more accurate and relevant results. 
# The contextual compression retriever works by analyzing the query and its context, and then applying compression techniques to reduce the amount of information while preserving the essential meaning. 
# This can be particularly useful in scenarios where the query context is lengthy or contains extraneous information that may not contribute to the retrieval of relevant documents. 
# By compressing the context, the retriever can better match the query with relevant documents in the database, improving the overall retrieval performance. 
# This technique can be implemented in various applications, such as search engines, question-answering systems, and recommendation systems, to enhance the retrieval of relevant information and improve user satisfaction.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
try:
    # LangChain 1.x
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    try:
        # Older LangChain re-export layout
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
    except ImportError:
        # Older submodule layout
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor

from dotenv import load_dotenv

# Load the OPENAI API key from the .evn file.
load_dotenv()

# Document Creation
# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# Create a FAISS vector store from the documents
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)

# Base retriever to fetch relevant documents based on similarity search
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Compression retriever that uses an LLM to extract the most relevant parts of the retrieved documents based on the query context
llm = ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

# Create an instance of the ContextualCompressionRetriever, which combines the base retriever and the compressor to retrieve relevant documents and compress their context based on the query.

ccr = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

# Query the retriever with a specific query to retrieve relevant documents and compress their context based on the query.
query = "What is photosynthesis and how does it relate to plants?"

relevant_docs = ccr.invoke(query)

# Print the relevant documents retrieved by the Contextual Compression Retriever, along with their metadata.
for i, doc in enumerate(relevant_docs):
    print(f"Relevant document {i+1}:\n", doc.page_content)
    print("Metadata:", doc.metadata)

