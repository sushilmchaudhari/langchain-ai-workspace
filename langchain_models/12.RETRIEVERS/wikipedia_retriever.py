# Wikipedia retriever is a tool that allows you to retrieve information from Wikipedia articles. 
# It uses the Wikipedia API to search for articles and extract relevant information based on your query. 
# This can be useful for various applications, such as answering questions, summarizing content, or providing background information on a topic.

from langchain_community.retrievers import WikipediaRetriever

# Create an instance of the WikipediaRetriever.
# The WikipediaRetriever class provides methods to search for articles and retrieve information from Wikipedia.

retriever = WikipediaRetriever(top_k_results=5, lang="en")  # Set the number of top results to retrieve and the language

query = "What is Epstin files?"

# Use the retriever to search for articles related to the query. The invoke method takes a query string and returns a list of relevant documents from Wikipedia.

relevant_docs = retriever.invoke(query)

print("Relevant documents retrieved from Wikipedia:\n", relevant_docs)

print(relevant_docs[0].page_content)  # Print the content of the first retrieved document
print(relevant_docs[0].metadata)  # Print the metadata of the first retrieved document
# print("\nNumber of relevant documents retrieved:", len(relevant_docs))