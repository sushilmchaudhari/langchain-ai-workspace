# The DuckDuckGoSearchRun tool allows you to perform web searches using the DuckDuckGo search engine.
# It provides a simple interface to query DuckDuckGo and retrieve search results, which can be useful for various applications such as information retrieval, question-answering, and content discovery.

from langchain_community.tools import DuckDuckGoSearchRun

# Create an instance of the DuckDuckGoSearchRun tool. This tool will allow us to perform web searches using the DuckDuckGo search engine.

search_tool = DuckDuckGoSearchRun()

# Use the search tool to perform a web search. The invoke method takes a query string and returns the search results from DuckDuckGo.

query = "What is the capital of India?"

search_results = search_tool.invoke(query)

print("Search results from DuckDuckGo:\n", search_results)

print("\nNumber of search results retrieved:", len(search_results))

# Print the name of the tool used for searching. The name attribute of the search_tool instance provides the name of the tool, which in this case is "DuckDuckGoSearchRun". This can be useful for logging or debugging purposes to identify which tool was used for the search.
print("\n Name of the tool:", search_tool.name)

# Print the description of the tool. The description attribute of the search_tool instance provides a brief description of what the tool does, which can be helpful for understanding its functionality and purpose.
print("\n Description of the tool:", search_tool.description)

# Arguments used for the search. The args attribute of the search_tool instance contains the arguments that were passed to the invoke method when performing the search. This can be useful for tracking the parameters used in the search and for debugging purposes.
print("\n Arguments used for the search:", search_tool.args)

# Schema details received by LLM. The schema attribute of the search_tool instance provides details about the schema of the tool, including the input and output formats. This can be helpful for understanding how to use the tool correctly and for integrating it with other components in a larger system.
print("\n Schema details received by LLM:", search_tool.args_schema.model_json_schema())


