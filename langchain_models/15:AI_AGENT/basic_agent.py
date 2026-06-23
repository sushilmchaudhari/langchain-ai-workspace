# This is creating a simple agent that can use the DuckDuckGoSearchRun tool to perform web searches to get the information.
# The agent will take a query as input, use the DuckDuckGoSearchRun tool to perform a web search, and return the search results.


from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create an instance of the DuckDuckGoSearchRun tool. This tool will allow us to perform web searches using the DuckDuckGo search engine.
search_tool = DuckDuckGoSearchRun()

# Crate an instance of the ChatOpenAI model. This will be used as the language model for our agent to process the input and generate responses.
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create an agent using the language model and the search tool.
agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="You are a helpful assistant. Use tools when needed and provide concise answers.",
)

# Run the agent with a query. The agent will process the query, use the search tool to perform a web search, and return the search results.
query = "What is the capital of India?"

response = agent.invoke({"messages": [("user", query)]})
print(response)
print(response["messages"][-1].content)
