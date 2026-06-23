# This is creating a simple agent that can use the DuckDuckGoSearchRun tool to perform web searches to get the information.
# The agent will take a query as input, use the DuckDuckGoSearchRun tool to perform a web search, and return the search results.
# This uses old LangChain agent creation method. The new method is to use the create_agent function from langchain.agents module.
# NOTE: This script intentionally uses deprecated LangChain Classic methods (AgentExecutor/create_react_agent and classic prompt flow) for legacy-learning purposes.


import os

from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor, tool
from langchain_community.tools import DuckDuckGoSearchRun
from langsmith import Client
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Create an instance of the DuckDuckGoSearchRun tool. This tool will allow us to perform web searches using the DuckDuckGo search engine.
search_tool = DuckDuckGoSearchRun()

@tool("get_current_weather")
def get_current_weather(city: str) -> str:
    """
    Get the current weather in a given city.
    """

    url = f"https://api.weatherstack.com/current?access_key={os.getenv('WEATHERSTACK_API_KEY')}&query={city}"
    
    response = requests.get(url)

    print(response.json())    

    return response.json()



# Pull a prompt from the LangSmith hub. This prompt will be used to guide the agent's behavior and responses. The prompt is pulled from the "hwchase17/react" repository on the hub, and it is marked as safe to pull publicly.
# Public prompts are blocked by default for safety; set this to True only when you explicitly trust the prompt source and contents.
prompt = Client().pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True,
)

print("Prompt from hub:", prompt)

# Crate an instance of the ChatOpenAI model. This will be used as the language model for our agent to process the input and generate responses.
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create an agent using the language model and the search tool.
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_current_weather],
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=[search_tool, get_current_weather], verbose=True)

# Run the agent with a query. The agent will process the query, use the search tool to perform a web search, and return the search results.
query = "Give me the capital of Karnataka and then its current weather"

response = agent_executor.invoke({"input": query})
print(response)
print(response["output"])
