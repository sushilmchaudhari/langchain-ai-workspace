# Here we are going to create a flow where:
# 1. We will define a custom tool using the @tool decorator.
# 2. Bind the tool to a language model and demonstrate how it can be invoked in a simple example.
# 3. Call the tool directly without using the invoke method to show its flexibility in usage.
# 4. Run/Execute the tool to perform a simple operation and print the result.
# 5. Share all this information to LLM again and print the result of actual query.
# 6. We will then use this custom tool in a simple example to demonstrate how it can be invoked and how it integrates with the language model.

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage    

load_dotenv()

# Define a custom tool using the @tool decorator. The name parameter specifies the name of the tool, the description parameter provides a brief description of what the tool does, and the args_schema parameter defines the schema for the arguments that the tool accepts.
@tool()
def multiply(a : float, b : float) -> float:
    """A simple function that takes two numbers as input and returns their product."""
    return a * b

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Bind the tool to a language model. This allows the language model to recognize and utilize the tool when generating responses. The bind_tools method takes a list of tools and makes them available for the language model to use in its responses.
llm_with_tools = llm.bind_tools([multiply])

query = HumanMessage(content="Explain me some multiplcation methods. Also What if we multiply 6 and 7?")

print("\n Human Message:\t", query)

# Create a message that includes the query we want to ask the language model. This message will be passed to the language model, which will then determine if it can use the multiply tool to generate a response based on the query. The message is created as a list of HumanMessage objects, which allows for more complex interactions and conversations with the language model.
message = [query]

# Call the tool directly without using the invoke method to show its flexibility in usage. We can call the multiply function directly with two numbers, and it will return their product.

tool_call = llm_with_tools.invoke(message)  # This is an alternative way to call the tool without using invoke method.

print("\n Tool Call Result:\t", tool_call)
print("\n Tool Calls Result:\t", tool_call.tool_calls)

# # Append the tool call result to the message list, which allows us to share this information with the language model for generating a final response. The content of the tool call is set to the string representation of the tool call result, which can be used by the language model to incorporate this information into its response.
message.append(tool_call)  

# print(message)

# Tool Execution: We will then execute the tool to perform a simple operation and print the result. The tool will be executed based on the query we provided, and the result will be printed to the console. 

tool_execution = multiply.invoke(tool_call.tool_calls[0])  # This will execute the tool based on the query and return the result.

print("\n Tool Execution Result:\n", tool_execution)

# Append the tool execution result to the message list as a ToolMessage, which allows us to share this information with the language model for generating a final response. The content of the ToolMessage is set to the string representation of the tool execution result, which can be used by the language model to incorporate this information into its response.
message.append(tool_execution)

print("\n --- Final Message with Tool Execution Result: --- \n", message)

# Finally, we will share all this information to LLM again and print the result of actual query. The language model will take into account the original query, the tool call, and the tool execution result to generate a final response that incorporates all of this information.

final_response = llm_with_tools.invoke(message)

print("\n Final Response from LLM:\t", final_response)
print("\n Final Response Content:\t", final_response.content)