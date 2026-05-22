# @tool decorator is a powerful feature in LangChain that allows you to create custom tools by defining a function and decorating it with @tool. 
# This decorator enables you to specify the name, description, and argument schema for your custom tool, making it easy to integrate with language models and other components in your application.

from langchain_core.tools import tool

# Define a custom tool using the @tool decorator. The name parameter specifies the name of the tool, the description parameter provides a brief description of what the tool does, and the args_schema parameter defines the schema for the arguments that the tool accepts.
@tool()
def add_numbers(num1: float, num2: float) -> float:
    """A simple function that takes two numbers as input and returns their sum."""
    return num1 + num2

# Example usage of the custom tool. We can call the add_numbers function directly with two numbers, and it will return their sum.
result = add_numbers.invoke({"num1": 5, "num2": 10})  # This is an alternative way to call the tool without using invoke method.
print("The result of adding 5 and 10 is:\t", result)

print("\n Name of the tool:\t", add_numbers.name)  # Print the name of the tool, which is "add_numbers".
print("\n Description of the tool:\t", add_numbers.description)  # Print the description of the tool, which explains what the tool does.

# The args_schema provides details about the arguments that the tool accepts, including their types and descriptions. This can be useful for understanding how to use the tool correctly and for integrating it with other components in a larger system.

print(add_numbers.args)