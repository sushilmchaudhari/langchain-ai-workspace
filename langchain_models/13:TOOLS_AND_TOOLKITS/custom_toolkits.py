# Toolkits are a collection of tools that can be used together to perform complex tasks. 
# They provide a way to organize and manage multiple tools, making it easier to use them in a cohesive manner. Toolkits can be used in various applications, such as natural language processing, data analysis, and machine learning. 
# They allow developers to create a set of tools that can be easily accessed and utilized by language models or other components in a larger system, enhancing the capabilities and functionality of the overall application.

from langchain_core.tools import Tool, tool

@tool()
def add_numbers(num1: float, num2: float) -> float:
    """A simple function that takes two numbers as input and returns their sum."""
    return num1 + num2

@tool()
def multiply_numbers(num1: float, num2: float) -> float:
    """A simple function that takes two numbers as input and returns their product."""
    return num1 * num2  

class MathToolkit:
    """A toolkit that contains multiple math-related tools."""
    
    def __init__(self):
        self.add_tool = add_numbers
        self.multiply_tool = multiply_numbers
    
    def get_tools(self):
        """Return a list of tools contained in the toolkit."""
        return [self.add_tool, self.multiply_tool]

    def get_tools(self):
        """Return a list of tools contained in the toolkit."""
        return [add_numbers, multiply_numbers]

# Example usage of the MathToolkit. We create an instance of the MathToolkit and then call the add_numbers and multiply_numbers tools to perform addition and multiplication operations, respectively.

toolkit = MathToolkit()

result_add = toolkit.add_tool.invoke({"num1": 3, "num2": 10})  # This is an alternative way to call the add_numbers tool without using invoke method.
print("The result of adding 3 and 10 is:\t", result_add)


for tool in toolkit.get_tools():
    if tool.name == "add_numbers":
        result = tool.invoke({"num1": 3, "num2": 7})
        print("\n Result of add_numbers tool:\t", result)
    elif tool.name == "multiply_numbers":
        result = tool.invoke({"num1": 4, "num2": 6})
        print("\n Result of multiply_numbers tool:\t", result)
