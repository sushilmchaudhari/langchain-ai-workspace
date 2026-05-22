# Structured tool with Pydantic is a powerful feature in LangChain that allows you to define custom tools with structured input and output using Pydantic models.
# This approach provides a clear and structured way to define the inputs and outputs of your tools, making it easier to integrate with language models and other components in your application.

from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

class MultiplyInputs(BaseModel):
    """A Pydantic model that defines the structure of the input for the custom tool."""
    num1: float = Field(..., description="The first number to multiply.")
    num2: float = Field(..., description="The second number to multiply.")


def multiply_func(num1: float, num2: float) -> float:
    """A simple function that takes two numbers as input and returns their product."""
    return num1 * num2

multiply = StructuredTool.from_function(
    func = multiply_func,
    args_schema = MultiplyInputs,
    description = "A tool to multiply two numbers together.",
    name = "multiply_numbers"
)

# Example usage of the custom tool. We can call the multiply function directly with two numbers, and it will return their product.

result = multiply.invoke({"num1": 5, "num2": 10})  # This is an alternative way to call the tool without using invoke method.
print("The result of multiplying 5 and 10 is:\t", result)

print("\n Name of the tool:\t", multiply.name)  # Print the name of the tool, which is "multiply_numbers".
print("\n Description of the tool:\t", multiply.description)  # Print the description of the tool, which explains what the tool does.
print("\n Arguments of the tool:\t", multiply.args)  # Print the arguments of the tool, which are defined in the MultiplyInputs Pydantic model.
print("\n Schema details received by LLM:\n", multiply.args_schema.model_json_schema())  # Print the schema details of the arguments received by the language model, which provides information about the structure and types of the inputs expected by the tool. This can be useful for understanding how to use the tool correctly and for integrating it with other components in a larger system.