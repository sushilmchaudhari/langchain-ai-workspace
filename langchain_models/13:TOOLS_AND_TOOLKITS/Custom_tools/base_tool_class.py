# BaseTool class is a foundational class for creating custom tools in the LangChain framework. 
# It provides a structure for defining tools that can be used by language models to perform specific tasks. 
# The BaseTool class includes methods for initializing the tool, defining its name and description, and implementing the logic for invoking the tool with specific inputs. 
# By inheriting from BaseTool, developers can create custom tools that integrate seamlessly with the LangChain ecosystem, allowing language models to utilize these tools to enhance their capabilities and perform a wider range of functions.

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInputs(BaseModel):
    """A Pydantic model that defines the structure of the input for the custom tool."""
    num1: float = Field(..., description="The first number to process.")
    num2: float = Field(..., description="The second number to process.")

class MultiplyTool(BaseTool):
    """A custom tool that multiplies two numbers together."""
    
    name: str = "multiply_tool"  # Name of the tool
    description: str = "A tool to multiply two numbers together."  # Description of the tool
    args_schema: Type[BaseModel] = MultiplyInputs  # Schema for the arguments that the tool accepts

    def _run(self, num1, num2) -> float:
        """Run the tool using validated keyword arguments from args_schema."""
        return num1 * num2

tool = MultiplyTool()

result = tool.invoke({"num1": 5, "num2": 10})  # This is an alternative way to call the tool without using invoke method.
print("The result of multiplying 5 and 10 is:\t", result)