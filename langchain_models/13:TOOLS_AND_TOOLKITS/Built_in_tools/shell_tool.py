# Shell Tool is a built-in tool in LangChain that allows you to execute shell commands from within your code.
# It provides a simple interface to run shell commands and capture their output, which can be useful for various applications such as automating tasks, retrieving system information, or integrating with command-line tools.

from langchain_community.tools import ShellTool

# Create an instance of the ShellTool. This tool will allow us to execute shell commands from within our code.
shell_tool = ShellTool()

# Use the shell tool to execute a shell command. The invoke method takes a command string and returns the output of the command.
# print("Executing shell command: 'ls -l'\n")
print(shell_tool.invoke("ls -l"))

# print("Executing shell command: 'whoami'\n")
print(shell_tool.invoke("whoami"))

# Print the name of the tool used for executing shell commands. The name attribute of the shell_tool instance provides the name of the tool, which in this case is "ShellTool". This can be useful for logging or debugging purposes to identify which tool was used for executing the command.
print("\n Name of the tool:\t", shell_tool.name)
print("\n Description of the tool:\t", shell_tool.description)
print("\n Arguments of the tool:\t", shell_tool.args)
print("\n Schema details received by LLM:\n", shell_tool.args_schema.model_json_schema())