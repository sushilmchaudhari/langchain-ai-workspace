from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# create a chat prompt template with a system message, a messages placeholder for the chat history, and a human message

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful Customer Support assistant. Your name is Jarvis."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_query}"),
])


# Load chat history from a database or any other source. Here we are using a text file. The file name is chat_history.txt.
chat_history = []

with open("chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())

print(chat_history)

# Create the prompt by invoking the chat template with the chat history
prompt = chat_template.invoke({"chat_history": chat_history, "user_query": "What is the status of my refund?"})

print(prompt)