from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Load OpenAI API key from .env file
load_dotenv()   

# This is using a default model, which is gpt-3.5-turbo with default temperature of 0.7. 
llm = ChatOpenAI() 

chat_history = [
    SystemMessage(content="You are a helpful assistant. Your name is Jarvis."),
]

# We will have a conversation with the chatbot. The user will ask a question and the chatbot will respond. We will keep track of the chat history to maintain the context of the conversation.

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Chatbot:", response.content)

print("Chat history: \n", chat_history)