from langchain_core.messages import HumanMessage, AIMessage, SystemMessage 
from langchain_openai  import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

msgs = [
    SystemMessage(content="You are a helpful assistant. Your name is Jarvis."),
    HumanMessage(content="What is your name?"),
]

response = llm.invoke(msgs)

msgs.append(AIMessage(content=response.content))

print("Chat history: \n", msgs)
