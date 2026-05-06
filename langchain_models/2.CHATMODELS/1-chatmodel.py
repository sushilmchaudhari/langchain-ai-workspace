from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model="gpt-4")

response = chat_model.invoke("What is the capital of India?")

print(response.content)