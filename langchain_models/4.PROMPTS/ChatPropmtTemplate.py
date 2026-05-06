from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate as CPT
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

chat_template = CPT([
    ("system", "You are a helpful assistant. Your name is Jarvis. You are {domain} expert."),
    ("human", "Tell me a joke about {topic}."),
]
)

prompt = chat_template.invoke({'domain': "technology", 'topic': "AI"})

result = model.invoke(prompt)

print(result.content)