from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}.",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"topic": "space exploration"})

print(response)

chain.get_graph().print_ascii()

