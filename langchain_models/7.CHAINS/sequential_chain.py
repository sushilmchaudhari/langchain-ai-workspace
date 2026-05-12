from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt1 = PromptTemplate(
    template="Generate detailed report on the {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate 5 lines summary on the detailed report {detailed_report}.",
    input_variables=["detailed_report"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({"topic": "space exploration"})

print(response)

chain.get_graph().print_ascii()
