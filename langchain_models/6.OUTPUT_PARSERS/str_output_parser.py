from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
from regex import template

load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     task="text-generation",
 )

model = ChatHuggingFace(llm=llm)

# Define a prompt template for the model to generate a detailed customer support response

template1 = PromptTemplate(
    template="You are a helpful Customer Support assistant. Your name is Jarvis. Answer the following customer query in detail: {customer_query}",
    input_variables=["customer_query"]
)

# Define a prompt for the model to generate a 5 lines summary of the detailed report on the customer query

template2 = PromptTemplate(
    template="Summarize the following customer support response in 5 lines: {detailed_response}",
    input_variables=["detailed_response"]
)

str_parser = StrOutputParser()

reponse_chain = template1 | model | str_parser | template2 | model | str_parser

result = reponse_chain.invoke({"customer_query": "What is the status of my refund?"})

print(result)