from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers.structured import StucturedOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# schema = [
#     ResponseSchema(name="name", description="Name of the character"),
#     ResponseSchema(name="age", description="Age of the character"),
#     ResponseSchema(name="house", description="House name of the character")
# ]

schema = [
    ResponseSchema(name="fact-1", description="fact 1 about the topic"),
    ResponseSchema(name="fact-2", description="fact 2 about the topic"),
    ResponseSchema(name="fact-3", description="fact 3 about the topic"),
]


structured_parser = StucturedOutputParser.from_response_schemas(schema)

# template = PromptTemplate(
#     template="Give me the top 5 characters in Games of Thrones with their name, age and house name with the following format instructions\n {format_instructions}",
#     partial_variables={"format_instructions": structured_parser.get_format_instructions()}
# )

template = PromptTemplate(
    template="Give me 3 interesting facts about {topic} with the following format instructions\n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": structured_parser.get_format_instructions()}
)

chain = template | model | structured_parser

parsed_result = chain.invoke({"topic": "Black holes"})

print(parsed_result)