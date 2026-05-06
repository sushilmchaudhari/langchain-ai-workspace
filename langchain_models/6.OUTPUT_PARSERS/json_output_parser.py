# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

json_parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the top 5 characters in Games of Thrones with their name, age and house name with the following format instructions\n {format_instructions}",
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)

# prompt = template.format()
# print(prompt)

# result = model.invoke(prompt)
# print(result)

# parsed_result = json_parser.parse(result.content)

chain = template | model | json_parser
parsed_result = chain.invoke({})


print(parsed_result)
print(type(parsed_result))
# print(parsed_result[0]['name'])
