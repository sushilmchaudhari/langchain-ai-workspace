from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# review = """Best foldable phone with optimum charging speed charging from 4-100% in 1hr 10min, functionality is amazing and hard to get away from phone. Design is top notch with biggest display front screen in comparison to previous folds and inner screen is just amazing to operate with almost weightless experience. Can be handled in single hand unfolded."""

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

str_parser = StrOutputParser()

# This parser is to parse the sentiment of the review as positive or negative using Pydantic models and output parser. the output will always be 'postive' or 'negative' based on the sentiment of the review.

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")

sentiment_parser = PydanticOutputParser(pydantic_object=Sentiment)

sentiment_prompt = PromptTemplate(
    template="Analyze the sentiment of the following review and classify it as positive, negative \n Review: {review} \n format instructions: {format_instructions}",
    input_variables=["review"],
    partial_variables={'format_instructions': sentiment_parser.get_format_instructions()}
)

classification_chain = sentiment_prompt | model | sentiment_parser

# print(classification_chain.invoke({"review": "The product is amazing and exceeded my expectations!"}))
# print(classification_chain.invoke({"review": "The product is terrible and did not meet my expectations."}))
# # To get only the sentiment without the field name.
# print(classification_chain.invoke({"review": "The product is amazing and exceeded my expectations!"}).sentiment)
# print(classification_chain.invoke({"review": "The product is terrible and did not meet my expectations."}).sentiment)

# response_prompt = PromptTemplate(
#     template="Generate {sentiment} response of the product based on the following review \n Review: {review}",
#     input_variables=["review", "sentiment"]
# )


positive_response_prompt = PromptTemplate(
    template="Generate a positive response to the following review \n Review: {review}",
    input_variables=["review"]
)

negative_response_prompt = PromptTemplate(
    template="Generate a negative response to the following review \n Review: {review}",
    input_variables=["review"]
)

branch_chain = RunnableBranch(
    # (condition1, chain1)
    # (condition1, chain1)
    # .
    # .
    # (Default condition, default_chain)
    (lambda x:x.sentiment == "positive", positive_response_prompt | model | str_parser),
    (lambda x:x.sentiment == "negative", negative_response_prompt | model | str_parser),
    RunnableLambda(lambda x: "Neutral sentiment, no response generated")
)


final_chain = classification_chain | branch_chain

customer_response = final_chain.invoke({"review": "The product is amazing and exceeded my expectations!"})

print(customer_response)