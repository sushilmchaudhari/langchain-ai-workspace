from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

# Define a TypedDict for the structured output - Schema for the customer support response

class CustomerReview(TypedDict):
    summary: str
    sentiment: str

# Create an instance of the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

structured_model = model.with_structured_output(CustomerReview)

# Define a prompt for the model to generate a customer support response
review = """Best foldable phone with optimum charging speed charging from 4-100% in 1hr 10min, functionality is amazing and hard to get away from phone. Design is top notch with biggest display front screen in comparison to previous folds and inner screen is just amazing to operate with almost weightless experience. Can be handled in single hand unfolded.
Camera is good but not the best in business, you won't be disappointed. I moved from Apple after 4 years and have no regrets given the overall experience with this phone for past 2 weeks. Battery lasts a day or so on moderate to heavy usage."""

# Generate a response from the model
response = structured_model.invoke(review)

# print(response)

# Accessing the structured output
print("Customer Review Summary:", response["summary"])
print("Sentiment Analysis:", response["sentiment"])
