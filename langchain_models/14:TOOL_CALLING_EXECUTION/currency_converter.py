# This example is a simple currency converter tool that can be used in a LangChain agent. It uses a free API to get the latest exchange rates and convert between different currencies.

import os
from typing import Annotated

from langchain_openai import ChatOpenAI, data
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@tool("get_currency_rate")
def get_currency_rate(base_currency: str, target_currency: str) -> float:
    """
    Get the exchange rate from base_currency to target_currency.
    """
    api_key = os.getenv("EXCHANGE_RATE_API_KEY")
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    
    
    return response.json()
    

    # if "conversion_rate" in data:
    #     return data["conversion_rate"]
    # else:
    #     raise ValueError(f"Could not get exchange rate for {base_currency} to {target_currency}")


# print(get_currency_rate.invoke({"base_currency": "USD", "target_currency": "INR"}))

@tool("convert_currency")
def convert_currency(amount: float, exchange_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Convert the amount from base_currency to target_currency using the exchange_rate.
    """
    return amount * exchange_rate