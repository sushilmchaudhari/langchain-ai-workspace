# WebBase Loader loads documents from the web.
# It is a convenient way to load documents from the web. 
# We can specify the URL of the web page to load the document from. 
# The WebBaseLoader will return a list of Document objects, where each Document object contains the content of the web page and its metadata. 
# We can access the content of the web page using the page_content attribute of the Document object, and we can access the metadata using the metadata attribute of the Document object.
# This is good for static web pages like Blogs, news articles, etc. but not good for dynamic web pages like social media, etc.
# For dynamic web pages, we can use Selenium or Playwright to load the web page and then use WebBaseLoader to extract the content and metadata from the loaded web page.

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

str_parser = StrOutputParser() 

prompt = PromptTemplate(
    template="Summarize the content of the web page below: \n {web_page_content}",
    input_variables=["web_page_content"]
) | llm | str_parser 

prompt2 = PromptTemplate(
    template="Answer the following question \n {question} on the topic: \n {web_page_content}",
    input_variables=["question", "web_page_content"]
)


url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

loader = WebBaseLoader(url)
docs = loader.load()

# This is one way to use the propmt with the LLM and parser to summarize the web page content.
result = prompt.invoke({"web_page_content": docs[0].page_content})
print("Summary of the web page:\n", result)

# This is another way to use the prompt with the LLM and parser to summarize the content of the web page.
res1 = (prompt2 | llm | str_parser).invoke({"question": "What is artificial intelligence?", "web_page_content": docs[0].page_content})
print("Answer of the question on the web page using LLM:\n", res1)


# print("Length of documents:\n", len(docs))       # Number of web pages loaded (usually 1) as there is only 1 url provided

# print("Document Object:\n", docs[0])              # Document object containing the content and metadata of the web page

# print("Content of the document:\n", docs[0].page_content)  # Content of the web page

# print("Metadata of the document:\n", docs[0].metadata)  # Metadata of the web page (e.g., URL, title, etc.) 
