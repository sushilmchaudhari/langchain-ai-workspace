# Semantic Meaning Splitter is a text splitter that splits text into chunks based on the semantic meaning of the text. 
# It uses a language model to analyze the text and determine the best way to split the text into chunks based on the meaning of the text. 
# The Semantic Meaning Splitter is useful when we want to split text into chunks based on the meaning of the text, rather than just splitting the text based on character count or natural language boundaries. 
# For example, if we have a long document that contains multiple sections, we can use the Semantic Meaning Splitter to split the document into chunks based on the sections of the document, rather than just splitting the document into chunks of a certain number of characters.

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.7,
)

sample_text = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = text_splitter.create_documents([sample_text])
print("Length of documents:", len(docs))
print("Documents:", docs)