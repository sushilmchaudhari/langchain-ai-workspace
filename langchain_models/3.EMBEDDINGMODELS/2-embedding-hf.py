from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv() 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 

result = embeddings.embed_query("The caipital of India is New Delhi.") 

print("Single query embeddings:", result)