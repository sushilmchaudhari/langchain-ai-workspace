from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

result = embedding_model.embed_query("The caipital of India is New Delhi.")


print("Single query embeddings:", result)

print("Multiple query embeddings:", embedding_model.embed_documents(["The caipital of India is New Delhi.", "The capital of USA is Washington DC."]))


