# Document Similarity and Semantic Search using OpenAI Embeddings and Langchain
# There are 5 different documents.
# User will ask a question and we will find the most similar document to the question using cosine similarity.
# We will use OpenAI's text-embedding-3-large model to generate embeddings for the documents and the user query.
# We will then calculate the cosine similarity between the user query embedding and each document embedding to find the most similar document.


# Import Embeddings library from langchain

from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load OpenAI API key from .env file
load_dotenv()

# Generate dynamic documents for testing.

documents = ["Cristiano Ronaldo is a Portuguese professional footballer who plays as a forward for Saudi Pro League club Al Nassr and captains the Portugal national team. He is widely regarded as one of the greatest footballers of all time. Ronaldo has won five Ballon d'Or awards, four European Golden Shoes, and has been named to the UEFA Team of the Year 14 times. He has scored over 800 senior career goals for club and country.",
"Lionel Messi is an Argentine professional footballer who plays as a forward for Ligue 1 club Paris Saint-Germain and captains the Argentina national team. He is widely regarded as one of the greatest footballers of all time. Messi has won seven Ballon d'Or awards, six European Golden Shoes, and has been named to the UEFA Team of the Year 15 times. He has scored over 750 senior career goals for club and country.",
"Sachin Tendulkar is an Indian former international cricketer and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. Tendulkar has scored 100 international centuries, the most by any player, and has a total of 34,357 runs in international cricket. He has received numerous awards, including the Bharat Ratna, India's highest civilian award.",
"Virat Kohli is an Indian international cricketer and the current captain of the Indian national team. He is widely regarded as one of the best batsmen in the world. Kohli has scored over 70 international centuries and has a total of 24,000 runs in international cricket. He has received several awards, including the Sir Garfield Sobers Trophy for ICC Cricketer of the Year in 2017 and 2018.",
"Michael Jordan is an American former professional basketball player and businessman. He is widely regarded as one of the greatest basketball players of all time. Jordan won six NBA championships with the Chicago Bulls and was named the NBA Finals MVP six times. He was a 14-time NBA All-Star and a five-time NBA Most Valuable Player. Jordan is also known for his successful business ventures, including the Air Jordan brand of basketball shoes"
]   


user_query = "Who is the best cricketer of all time?"

# Initialize the OpenAI Embeddings model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

# Generate embeddings for the documents
document_embeddings = embedding_model.embed_documents(documents)

# Generate embedding for the user query
query_embedding = embedding_model.embed_query(user_query)   

# Calculate cosine similarity between the user query embedding and each document embedding
# The output is in 2D list format. 
# We will take the first element of the list to get the similarity scores for each document.


# This returns a NumPy array, so argmax() works
similarities = cosine_similarity([query_embedding], document_embeddings)[0]

# Find the index of the most similar document
# The argmax function is Numpy function that returns the index of the maximum value in a NumPy array.
most_similar_doc_index = similarities.argmax()

# Print the most similar document and its similarity score
print("User Query:\n", user_query)
print("Most similar document:\n", documents[most_similar_doc_index])
print("Similarity score:", similarities[most_similar_doc_index])


# # There is another way to find out the index of the most similar document using enumerate function.
# # We will use the enumerate function to iterate through the similarities list and find the index of the most similar document.
# index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]

# # index = max(enumerate(lst), key=lambda x: x[1])[0]

# print("Similarity score:", score)
# print("Similarity Index", index)
# print("Most similar document:", documents[index])




