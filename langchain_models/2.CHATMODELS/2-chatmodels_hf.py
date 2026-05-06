from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="deepseek-ai/DeepSeek-R1-0528",
     task="text-generation",
     temperature=0.9,
     max_new_tokens=512,     
)

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=1.0,
    max_new_tokens=200,
)

model = ChatHuggingFace(llm=llm) 
model1 = ChatHuggingFace(llm=llm1) 

system_msg = SystemMessage("You are a teacher.")
human_msg = HumanMessage("Teach me about the capital of India in 100 words.")

message = [system_msg, human_msg]


# result = model.invoke("What is the capital of India?")
result1 = model1.invoke(message)  

# print(result.content)
result_list = result1.content.split(" ")

print(result1.content)

print(len(result1.content.split(" ")))

