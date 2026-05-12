from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

parser = StrOutputParser()

# prompt = PromptTemplate(
#     template="Generate interesting facts about {topic}. Keep the response under 500 words",
#     input_variables=["topic"]
# )

prompt1 = PromptTemplate(
    template="Write a tweet on {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(                       #
    template="Write a LinkedIn post on {topic}.",
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet': prompt1 | model | parser,
    'linkedin_post': prompt2 | model | parser
})

result = parallel_chain.invoke({"topic": "AI in general"})

print(result)

print("Tweet:", result['tweet'])
print("LinkedIn Post:", result['linkedin_post'])
