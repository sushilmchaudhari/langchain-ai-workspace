from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

parser = StrOutputParser()

# prompt = PromptTemplate(
#     template="Generate interesting facts about {topic}. Keep the response under 500 words",
#     input_variables=["topic"]
# )

prompt = PromptTemplate(
    template="Tell me a joke about {topic}.",
    input_variables=["topic"]
)


#  This is the chain where model and parser are already used. So we can directly use them without needing to create a new chain for this.
prompt2 = PromptTemplate(                       #
    template="Explain the joke {joke}",
    input_variables=['joke']
) | model | parser


# We can also create a sequence of runnables using RunnableSequence. This will allow us to create a sequence of runnables and use them in a single chain. We can also use the same runnables in different chains without needing to create new instances of them.
seq_chain = RunnableSequence(prompt, model, parser, prompt2)

print(seq_chain.invoke({"topic": "AI in general"}))