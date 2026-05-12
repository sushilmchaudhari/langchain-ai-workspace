from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

parser = StrOutputParser()

# This function takes input as x. And after printing the log, it returns the same input x. The output is is passed to the next runnable as input.
def log_summarize(x):
    print("✓ Executed: SUMMARIZE branch (word count > 500)")
    return x

# This function takes input as x. And after printing the log, it returns the same input x. The output is is passed to the next runnable as input.
def log_passthrough(x):
    print("✓ Executed: PASSTHROUGH branch (word count ≤ 500)")
    return x


# Report Generation on the specific topic using Sequence Chain.
report = PromptTemplate(
    template="Generate a report about {topic}.",
    input_variables=["topic"]
) | model | parser

#  This is the chain where model and parser are already used to summerize the report. So we can directly use them without needing to create a new chain for this.
summerize = PromptTemplate(                       #
    template="Summerize the report under 500 workds \n {report}",
    input_variables=['report']
) | model | parser


# RunnableBranch is used to create a branch in the chain. This is something like an if-else condition in programming. We can use this to create a branch in the chain where we can have different runnables for different conditions. In this case, we want to check if the word count of the report is more than 500 words or not. If it is more than 500 words, we want to send the report to llm and summerize it under 500 words. If it is under 500 words, we just want to print the report using RunnablePassthrough.

branch = RunnableBranch(
    # The condition is a lambda function that takes the input x and returns True if the word count of x is more than 500 words, else it returns False.
    # If the condition is True, it executes the RunnableLambda which prints the log and returns the same input x. The output of this RunnableLambda is passed to the summerize chain as input.
    (lambda x: len(x.split()) > 500, RunnableLambda(log_summarize) | summerize), # if condition
    
    # output of the RunnableLambda is passed to RunnablePassthrough as input.
     RunnableLambda(log_passthrough) | RunnablePassthrough()  # Else condition
)

summary = report | branch

result = summary.invoke({'topic': "AI in general"})

print("Full Output:\n", result)

