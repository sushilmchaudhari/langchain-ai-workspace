import random

class NakliLLM:
    def __init__(self, name="NakliChatGPT"):
        self.name = name

    def generate(self, prompt):
        responses = [
            f"{self.name} says: The answer to '{prompt}' is 42.",
            f"{self.name} says: I'm not sure about '{prompt}', but it sounds interesting!",
            f"{self.name} says: '{prompt}' is a great question! Let me think about it.",
        ]
        return random.choice(responses)

class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)


prompt = NakliPromptTemplate(
    template="What is the meaning of {input}?",
    input_variables=["input"]
)

model = NakliLLM()       # Provide the value of name variable while creating an object: Default is already set 

result = model.generate(prompt.format({"input": "life"}))
print(result)


class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.prompt = prompt
        self.llm = llm

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        
        
        return self.llm.generate(final_prompt)

template = NakliPromptTemplate(
    template='Write a {length} paragraph about {topic}',
    input_variables=['length', 'topic']
)

llm = NakliLLM(name="FakeGPT")

chain = NakliLLMChain(llm=llm, prompt=template)

chain.run({"length": "short", "topic": "What is life"})     # This will print a response from the NakliLLM based on the formatted prompt. It will use the default name.

# to print the result of the chain run, you can assign it to a variable and then print it:
result = chain.run({"length": "short", "topic": "What is life"})
print(result)

