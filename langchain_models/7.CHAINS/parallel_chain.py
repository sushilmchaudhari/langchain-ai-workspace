from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short and simple frequently asked questions from the following text \n {topic}.",
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="Merge notes and faqs in a single document, notes -> {notes} and faqs -> {faqs}.",
    input_variables=["notes", "faqs"]
)

parallel_chain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "faqs": prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

final_chain = parallel_chain | merge_chain

time_management = """Time management is the process of planning and controlling time spent on specific tasks to increase efficiency, productivity, and balance. It involves setting goals, prioritizing tasks, and reducing procrastination to reduce stress and improve performance. Effective techniques include using to-do lists, scheduling, and prioritizing tasks.Key Time Management Strategies and TechniquesPrioritize Tasks: Use the "5 P's" (Prioritize, Plan, Prepare, Pace, and Persist) or evaluate tasks based on urgency and impact.Set Goals: Establish clear, realistic goals to stay focused.Use To-Do Lists/Calendars: Organize tasks by writing them down and scheduling them on a calendar.Pomodoro Technique: Work in 25-minute focused intervals followed by 5-minute breaks to maintain energy and focus.Avoid Multitasking: Focus on one task at a time to increase efficiency.Delegate/Say No: Recognize when to delegate tasks or turn down requests to avoid overloading.Schedule Breaks: Include breaks to recharge and avoid burnout.Benefits of Effective Time ManagementReduced Stress: A organized schedule lowers anxiety and feeling overwhelmed.Increased Productivity: Completing more tasks in less time.Improved Work-Life Balance: Creating time for personal life and relaxation.Better Reputation: Meeting deadlines and producing quality work boosts professional reputation.Common Time Management MistakesProcrastination: Putting off tasks, which leads to last-minute rushes.Perfectionism: Focusing too much on minor details, which slows down completion.Poor Planning: Failing to evaluate priorities or failing to plan for unexpected interruptions.Tools for Time ManagementDigital tools like Notion or Evernote.Physical planners and notebooks.Voice-to-text for quick notes.By implementing these strategies, individuals can manage their time more effectively and achieve a better balance in their personal and professional lives.
"""
response = final_chain.invoke({"topic": time_management})


print(response)
final_chain.get_graph().print_ascii()