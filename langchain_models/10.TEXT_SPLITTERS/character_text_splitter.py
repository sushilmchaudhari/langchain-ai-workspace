# Character Text Splitter is a simple text splitter that splits text based on a length of characters. 
# It takes a chunk size and a chunk overlap as parameters. 
# The chunk size is the maximum number of characters in each chunk.
# The chunk overlap is the number of characters that overlap between chunks. 
# The splitter will split the text into chunks of the specified size, with the specified overlap between chunks.
# It returns a list of strings, where each string is a chunk of the original text.

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Get the directory of the current script and construct the path to report.pdf
script_dir = os.path.dirname(os.path.abspath(__file__))
report_path = os.path.join(script_dir, '..', '9.DOCUMENT_LOADERS', 'example_data', 'report.pdf')

loader = PyPDFLoader(report_path)

pdf_docs = loader.load()

text = """ Time management is the process of planning and controlling time spent on specific tasks to increase efficiency, productivity, and balance. It involves setting goals, prioritizing tasks, and reducing procrastination to reduce stress and improve performance. Effective techniques include using to-do lists, scheduling, and prioritizing tasks.Key Time Management Strategies and TechniquesPrioritize Tasks: Use the "5 P's" (Prioritize, Plan, Prepare, Pace, and Persist) or evaluate tasks based on urgency and impact.Set Goals: Establish clear, realistic goals to stay focused.Use To-Do Lists/Calendars: Organize tasks by writing them down and scheduling them on a calendar.Pomodoro Technique: Work in 25-minute focused intervals followed by 5-minute breaks to maintain energy and focus.Avoid Multitasking: Focus on one task at a time to increase efficiency.Delegate/Say No: Recognize when to delegate tasks or turn down requests to avoid overloading.Schedule Breaks: Include breaks to recharge and avoid burnout.Benefits of Effective Time ManagementReduced Stress: A organized schedule lowers anxiety and feeling overwhelmed.Increased Productivity: Completing more tasks in less time.Improved Work-Life Balance: Creating time for personal life and relaxation.Better Reputation: Meeting deadlines and producing quality work boosts professional reputation.Common Time Management MistakesProcrastination: Putting off tasks, which leads to last-minute rushes.Perfectionism: Focusing too much on minor details, which slows down completion.Poor Planning: Failing to evaluate priorities or failing to plan for unexpected interruptions.Tools for Time ManagementDigital tools like Notion or Evernote.Physical planners and notebooks.Voice-to-text for quick notes.By implementing these strategies, individuals can manage their time more effectively and achieve a better balance in their personal and professional lives.
"""

splitter = CharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20,
    separator=''
)

docs = splitter.split_text(text)
print("Split Text is: \n", docs)
print("Number of chunks:", len(docs))

# We can also split the content of a document object using the create_documents method. 
# The create_documents method takes a list of strings as input and returns a list of Document objects, where each Document object contains a chunk of the original text and its metadata. 
# We can specify the metadata for each chunk using the metadatas parameter, which takes a list of dictionaries as input. Each dictionary in the list corresponds to a chunk and contains the metadata for that chunk.

metadata = [{"source": "time_management.txt"}]
docs1 = splitter.create_documents([text])
docs2_with_metadata = splitter.create_documents([text], metadatas=metadata)
print("Split Documents is: \n", docs1)
print("Number of documents:", len(docs1))
print("Split Documents with metadata is: \n", docs2_with_metadata)
print("Number of documents with metadata:", len(docs2_with_metadata))

# We can also split the content of a pdf by using the PyPDF loader to load the pdf and then using the split_documents method of the splitter to split the content of the pdf into chunks. 
# The split_documents method takes a list of Document objects as input and returns a list of Document objects, where each Document object contains a chunk of the original text and its metadata. 
# We can specify the metadata for each chunk using the metadatas parameter, which takes a list of dictionaries as input. Each dictionary in the list corresponds to a chunk and contains the metadata for that chunk.

pdf_docs_split = splitter.split_documents(pdf_docs)
print("Split PDF Documents is: \n", pdf_docs_split)
print("Number of split PDF documents:", len(pdf_docs_split))  

