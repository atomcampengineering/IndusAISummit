import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
# commenting this out because agent should rely entirely on their Internal Knowledge
#from crewai_tools import SerperDevTool # Or use DuckDuckGo Search

# 1. Setup API Keys
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2. Define the Brain (LLM)
llm = ChatOpenAI(model="gpt-4o")

# 3. Define the Agents (The Team) [cite: 86, 87]
researcher = Agent(
    role='Deep Tech Researcher',
    goal='Identify key breakthroughs in AI chips and hardware',
    backstory='You are a curious tech journalist with a sharp eye for detail and technical specs.',
    tools=[], # Add search tools here
    verbose=True,
    llm=llm
)

analyst = Agent(
    role='Market Impact Analyst',
    goal='Analyze how new hardware impacts local tech markets',
    backstory='You are a financial expert who translates technical jargon into business impact.',
    verbose=True,
    llm=llm
)

# 4. Define the Tasks (The Jobs) [cite: 89, 90]
research_task = Task(
    description='Analyze the latest NVIDIA earnings and hardware releases.',
    expected_output='A summary of 3 major hardware updates.',
    agent=researcher
)

analysis_task = Task(
    description='Evaluate the price and availability impact of these updates in the local market.',
    expected_output='A 2-paragraph business impact report.',
    agent=analyst
)

# 5. Assemble the Crew & Kickoff [cite: 92, 93]
industry_crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=Process.hierarchical, # Process.sequential, Process.hierarchical
    manager_llm=llm
)

print("### Starting the Mission ###")
result = industry_crew.kickoff()

print("\n\n########################")
print("## FINAL REPORT OUTPUT ##")
print("########################\n")
print(result)