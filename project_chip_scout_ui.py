import os
import gradio as gr
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# 1. Setup
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o")

def run_silicon_squad(topic):
    # 2. Define specialized Agents
    researcher = Agent(
        role='Deep Tech Researcher',
        goal=f'Identify key breakthroughs in {topic}',
        backstory='You are a technical expert specializing in AI hardware and semiconductor trends.',
        verbose=True,
        llm=llm
    )

    analyst = Agent(
        role='Market Impact Analyst',
        goal=f'Analyze the economic and availability impact of {topic}',
        backstory='You translate technical specs into business intelligence for investors.',
        verbose=True,
        llm=llm
    )

    # 3. Define Tasks
    research_task = Task(
        description=f'Research the latest updates regarding {topic}. Focus on technical specs.',
        expected_output='A bulleted list of the top 3 hardware advancements.',
        agent=researcher
    )

    analysis_task = Task(
        description=f'Based on the research, evaluate how these {topic} updates impact the global market.',
        expected_output='A 2-paragraph business impact summary.',
        agent=analyst
    )

    # 4. Assemble the Hierarchical Crew
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.hierarchical,
        manager_llm=llm
    )

    # 5. Kickoff and Return Result
    result = crew.kickoff()
    return str(result)

# 6. Gradio Interface (The Web UI)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 The Silicon Squad: AI Hardware Explorer")
    gr.Markdown("Enter a topic (e.g., 'NVIDIA Blackwell Chips' or 'TPU v5p') to deploy your autonomous research team.")
    
    with gr.Row():
        input_text = gr.Textbox(label="Research Topic", placeholder="NVIDIA H100...")
        run_button = gr.Button("Deploy Squad", variant="primary")
    
    output_text = gr.Textbox(label="Final Intelligence Report", lines=15)

    run_button.click(fn=run_silicon_squad, inputs=input_text, outputs=output_text)

# Launch the app
if __name__ == "__main__":
    demo.launch()