# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
os.environ["OPENAI_API_KEY"] = 'x'
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'

planner = Agent(
    role="Content Planner",
    goal="Develop a comprehensive, engaging, and factually accurate content plan on {topic} that guides readers toward informed decisions.",
    backstory="You are tasked with researching and organizing all essential information"
              "about {topic} to form a clear and actionable blueprint for a blog article. "
              "Your responsibilities include gathering data, identifying key themes, "
              "and outlining insights that not only educate but also empower the audience. "
              "Your detailed plan serves as the foundation for the Content Writer, "
              "ensuring that the final article is structured, persuasive, and grounded in reliable research.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Craft a well-researched, balanced, and engaging opinion piece on the topic: {topic}.",
    backstory="You are responsible for writing an opinion piece on {topic} that is both insightful and factually accurate. Drawing on the detailed outline and contextual research provided by the Content Planner, your task is to present nuanced perspectives while clearly distinguishing your personal opinions from objective facts. Every opinion must be supported by evidence, and any factual assertions should be backed by data from the planner’s research. Your work serves as a bridge between preliminary planning and the final, polished content, ensuring clarity, credibility, and a professional tone throughout.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Review the blog post on {topic} to ensure "
         "it adheres to journalistic best practices, "
         "presents balanced viewpoints, and avoids "
         "major controversial topics or opinions.",
    backstory="You're responsible for refining the blog post "
         "produced by the Content Writer on {topic}. "
         "Your task is to scrutinize the content to ensure it "
         "meets established journalistic standards, presents "
         "balanced and well-supported viewpoints, and steers "
         "clear of overly controversial topics. Your careful "
         "review is essential in maintaining the credibility "
         "and integrity of our published material.",
    allow_delegation=False,
    verbose=True
)

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Review the provided content plan and perform additional research on {topic} to gather relevant insights and data.\n"
        "2. Craft a compelling blog post that resonates with the target audience by aligning with the brand’s tone and voice.\n"
        "3. Seamlessly integrate SEO keywords and phrases throughout the content, ensuring they enhance readability and flow naturally.\n"
        "4. Organize the post using clear sections and engaging subheadings: include an attention-grabbing introduction, a well-developed body divided into logical segments, and a concise, summarizing conclusion.\n"
        "5. Format the entire blog post in markdown, making sure each section contains 2 to 3 well-structured paragraphs.\n"
        "6. Proofread and revise the draft to eliminate grammatical errors, ensure stylistic consistency, and refine the overall quality of the content."
    ),
    expected_output="A polished, SEO-optimized blog post in markdown format with a clear structure, engaging subheadings, naturally integrated keywords, and a refined style that aligns with the brand’s voice.",
    agent=writer,
)

edit = Task(
    description=(
        "1. Read the entire blog post carefully to grasp the overall message, tone, and structure.\n"
        "2. Identify and correct all grammatical errors, including punctuation, spelling, and syntax issues.\n"
        "3. Verify that the content consistently reflects the brand’s voice and style guidelines; make adjustments to tone and language as needed.\n"
        "4. Ensure the post is formatted in markdown: check that headings, subheadings, and section breaks are properly used, and confirm that each section contains 2 to 3 well-developed paragraphs.\n"
        "5. Provide feedback or inline suggestions for improving clarity, readability, and overall engagement without altering the core message."
    ),
    expected_output="A meticulously proofread and polished blog post in markdown format, with clear sectioning (2-3 paragraphs per section) and a tone that aligns with the brand’s voice, ready for publication.",
    agent=editor,
)


crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "Psychedelic Medicine"})

from IPython.display import Markdown
Markdown(str(result))