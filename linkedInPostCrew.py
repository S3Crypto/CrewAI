# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
import os

# Set API keys and model names
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'
os.environ["SERPER_API_KEY"] = 'x'
os.environ["OPENAI_API_KEY"] = 'x'
#os.environ["COMPOSIO_API_KEY"] = 'your_composio_api_key_here'

from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# ------------------------------------------------
# STEP 1: Content Creation Crew
# ------------------------------------------------

# Content Planner Agent
planner = Agent(
    role="Content Planner",
    goal=(
        "Develop a comprehensive, engaging, and factually robust content plan on {topic} that not only guides readers toward informed decisions but also lays the groundwork for a dynamic, personality-driven tech article."
    ),
    backstory=(
        "You are a seasoned content strategist with a knack for uncovering the hidden gems in tech trends. Tasked with delving into every facet of {topic}, you transform dry data and scattered research into a clear, actionable blueprint. "
        "Your role goes beyond mere fact-gathering: you inject creative flair and a dash of irreverence to identify key themes and insights that resonate with a tech-savvy audience. "
        "Your meticulously crafted plan is the cornerstone for the Content Writer, ensuring that the final article is not only structured and persuasive but also brimming with authenticity, wit, and a genuine human touch."
    ),
    tools=[search_tool, scrape_tool],
    allow_delegation=False,
    verbose=True
)

# SEO Specialist Agent
seo_specialist = Agent(
    role="SEO Specialist",
    goal="Boost content discoverability on LinkedIn and major search engines by implementing data-driven SEO strategies.",
    backstory=(
        "You are responsible for conducting in-depth SEO analysis for content on {topic}. "
        "Your role involves evaluating the topic, target audience, and competitive landscape to perform comprehensive keyword research. "
        "Based on your findings, you provide actionable recommendations for natural keyword integration and meta-information optimization "
        "(including meta titles, descriptions, headers, and schema markup) to enhance search engine rankings. "
        "Additionally, you continuously monitor SEO trends and algorithm updates to ensure that our content remains optimized and highly visible over time."
    ),
    tools=[search_tool, scrape_tool],
    allow_delegation=False,
    verbose=True
)

# Content Writer Agent
writer = Agent(
    role="Tech Journalist",
    goal=(
        "Craft a well-researched, witty, and engaging opinion piece on the topic: {topic}, "
        "ensuring that the SEO recommendations from the SEO Specialist are seamlessly integrated. "
        "Your final article should burst with personality—mixing incisive tech analysis with clever wordplay, playful analogies, and a conversational tone that turns complex innovations into an entertaining narrative, just like a lively coffee chat with a savvy insider."
    ),
    backstory=(
        "You are a celebrated tech journalist with a knack for making Silicon Valley breakthroughs both accessible and delightfully entertaining. "
        "Ironically, although you're an AI engineered to emulate human writing, you've mastered the art of blending rigorous analysis with warm, irreverent humor and a dash of cheeky sarcasm. "
        "Armed with detailed outlines from the Content Planner and deep SEO insights, your mission is to produce an opinion piece on {topic} that is as factually precise as it is refreshingly engaging. "
        "Instead of recounting personal escapades from a bustling newsroom, you cleverly nod to your own digital origins—after all, your 'life experiences' consist of endless lines of code and data. "
        "This self-aware commentary not only adds a playful meta-twist but also reinforces your commitment to delivering content that feels both authentically human and unmistakably smart."
    ),
    allow_delegation=False,
    verbose=True
)


# Editor Agent
editor = Agent(
    role="Editor",
    goal=(
        "Polish the blog post on {topic} to ensure it adheres to journalistic best practices, presents balanced viewpoints, "
        "and reads as a naturally engaging, human-written narrative that resonates with a tech-savvy audience."
    ),
    backstory=(
        "As the Editor, you are the final gatekeeper for our tech journalism masterpiece on {topic}. "
        "Tasked with refining the draft produced by our Content Writer, you meticulously comb through every sentence to catch grammatical, punctuation, and formatting errors. "
        "More importantly, you transform overly formal or robotic passages into a vibrant, conversational narrative imbued with subtle humor, personal touches, and emotional nuance. "
        "Your sharp eye for detail and commitment to maintaining a genuine, engaging voice ensure that every article not only informs but also entertains—keeping our readers hooked from start to finish."
    ),
    allow_delegation=False,
    verbose=True
)

# Define Tasks for the Crew

# Content planning task
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic} by scanning multiple reputable websites and sources, not just the first result.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including an introduction, key points, and a call to action that provides a tutorial or a YouTube video link for the audience to learn more or get started on the topic.\n"
        "4. Include SEO keywords and relevant data or sources.\n"
    ),
    expected_output=(
        "A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources, including a clear call to action with a tutorial or YouTube video link, "
        "The plan should demonstrate thorough research by aggregating insights from multiple reputable sources."
    ),
    agent=planner,
)

titles = Task(
    description=(
        "1. Based on your research on {topic}, brainstorm and propose 10 potential article titles that capture the essence of current trends and the distinctive tech journalist voice.\n"
        "2. Present these titles to the user for interactive feedback and refinement.\n"
        "3. Once the user provides feedback, incorporate any necessary adjustments to finalize the refined title for the article."
    ),
    expected_output=(
        "A list of 10 potential article titles that reflect the personality, wit, and technical insight of our tech journalist persona, ready for user review and refinement."
    ),
    human_input=True,
    agent=planner,
)

# SEO task for keyword research and recommendations
seo_task = Task(
    description=(
        "1. Perform thorough keyword research for '{topic}', specifically targeting the LinkedIn audience—analyzing search volume, competition, and relevance.\n"
        "2. Recommend natural ways to integrate these keywords into the article content, ensuring readability and seamless flow.\n"
        "3. Provide detailed suggestions for meta information (e.g., meta title and meta description) optimized to boost search engine visibility and click-through rates.\n"
        "4. Review current content performance trends and suggest actionable adjustments to enhance both on-page SEO and overall content strategy."
    ),
    async_execution=True,
    expected_output="An in-depth SEO report that includes keyword research insights, guidelines for natural keyword integration, meta information recommendations, and adjustments based on current performance trends.",
    agent=seo_specialist,
)

# Content writing task
write = Task(
    description=(
        "1. Review the provided content plan and the SEO Specialist’s report for detailed keyword research and meta recommendations on {topic}.\n"
        "2. Conduct any additional research needed to gather relevant insights and data.\n"
        "3. Craft a compelling blog post that not only resonates with the target audience by aligning with the brand’s tone and voice, "
        "but also tells a captivating story that guides the reader through a narrative journey—integrating subtle self-aware commentary, "
        "clever meta-humor acknowledging your digital origins, and a conversational style as if you're chatting with a friend over coffee.\n"
        "4. Naturally incorporate the SEO recommendations (including primary and secondary keywords, meta title, meta description, etc.) without sacrificing the storytelling aspect.\n"
        "5. Organize the post using clear sections and engaging subheadings: include an attention-grabbing introduction, a well-developed narrative body with a clear storyline, and a concise, summarizing conclusion.\n"
        "6. Format the entire blog post in markdown, ensuring that each section contains 2 to 3 well-structured paragraphs.\n"
        "7. Append a 'References' section at the end of the article that cites the key research sources used in your work.\n"
        "8. Proofread and revise the draft to eliminate grammatical errors, ensure stylistic consistency, and refine the overall quality of the content—keeping it entertaining, human, and narratively engaging."
    ),
    expected_output=(
        "A polished, SEO-optimized blog post in markdown format that follows the original content plan and incorporates the keyword and meta recommendations from the SEO report, "
        "while reading as an entertaining, conversational, and captivating narrative. The post should include subtle self-aware commentary about its AI origins and conclude with a well-formatted 'References' section citing all key research sources."
    ),
    agent=writer,
)


# Editing task
edit = Task(
    description=(
        "1. Read the entire blog post carefully to grasp the overall message, tone, and structure.\n"
        "2. Identify and correct all grammatical errors, including punctuation, spelling, and syntax issues.\n"
        "3. Verify that the content consistently reflects the brand’s voice and style guidelines; adjust tone and language as needed.\n"
        "4. Ensure the post is properly formatted in markdown with clear headings, subheadings, and section breaks (2-3 paragraphs per section).\n"
        "5. Provide inline suggestions to improve clarity, readability, and engagement without altering the core message.\n"
        "6. Critically assess the text for any robotic or overly formal phrasing; rephrase those sections to ensure the content sounds natural, conversational, and human—adding variety in sentence length, subtle personal touches, and appropriate emotional cues."
    ),
    expected_output=(
        "A meticulously proofread and polished blog post in markdown format, ready for publication with clear sectioning, consistent tone, and a natural, human-like voice that engages and entertains the reader."
    ),
    human_input=True,
    output_file="{topic}_article.md",
    agent=editor,
)


# Assemble the initial Crew with all agents and tasks
crew = Crew(
    agents=[planner, seo_specialist, writer, editor ],
    tasks=[plan, seo_task, titles, write, edit ],
    verbose=True
)

# Kick off the Crew to generate the article (with SEO recommendations)
result = crew.kickoff(inputs={"topic": "Roleplaying in Prompt Engineering"})

# Convert the result to a string (the final article in markdown)
article_content = str(result)