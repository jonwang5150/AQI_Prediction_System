from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from textwrap import dedent
from sql_tool import SQLTool
from matplotlib_tool import MatplotlibTool
from rag_tool import KnowledgeBaseTool
from weather_tool import WeatherForecastTool


AGENT_SYSTEM_PROMPT = dedent("""\

""")

sql_model = ChatOllama(#model='deepseek-v4-pro:cloud',
                            model='gpt-oss:120b-cloud',
                            base_url='https://ollama.com',
                            name='sql model', temperature=0)

rag_model = ChatOllama(#model='deepseek-v4-pro:cloud',
                            model='gpt-oss:120b-cloud',
                            base_url='https://ollama.com',
                            name='rag model', temperature=0)

matplotlib_model = ChatOllama(#model='deepseek-v4-pro:cloud',
                            model='gpt-oss:120b-cloud',
                            base_url='https://ollama.com',
                            name='matplotlib model', temperature=0)



tools = [
    WeatherForecastTool(),
    KnowledgeBaseTool.create(rag_model),
    SQLTool.create(sql_model),
    MatplotlibTool.create(matplotlib_model),
]


agent = create_agent(
    model="gpt-oss:120b-cloud",
    tools=tools,
    system_prompt=AGENT_SYSTEM_PROMPT,
    name = "aqi_agent"
)

content = dedent("""\

""")

for update in agent.stream(
    {"messages": {"role": "user", "content": content}},
    stream_mode="updates",
):
    print(update)
