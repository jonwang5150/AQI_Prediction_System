from langchain.agents import create_agent
from sql_tool import SQLTool
from search_tool import SearchTool

AGENT_SYSTEM_PROMPT = dedent("""\

""")




tools = [SQLTool.create(),SearchTool()]




agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=AGENT_SYSTEM_PROMPT,
    name = ""
)

content = dedent("""\

""")

for update in agent.stream(
    {"messages": {"role": "user", "content": content}},
    stream_mode="updates",
):
    print(update)