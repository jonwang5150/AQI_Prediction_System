"""Offline checks for agent wiring, tool failures, and actual PNG rendering."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from ollama import ResponseError

from agent.matplotlib_tool import MatplotlibTool, extract_code
from agent.sql_tool import SQLTool
from agent.weather_tool import WeatherForecastTool


class AgentToolTests(unittest.TestCase):
    def test_agent_initializes_without_sending_a_question(self):
        from agent.agent import agent, tools, agent_system_prompt, AGENT_SYSTEM_PROMPT
        self.assertIsNotNone(agent)
        self.assertEqual(len(tools), 4)
        for tool in tools:
            self.assertIn(tool.name, AGENT_SYSTEM_PROMPT)
        request = MagicMock()
        handler = MagicMock()
        agent_system_prompt.wrap_model_call(request, handler)
        content = request.override.call_args.kwargs['system_message'].content
        self.assertIn('Asia/Taipei', content)
        self.assertIn('truncated=true', content)

    def test_sql_async_and_service_failure(self):
        def unavailable(_):
            raise ResponseError('secret-token', status_code=401)
        tool = SQLTool(runnable=RunnableLambda(unavailable), engine=MagicMock())
        result = json.loads(asyncio.run(tool.ainvoke({'query': 'test'})))
        self.assertIn('error', result)
        self.assertNotIn('secret-token', str(result))
        tool.engine.connect.assert_not_called()

    def test_weather_rejects_empty_success_payload(self):
        session = MagicMock()
        session.get.return_value.json.return_value = {}
        result = json.loads(WeatherForecastTool(session=session).invoke({}))
        self.assertIn('error', result)

    def test_plot_code_extraction(self):
        self.assertEqual(extract_code('Text\n```python\nprint(1)\n```'), 'print(1)')
        self.assertEqual(extract_code('```py\nprint(2)\n```'), 'print(2)')

    def test_plot_pipeline_creates_png(self):
        code = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.plot([1, 2], [3, 4])
plt.savefig(OUTPUT_PATH)
plt.close()
'''
        model = RunnableLambda(lambda _: AIMessage(content=f'```python\n{code}```'))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            tool = MatplotlibTool.create(model, output_dir=folder)
            result = tool.invoke({'query': 'Plot the supplied two points.'})
            images = list(Path(folder).glob('*.png'))
            self.assertEqual(len(images), 1, result)
            self.assertTrue(images[0].read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))

    def test_plot_service_failure_does_not_leak_exception(self):
        def unavailable(_):
            raise ResponseError('secret-token', status_code=401)
        result = MatplotlibTool(runnable=RunnableLambda(unavailable)).invoke({'query': 'Plot'})
        self.assertIn('失敗', result)
        self.assertNotIn('secret-token', result)


if __name__ == '__main__':
    unittest.main()
