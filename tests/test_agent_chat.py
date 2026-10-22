"""Agent chat adapter and Gradio callbacks, without external model calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent.agent import chat_with_agent


# 驗證聊天歷史、工具結果與 Gradio 對話事件的整合行為。
class AgentChatTests(unittest.TestCase):
    # 確認追問保留工具上下文，且不同對話的歷史互不影響。
    def test_follow_up_keeps_tool_context_and_separate_history(self):
        original = [HumanMessage(content='first'), AIMessage(content='answer')]
        runner = Mock()
        # 模擬 Agent 回傳含工具來源與最終回答的訊息序列。
        def invoke(state, config):
            return {'messages': [*state['messages'],
                ToolMessage(name='AQIKnowledgeBase', tool_call_id='x',
                            content=json.dumps({'references': 'source [1]'})),
                AIMessage(content='next answer')]}
        runner.invoke.side_effect = invoke
        result = chat_with_agent('follow up', original, runner=runner)
        self.assertEqual(len(original), 2)
        self.assertEqual(result['answer'], 'next answer')
        self.assertEqual(result['references'], 'source [1]')
        self.assertEqual(runner.invoke.call_args.args[0]['messages'][:2], original)
        chat_with_agent('new session', runner=runner)
        self.assertEqual(len(runner.invoke.call_args.args[0]['messages']), 1)

    # 確認只公開圖表目錄內、由成功工具結果回傳的圖檔。
    def test_only_successful_tool_chart_under_chart_directory_is_exposed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / 'chart.png'
            good.write_bytes(b'image')
            runner = Mock()
            runner.invoke.side_effect = lambda state, config: {'messages': [*state['messages'],
                ToolMessage(name='MatplotlibTool', tool_call_id='1', content=f'圖表已產生，檔案路徑：{good}\n\ncode'),
                ToolMessage(name='MatplotlibTool', tool_call_id='2', content=f'圖表已產生，檔案路徑：{Path(__file__).resolve()}'),
                AIMessage(content='done')]}
            with patch('agent.agent.CHART_DIRECTORY', root):
                self.assertEqual(chat_with_agent('plot', runner=runner)['charts'], [str(good.resolve())])

    # 確認呼叫失敗時保留原歷史，且不洩漏內部例外內容。
    def test_failure_preserves_history_without_exposing_exception(self):
        runner = Mock()
        runner.invoke.side_effect = RuntimeError('secret database URL')
        history = [HumanMessage(content='old')]
        result = chat_with_agent('new', history, runner=runner)
        self.assertFalse(result['ok'])
        self.assertEqual(result['history'], history)
        self.assertNotIn('secret', result['answer'])

    # 確認空白問題不會觸發 Agent 呼叫。
    def test_empty_input_does_not_call_agent(self):
        runner = Mock()
        self.assertFalse(chat_with_agent(' ', runner=runner)['ok'])
        runner.invoke.assert_not_called()

    # 確認面板透過聊天介面更新回覆、圖表與清除狀態。
    def test_panel_events_use_adapter_and_clear_state(self):
        import gradio as gr
        from prediction_visualization.visualizations import build_agent_panel
        with gr.Blocks(analytics_enabled=False) as app:
            build_agent_panel()
        respond = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, '__name__', '') == 'respond')
        with patch('agent.agent.chat_with_agent', return_value={
                'answer': 'answer', 'history': ['state'], 'charts': [], 'references': 'source'}) as adapter:
            updates = list(respond('question', [], []))
        self.assertEqual(updates[-1][1][-1], {'role': 'assistant', 'content': 'answer'})
        self.assertEqual(updates[-1][2], ['state'])
        self.assertTrue(updates[-1][6]['interactive'])
        self.assertEqual(updates[-1][7], {'__type__': 'update'})
        adapter.assert_called_once_with('question', [])
        clear = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, '__name__', '') == '<lambda>')
        cleared = clear()
        self.assertEqual(cleared[:5], ('', [], [], [], ''))
        self.assertFalse(cleared[5]['visible'])
        with patch('agent.agent.chat_with_agent', return_value={
                'answer': 'chart', 'history': [], 'charts': ['chart.png'], 'references': ''}):
            plotted = list(respond('plot', [], []))[-1]
        self.assertTrue(plotted[7]['visible'])
        self.assertEqual(plotted[8]['elem_classes'], ['aqi-agent-wide'])
        self.assertTrue(plotted[9]['visible'])
        response_event = next(fn for fn in app.fns.values() if fn.fn is respond)
        self.assertEqual(len(plotted), len(response_event.outputs))
        app.close()


if __name__ == '__main__':
    unittest.main()
