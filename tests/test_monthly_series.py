"""跨月平均 AQI 製圖資料的完整性與路由測試，不呼叫外部服務。"""
import json
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from agent.agent import chat_with_agent
from agent.matplotlib_tool import MatplotlibTool
from agent.rag_tool import (
    KnowledgeBaseTool, SUMMARY_SOURCES, _format_monthly_series,
    _monthly_series_period, query_monthly_series,
)


class MonthlySeriesTests(unittest.TestCase):
    def document(self, month, value='55.00', station='木柵站', year=2026):
        return {'document': f'平均實際 AQI：{value}\n配對樣本數：240', 'cmetadata': {
            'station': station, 'source': SUMMARY_SOURCES[1],
            'month': f'{year}-{month:02d}-01 00:00:00'}}

    def test_screenshot_question_and_range_variants(self):
        for question in (
            '幫我將2026/1~9月的平均aqi轉成長條圖',
            '木柵2026年1月至9月的平均 AQI',
            '2026 年 1～9 月平均 AQI',
            '2026-01 到 2026-09月平均AQI',
        ):
            with self.subTest(question=question):
                self.assertEqual(_monthly_series_period(question), (2026, 1, 9))

    def test_single_dates_and_other_metrics_keep_existing_routes(self):
        for question in ('2026/1/9平均AQI', '2026-01-09平均AQI', '2026年1月平均AQI',
                         '2026年1~9月每日平均AQI', '2026年1~9月平均預測AQI', '2026年1~9月MAE'):
            with self.subTest(question=question):
                self.assertIsNone(_monthly_series_period(question))

    def test_invalid_and_cross_year_ranges_are_rejected(self):
        for question in ('2026/9~1月平均AQI', '2026/0~9月平均AQI',
                         '2026/1~13月平均AQI', '2026/11~2027/2月平均AQI'):
            with self.subTest(question=question), self.assertRaises(ValueError):
                _monthly_series_period(question)

    def test_nine_months_are_complete_and_sorted_without_top_k(self):
        result = _format_monthly_series([self.document(month, str(month))
                                         for month in range(9, 0, -1)], 2026, 1, 9)
        self.assertTrue(result['complete'])
        self.assertEqual(result['expected_months'], 9)
        self.assertEqual(result['retrieved_months'], 9)
        self.assertEqual([row['month'] for row in result['rows']],
                         [f'2026-{month:02d}' for month in range(1, 10)])
        self.assertEqual([row['avg_actual_aqi'] for row in result['rows']], list(range(1, 10)))
        self.assertEqual(result['rows'][0]['sample_count'], 240)

    def test_missing_months_and_averages_remain_missing(self):
        result = _format_monthly_series([
            self.document(1), self.document(2, '無有效預測配對，無法計算')], 2026, 1, 3)
        self.assertFalse(result['complete'])
        self.assertEqual(result['missing_months'], ['2026-03'])
        self.assertEqual(result['months_without_average'], ['2026-02'])
        self.assertIsNone(result['rows'][1]['avg_actual_aqi'])
        self.assertIsNone(result['rows'][2]['avg_actual_aqi'])

    def test_wrong_scope_duplicate_months_and_invalid_values_are_rejected(self):
        for documents in (
            [self.document(1, station='其他站')], [self.document(1)] * 2,
            [self.document(10)], [self.document(1, 'NaN')],
        ):
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                _format_monthly_series(documents, 2026, 1, 9)

    @patch('agent.rag_tool.query_monthly_series')
    @patch('agent.rag_tool.query_knowledge_base')
    def test_monthly_range_routes_without_embeddings_or_llm(self, rag, series):
        series.return_value = _format_monthly_series([
            self.document(month) for month in range(1, 10)], 2026, 1, 9)
        def no_model(_):
            raise AssertionError('Stored monthly series must not call an LLM')
        result = json.loads(KnowledgeBaseTool.create(RunnableLambda(no_model)).invoke({
            'question': '幫我將2026/1~9月的平均aqi轉成長條圖'}))
        series.assert_called_once_with(2026, 1, 9)
        rag.assert_not_called()
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['rows']), 9)

    @patch('agent.rag_tool._summary_engine')
    def test_query_is_read_only_bounded_and_station_scoped(self, engine):
        connection = MagicMock()
        engine.return_value.connect.return_value.__enter__.return_value = connection
        for year in (2026, 2027, 2030):
            with self.subTest(year=year):
                connection.reset_mock()
                connection.execute.return_value.mappings.return_value.all.return_value = [self.document(12, year=year)]
                result = query_monthly_series(year, 12, 12)
                calls = connection.execute.call_args_list
                self.assertEqual(str(calls[0].args[0]), 'SET TRANSACTION READ ONLY')
                self.assertIn('statement_timeout', str(calls[1].args[0]))
                self.assertEqual(calls[2].args[1], {
                    'collection': 'aqi_knowledge_base', 'station': '木柵站',
                    'source': SUMMARY_SOURCES[1], 'start': f'{year}-12', 'end': f'{year + 1}-01'})
                self.assertTrue(result['complete'])

    @patch('agent.rag_tool._summary_engine')
    def test_unsupported_year_does_not_query_database(self, engine):
        result = query_monthly_series(2025, 1, 9)
        self.assertFalse(result['retryable'])
        engine.assert_not_called()

    @patch('agent.rag_tool.query_monthly_series')
    def test_nine_month_chart_reaches_gradio_when_final_model_text_is_empty(self, series):
        import gradio as gr
        from gradio.state_holder import SessionState
        from prediction_visualization.visualizations import build_agent_panel

        series.return_value = _format_monthly_series([
            self.document(month, str(month * 5)) for month in range(1, 10)], 2026, 1, 9)
        question = '幫我將2026/1~9月的平均aqi轉成長條圖'
        knowledge = KnowledgeBaseTool.create(RunnableLambda(lambda _: AIMessage(content='unused')))
        payload = knowledge.invoke({'question': question})
        rows = json.loads(payload)['rows']
        self.assertEqual(len(rows), 9)
        code = (
            'import matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\n'
            f'rows = {rows!r}\n'
            'plt.bar([row["month"] for row in rows], [row["avg_actual_aqi"] for row in rows])\n'
            'plt.savefig(OUTPUT_PATH)\nplt.close()\n'
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            plotter = MatplotlibTool.create(
                RunnableLambda(lambda _: AIMessage(content=f'```python\n{code}```')), output_dir=directory)
            chart_output = plotter.invoke({'query': payload})
            runner = Mock()
            runner.invoke.side_effect = lambda state, config: {'messages': [
                *state['messages'],
                ToolMessage(name='AQIKnowledgeBase', tool_call_id='data', content=payload),
                ToolMessage(name='MatplotlibTool', tool_call_id='chart', content=chart_output),
                AIMessage(content='')]}
            with patch('agent.agent.CHART_DIRECTORY', Path(directory)):
                result = chat_with_agent(question, runner=runner)
            self.assertTrue(result['ok'])
            self.assertEqual(len(result['charts']), 1, result['answer'])
            self.assertTrue(Path(result['charts'][0]).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))

            with gr.Blocks(analytics_enabled=False) as app:
                build_agent_panel()
            respond = next(fn for fn in app.fns.values() if fn.fn.__name__ == 'respond')

            async def send_request():
                state = SessionState(app)
                pending = await app.process_api(respond, [question, [], None], state=state)
                self.assertTrue(pending['is_generating'])
                return await app.process_api(respond, [], state=state, iterator=pending['iterator'])

            try:
                with patch('agent.agent.chat_with_agent', return_value=result):
                    response = asyncio.run(send_request())
                self.assertTrue(response['data'][1][-1]['content'])
                self.assertEqual(len(response['data'][3]), 1)
                self.assertTrue(response['data'][7]['visible'])
                self.assertTrue(response['data'][9]['visible'])
            finally:
                app.close()


if __name__ == '__main__':
    unittest.main()
