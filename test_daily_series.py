import json
import unittest
from unittest.mock import patch
from langchain_core.runnables import RunnableLambda
from agent.rag_tool import (
    _daily_series_period, _format_daily_series, KnowledgeBaseTool, SUMMARY_SOURCES,
)


class DailySeriesTests(unittest.TestCase):
    def document(self, day, value='55.00', station='木柵站'):
        return {'document': f'平均實際 AQI：{value}', 'cmetadata': {
            'station': station, 'source': SUMMARY_SOURCES[0], 'date': day, 'sample_count': 24}}

    def test_complete_month_sorted_without_top_k(self):
        documents = [self.document(f'2026-06-{day:02d}') for day in range(30, 0, -1)]
        result = _format_daily_series(documents, 2026, 6)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['rows']), 30)
        self.assertEqual(result['rows'][0]['date'], '2026-06-01')
        self.assertEqual(result['rows'][-1]['date'], '2026-06-30')
        self.assertEqual(result['rows'][0]['avg_actual_aqi'], 55)

    def test_missing_dates_and_values_are_not_zero(self):
        result = _format_daily_series([
            self.document('2026-06-01'), self.document('2026-06-02', '無有效預測配對，無法計算')], 2026, 6)
        self.assertFalse(result['complete'])
        self.assertEqual(result['dates_without_average'], ['2026-06-02'])
        self.assertEqual(len(result['missing_dates']), 28)
        self.assertIsNone(result['rows'][2]['avg_actual_aqi'])

    def test_wrong_scope_or_duplicates_fail(self):
        for docs in ([self.document('2026-06-01', station='其他站')],
                     [self.document('2026-06-01')] * 2):
            with self.assertRaises(ValueError):
                _format_daily_series(docs, 2026, 6)

    @patch('agent.rag_tool.query_daily_series')
    def test_tool_routes_month_series_without_llm(self, series):
        series.return_value = {'rows': [], 'complete': False}
        def no_model(_):
            raise AssertionError('Series retrieval must not call an LLM')
        tool = KnowledgeBaseTool.create(RunnableLambda(no_model))
        result = json.loads(tool.invoke({'question': '將2026年6月份每日平均aqi資料做成趨勢圖'}))
        series.assert_called_once_with(2026, 6)
        self.assertFalse(result['complete'])

    def test_other_questions_keep_normal_rag(self):
        for question in ('2026/6/9 平均AQI', '2026年6月平均AQI', '2026年6月每日MAE', '2026年6月哪天AQI最高'):
            self.assertIsNone(_daily_series_period(question))


if __name__ == '__main__':
    unittest.main()
