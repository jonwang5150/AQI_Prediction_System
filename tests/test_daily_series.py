import json
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.runnables import RunnableLambda
from agent.rag_tool import (
    _daily_series_period, _format_daily_series, KnowledgeBaseTool, SUMMARY_SOURCES,
    query_daily_series,
)


# 驗證每日摘要序列的排序、完整性與工具路由。
class DailySeriesTests(unittest.TestCase):
    # 建立指定日期與測站的每日摘要測試資料。
    def document(self, day, value='55.00', station='木柵站'):
        return {'document': f'平均實際 AQI：{value}', 'cmetadata': {
            'station': station, 'source': SUMMARY_SOURCES[0], 'date': day, 'sample_count': 24}}

    # 確認整月摘要全數保留並依日期排序。
    def test_complete_month_sorted_without_top_k(self):
        documents = [self.document(f'2026-06-{day:02d}') for day in range(30, 0, -1)]
        result = _format_daily_series(documents, 2026, 6)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['rows']), 30)
        self.assertEqual(result['rows'][0]['date'], '2026-06-01')
        self.assertEqual(result['rows'][-1]['date'], '2026-06-30')
        self.assertEqual(result['rows'][0]['avg_actual_aqi'], 55)

    # 確認缺少日期或平均值時明確標示缺值，不當作零。
    def test_missing_dates_and_values_are_not_zero(self):
        result = _format_daily_series([
            self.document('2026-06-01'), self.document('2026-06-02', '無有效預測配對，無法計算')], 2026, 6)
        self.assertFalse(result['complete'])
        self.assertEqual(result['dates_without_average'], ['2026-06-02'])
        self.assertEqual(len(result['missing_dates']), 28)
        self.assertIsNone(result['rows'][2]['avg_actual_aqi'])

    # 確認其他測站或重複日期的摘要會被拒絕。
    def test_wrong_scope_or_duplicates_fail(self):
        for docs in ([self.document('2026-06-01', station='其他站')],
                     [self.document('2026-06-01')] * 2):
            with self.assertRaises(ValueError):
                _format_daily_series(docs, 2026, 6)

    @patch('agent.rag_tool._summary_engine')
    def test_later_years_query_correct_dates_and_preserve_missing_data(self, engine):
        connection = MagicMock()
        engine.return_value.connect.return_value.__enter__.return_value = connection
        connection.execute.return_value.mappings.return_value.all.return_value = []
        for year, month, end, days in ((2027, 12, '2028-01-01', 31),
                                       (2028, 2, '2028-03-01', 29)):
            with self.subTest(year=year, month=month):
                result = query_daily_series(year, month)
                params = connection.execute.call_args.args[1]
                self.assertEqual(params['start'], f'{year}-{month:02d}-01')
                self.assertEqual(params['end'], end)
                self.assertFalse(result['complete'])
                self.assertEqual(len(result['missing_dates']), days)
                self.assertTrue(all(row['avg_actual_aqi'] is None for row in result['rows']))

    @patch('agent.rag_tool._summary_engine')
    def test_years_before_2026_do_not_query_database(self, engine):
        result = query_daily_series(2025, 12)
        self.assertFalse(result['retryable'])
        self.assertIn('2026 年起', result['error'])
        engine.assert_not_called()

    # 確認整月每日平均需求直接取得序列，不呼叫語言模型。
    @patch('agent.rag_tool.query_daily_series')
    def test_tool_routes_month_series_without_llm(self, series):
        series.return_value = {'rows': [], 'complete': False}
        # 在誤觸模型時立即讓測試失敗，確保序列不經模型生成。
        def no_model(_):
            raise AssertionError('Series retrieval must not call an LLM')
        tool = KnowledgeBaseTool.create(RunnableLambda(no_model))
        result = json.loads(tool.invoke({'question': '將2026年6月份每日平均aqi資料做成趨勢圖'}))
        series.assert_called_once_with(2026, 6)
        self.assertFalse(result['complete'])

    # 確認非整月每日平均需求仍走一般 RAG 查詢。
    def test_other_questions_keep_normal_rag(self):
        for question in ('2026/6/9 平均AQI', '2026年6月平均AQI', '2026年6月每日MAE', '2026年6月哪天AQI最高'):
            self.assertIsNone(_daily_series_period(question))


if __name__ == '__main__':
    unittest.main()
