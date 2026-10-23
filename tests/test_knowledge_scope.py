"""Station-scoped knowledge base contracts; no external model calls."""
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from build_pgvector import DAILY_QUERY, MONTHLY_QUERY, _build_monthly_documents, _document_id
from agent import rag_tool


# 驗證知識庫文件與檢索條件維持指定測站範圍。
class KnowledgeScopeTests(unittest.TestCase):
    # 確認月摘要標示測站，且文件 ID 不因內容更新而改變。
    def test_monthly_document_labels_scope_and_stable_identity(self):
        row = dict(month=datetime(2024, 2, 1), avg_actual=42, avg_predicted=43,
                   mae=2, rmse=3, bias=1, sample_count=10)
        document = _build_monthly_documents([row])[0]
        self.assertEqual(document.metadata['station'], '木柵站')
        self.assertIn('測站：木柵站', document.page_content)
        original_id = _document_id(document)
        document.page_content += '\nupdated'
        self.assertEqual(original_id, _document_id(document))
        document.metadata['station'] = '另一站'
        self.assertNotEqual(original_id, _document_id(document))

    def test_summary_build_queries_have_no_end_year_limit(self):
        for query in (DAILY_QUERY, MONTHLY_QUERY):
            self.assertIn("DATE '2026-01-01'", query)
            self.assertNotIn("DATE '2027-01-01'", query)
            self.assertNotRegex(query, r"(?:target_at|observed_at)\s*<")

    # 確認指定日期的摘要來源標示測站。
    @patch('agent.rag_tool._query_summary_documents')
    def test_specific_date_labels_station_in_sources(self, documents):
        documents.return_value = [SimpleNamespace(
            page_content='平均實際 AQI：42',
            metadata={'source': rag_tool.SUMMARY_SOURCES[0], 'station': '木柵站', 'date': '2026-04-09'})]
        model = Mock()
        model.invoke.return_value = '42 [1]'
        _, sources = rag_tool.query_knowledge_base('2026/4/9 AQI', runnable=model)
        documents.assert_called_once_with('2026/4/9 AQI')
        self.assertIn('測站：木柵站', sources)


if __name__ == '__main__':
    unittest.main()
