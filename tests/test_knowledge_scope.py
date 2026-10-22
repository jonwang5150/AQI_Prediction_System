"""Station-scoped knowledge base contracts; no external model calls."""
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from build_pgvector import _build_monthly_documents, _document_id
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

    # 確認指定日期檢索同時套用木柵站條件並標示來源測站。
    @patch('agent.rag_tool._vector_store')
    def test_station_filter_combines_with_specific_date(self, store):
        store.return_value.similarity_search.return_value = [SimpleNamespace(
            page_content='平均實際 AQI：42',
            metadata={'source': rag_tool.SUMMARY_SOURCES[0], 'station': '木柵站', 'date': '2026-04-09'})]
        model = Mock()
        model.invoke.return_value = '42 [1]'
        _, sources = rag_tool.query_knowledge_base('2026/4/9 AQI', runnable=model)
        self.assertEqual(store.return_value.similarity_search.call_args.kwargs['filter'], {
            '$and': [{'station': '木柵站'}, {'$and': [
                {'source': rag_tool.SUMMARY_SOURCES[0]}, {'date': '2026-04-09'}]}]})
        self.assertIn('測站：木柵站', sources)


if __name__ == '__main__':
    unittest.main()
