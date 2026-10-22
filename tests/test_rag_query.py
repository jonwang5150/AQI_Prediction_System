"""不呼叫外部服務的 RAG 邊界與介面整合測試。"""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agent import rag_tool


# 離線驗證 RAG 日期篩選、來源引用、錯誤處理與介面整合。
class RagQueryTests(unittest.TestCase):
    # 確認知識庫工具使用外部傳入模型，並回傳摘要來源。
    @patch("agent.rag_tool._vector_store")
    def test_tool_uses_external_model(self, store):
        store.return_value.similarity_search.return_value = [SimpleNamespace(
            page_content="MAE：3.2",
            metadata={"source": rag_tool.SUMMARY_SOURCES[0], "date": "2026-10-14"},
        )]
        model = RunnableLambda(lambda _: AIMessage(content="外部模型回答 [1]"))

        result = json.loads(
            rag_tool.KnowledgeBaseTool.create(model).invoke({"question": "MAE？"})
        )

        self.assertEqual(result["answer"], "外部模型回答 [1]")
        self.assertIn("MAE：3.2", result["references"])

    # 確認明確年月只選取對應月份的月摘要。
    def test_explicit_month_selects_monthly_documents(self):
        for question in ("2026 年 8月的平均實際 AQI 與平均預測 AQI 差多少？", "2026-08 的平均 AQI", "2026/8 平均 AQI"):
            with self.subTest(question=question):
                query_filter, label = rag_tool._question_filter(question)
                self.assertEqual(query_filter, {"$and": [
                    {"source": rag_tool.SUMMARY_SOURCES[1]}, {"month": {"$like": "2026-08-%"}}]})

    # 確認完整日期會查日摘要，不誤判為月份查詢。
    def test_daily_date_is_not_mistaken_for_month(self):
        for question in ("2026 年 8 月 7 日的 MAE", "2026-08-07 MAE", "2026/8/7 MAE"):
            query_filter, _ = rag_tool._question_filter(question)
            self.assertEqual(query_filter, {"$and": [
                {"source": rag_tool.SUMMARY_SOURCES[0]}, {"date": "2026-08-07"}]})

    # 確認指定月份無摘要時不改查其他期間或呼叫模型。
    @patch("agent.rag_tool._vector_store")
    def test_missing_month_does_not_fall_back_to_other_periods(self, store):
        store.return_value.similarity_search.return_value = []
        runnable = Mock()
        answer, _ = rag_tool.query_knowledge_base(
            "2026 年 8 月平均 AQI", runnable=runnable)
        self.assertIn("2026-08", answer)
        self.assertIn("找不到", answer)
        runnable.invoke.assert_not_called()
        self.assertEqual(store.return_value.similarity_search.call_count, 1)

    # 確認無效日期在連線向量庫前即被拒絕。
    @patch("agent.rag_tool._vector_store")
    def test_invalid_date_does_not_query(self, store):
        answer, _ = rag_tool.query_knowledge_base(
            "2026-02-30 的 AQI", runnable=Mock())
        self.assertIn("日期無效", answer)
        store.assert_not_called()

    # 確認空白問題不會建立向量庫連線。
    @patch("agent.rag_tool._vector_store")
    def test_blank_question_does_not_connect(self, store):
        answer, sources = rag_tool.query_knowledge_base("  ", runnable=Mock())
        self.assertIn("請先輸入", answer)
        self.assertEqual(sources, "")
        store.assert_not_called()

    # 確認沒有檢索結果時不要求模型產生回答。
    @patch("agent.rag_tool._vector_store")
    def test_empty_collection_does_not_call_model(self, store):
        store.return_value.similarity_search.return_value = []
        runnable = Mock()
        answer, sources = rag_tool.query_knowledge_base("MAE？", runnable=runnable)
        self.assertIn("尚無", answer)
        runnable.invoke.assert_not_called()

    # 確認回答使用檢索摘要，且查詢限於允許的測站與來源。
    @patch("agent.rag_tool._vector_store")
    def test_grounded_answer_and_source_filter(self, store):
        store.return_value.similarity_search.return_value = [SimpleNamespace(
            page_content="MAE：3.2", metadata={"source": rag_tool.SUMMARY_SOURCES[0], "date": "2026-10-14"})]
        runnable = Mock()
        runnable.invoke.return_value = "MAE 為 3.2 [1]"
        answer, sources = rag_tool.query_knowledge_base("MAE？", runnable=runnable)
        self.assertIn("3.2 [1]", answer)
        self.assertIn("2026-10-14", sources)
        self.assertEqual(store.return_value.similarity_search.call_args.kwargs["filter"],
                         {"$and": [{"station": "木柵站"}, {"source": {"$in": rag_tool.SUMMARY_SOURCES}}]})
        self.assertIn("MAE：3.2", runnable.invoke.call_args.args[0]["sources"])

    # 確認模型失敗時仍保留摘要來源，且隱藏內部錯誤內容。
    @patch("agent.rag_tool._vector_store")
    def test_model_failure_preserves_sources_without_exposing_error(self, store):
        store.return_value.similarity_search.return_value = [SimpleNamespace(
            page_content="MAE：3.2", metadata={})]
        runnable = Mock()
        runnable.invoke.side_effect = ConnectionError("secret-connection-string")
        answer, sources = rag_tool.query_knowledge_base("MAE？", runnable=runnable)
        self.assertIn("產生回答失敗", answer)
        self.assertNotIn("secret", answer)
        self.assertIn("MAE：3.2", sources)

    # 確認儀表板含助理與既有圖表，且檢視切換的輸出正確。
    def test_dashboard_builds_with_rag_and_existing_charts(self):
        from prediction_visualization.visualizations import build_prediction_app
        data = pd.DataFrame({"日期": pd.date_range("2026-10-14", periods=3, freq="h"),
                             "預測AQI": [30, 35, 40], "實際AQI": [31, 34, 42]})
        app = build_prediction_app({"single": data, "8h": data},
                                   update_callback=lambda progress: "完成", reload_callback=lambda: {})
        values = [item.get("props", {}).get("value") for item in app.config["components"]]
        self.assertIn("智能助理", values)
        self.assertIn("送出訊息", values)
        self.assertIn("清除對話", values)
        self.assertIn("更新至最新", values)
        self.assertTrue(any(getattr(fn.fn, "__name__", None) == 'respond'
                            for fn in app.fns.values()))
        views = {fn.fn.args[0]: fn for fn in app.fns.values()
                 if getattr(fn.fn, "args", ()) and getattr(fn.fn.func, "__name__", "") == "select_view"}
        rag_result = views["rag"].fn()
        self.assertEqual(len(rag_result), len(views["rag"].outputs))
        self.assertFalse(rag_result[-2]["visible"])
        self.assertTrue(rag_result[-1]["visible"])
        metrics_result = views["metrics"].fn()
        self.assertTrue(metrics_result[-2]["visible"])
        self.assertFalse(metrics_result[-1]["visible"])
        self.assertTrue(metrics_result[1]["visible"])
        self.assertNotIn("收合查詢", values)
        app.close()

    # 確認雲端服務錯誤提供 HTTP 狀態，且不洩漏上游細節。
    @patch("agent.rag_tool._vector_store")
    def test_cloud_service_error_is_actionable(self, store):
        from ollama import ResponseError
        store.return_value.similarity_search.return_value = [SimpleNamespace(
            page_content="MAE：3.2", metadata={})]
        runnable = Mock()
        runnable.invoke.side_effect = ResponseError("upstream details", status_code=502)
        answer, sources = rag_tool.query_knowledge_base("MAE？", runnable=runnable)
        self.assertIn("HTTP 502", answer)
        self.assertNotIn("upstream details", answer)
        self.assertIn("MAE：3.2", sources)


if __name__ == "__main__":
    unittest.main()
