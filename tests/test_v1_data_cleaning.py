import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd
from pipeline_v1.data_cleaning import FEATURES, load_and_clean_data


# 驗證單步模型的資料庫讀取範圍與清理邊界。
class DatabaseCleaningTests(unittest.TestCase):
    # 確認預設區間包含結束日全天，保留欄位並依時間排序。
    @patch('pipeline_v1.data_cleaning.create_engine')
    @patch('pipeline_v1.data_cleaning.read_database_observations')
    def test_default_period_includes_last_day_and_keeps_columns(self, read, engine):
        read.return_value = pd.DataFrame({
            '日期': pd.to_datetime(['2025-12-31 23:00', '2025-01-01 00:00']),
            'AQI': [20, 30], **{feature: [1, 2] for feature in FEATURES}})
        with contextlib.redirect_stdout(io.StringIO()):
            data, features = load_and_clean_data()
        _, station, start, end = read.call_args.args
        self.assertEqual(station, '木柵站')
        self.assertEqual(start, pd.Timestamp('2025-01-01'))
        self.assertEqual(end, pd.Timestamp('2026-01-01') - pd.Timedelta(microseconds=1))
        self.assertEqual(features, FEATURES)
        self.assertTrue(data['日期'].is_monotonic_increasing)
        self.assertIn('AQI', data)
        engine.return_value.dispose.assert_called_once()

    # 確認起訖日期顛倒時在連線前拒絕執行。
    @patch('pipeline_v1.data_cleaning.create_engine')
    def test_invalid_period_does_not_connect(self, engine):
        with self.assertRaises(ValueError):
            load_and_clean_data(start='2025-12-31', end='2025-01-01')
        engine.assert_not_called()

    # 確認查無觀測資料時明確報錯並釋放引擎。
    @patch('pipeline_v1.data_cleaning.create_engine')
    @patch('pipeline_v1.data_cleaning.read_database_observations', return_value=pd.DataFrame())
    def test_empty_database_result_is_explicit(self, read, engine):
        with self.assertRaisesRegex(ValueError, '沒有觀測資料'):
            load_and_clean_data()
        engine.return_value.dispose.assert_called_once()


if __name__ == '__main__':
    unittest.main()
