import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd
from pipeline_v2.data_cleaning_new import FEATURES_NEW, load_and_clean_data_new


class DatabaseCleaningV2Tests(unittest.TestCase):
    @patch('pipeline_v2.data_cleaning_new.create_engine')
    @patch('pipeline_v2.data_cleaning_new.read_database_observations')
    def test_default_range_and_original_interpolation(self, read, engine):
        read.return_value = pd.DataFrame({
            '日期': pd.to_datetime(['2025-12-31 23:00', '2025-12-31 21:00', '2025-12-31 22:00']),
            **{feature: [3, 1, None] for feature in FEATURES_NEW}})
        with contextlib.redirect_stdout(io.StringIO()):
            data, features = load_and_clean_data_new()
        _, station, start, end = read.call_args.args
        self.assertEqual(station, '木柵站')
        self.assertEqual(start, pd.Timestamp('2025-01-01'))
        self.assertEqual(end, pd.Timestamp('2026-01-01') - pd.Timedelta(microseconds=1))
        self.assertEqual(features, FEATURES_NEW)
        self.assertEqual(data['AQI'].tolist(), [1, 2, 3])
        engine.return_value.dispose.assert_called_once()

    @patch('pipeline_v2.data_cleaning_new.create_engine')
    def test_invalid_range_does_not_connect(self, engine):
        with self.assertRaises(ValueError):
            load_and_clean_data_new(start='2026-01-01', end='2025-12-31')
        engine.assert_not_called()

    @patch('pipeline_v2.data_cleaning_new.create_engine')
    @patch('pipeline_v2.data_cleaning_new.read_database_observations', return_value=pd.DataFrame())
    def test_empty_result(self, read, engine):
        with self.assertRaisesRegex(ValueError, '沒有觀測資料'):
            load_and_clean_data_new()
        engine.return_value.dispose.assert_called_once()

    @patch('pipeline_v2.data_cleaning_new.create_engine')
    @patch('pipeline_v2.data_cleaning_new.read_database_observations')
    def test_duplicate_times_still_rejected(self, read, engine):
        read.return_value = pd.DataFrame({'日期': ['2025-01-01'] * 2,
                                         **{feature: [1, 2] for feature in FEATURES_NEW}})
        with self.assertRaisesRegex(ValueError, '重複'):
            load_and_clean_data_new()


if __name__ == '__main__':
    unittest.main()
