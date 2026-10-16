"""不呼叫外部網路的木柵天氣工具測試。"""

import json
import unittest
from unittest.mock import Mock

import requests

from agent.weather_tool import (
    FORECAST_API_URL,
    MUZHA_LATITUDE,
    MUZHA_LONGITUDE,
    WeatherForecastTool,
)


class WeatherForecastToolTests(unittest.TestCase):
    def test_returns_structured_muzha_forecast(self):
        response = Mock()
        response.json.return_value = {
            "timezone": "Asia/Taipei",
            "current": {
                "time": "2026-10-15T20:00",
                "temperature_2m": 25.1,
                "relative_humidity_2m": 78,
                "apparent_temperature": 27.0,
                "precipitation": 0.0,
                "weather_code": 2,
                "wind_speed_10m": 8.2,
                "wind_direction_10m": 90,
            },
            "daily": {
                "time": ["2026-10-15", "2026-10-16"],
                "weather_code": [2, 61],
                "temperature_2m_max": [29.0, 27.0],
                "temperature_2m_min": [22.0, 21.0],
                "apparent_temperature_max": [31.0, 29.0],
                "apparent_temperature_min": [23.0, 22.0],
                "precipitation_probability_max": [20, 70],
                "precipitation_sum": [0.0, 8.5],
                "wind_speed_10m_max": [12.0, 18.0],
                "wind_gusts_10m_max": [20.0, 30.0],
                "sunrise": ["2026-10-15T05:52", "2026-10-16T05:52"],
                "sunset": ["2026-10-15T17:28", "2026-10-16T17:27"],
            },
        }
        session = Mock()
        session.get.return_value = response

        result = json.loads(
            WeatherForecastTool(session=session).invoke({"forecast_days": 2})
        )

        self.assertEqual(result["location"]["name"], "木柵")
        self.assertEqual(result["current"]["weather"], "局部多雲")
        self.assertEqual(result["daily_forecast"][1]["weather"], "小雨")
        _, kwargs = session.get.call_args
        self.assertEqual(session.get.call_args.args[0], FORECAST_API_URL)
        self.assertEqual(kwargs["params"]["latitude"], MUZHA_LATITUDE)
        self.assertEqual(kwargs["params"]["longitude"], MUZHA_LONGITUDE)
        self.assertNotIn("location", kwargs["params"])

    def test_network_failure_returns_safe_error(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError("private details")

        result = json.loads(WeatherForecastTool(session=session).invoke({}))

        self.assertIn("error", result)
        self.assertNotIn("private details", result["error"])
        self.assertEqual(result["location"], "木柵")


if __name__ == "__main__":
    unittest.main()
