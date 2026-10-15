"""固定查詢木柵天氣預報的 LangChain 工具。"""

from __future__ import annotations

import json
import logging
from textwrap import dedent
from typing import Any

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
MUZHA_LATITUDE = 24.99823
MUZHA_LONGITUDE = 121.573125
TIMEZONE = "Asia/Taipei"

WEATHER_DESCRIPTIONS = {
    0: "晴朗",
    1: "大致晴朗",
    2: "局部多雲",
    3: "陰天",
    45: "霧",
    48: "霧淞霧",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "強毛毛雨",
    56: "小凍毛毛雨",
    57: "強凍毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "小凍雨",
    67: "強凍雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "霰",
    80: "小陣雨",
    81: "中等陣雨",
    82: "強陣雨",
    85: "小陣雪",
    86: "強陣雪",
    95: "雷雨",
    96: "雷雨伴隨小冰雹",
    99: "雷雨伴隨強冰雹",
}


class WeatherForecastInput(BaseModel):
    forecast_days: int = Field(
        default=3,
        ge=1,
        le=7,
        description="要取得的木柵逐日預報天數，範圍為 1 到 7 天",
    )


def _value_at(values: dict, key: str, index: int):
    items = values.get(key, [])
    return items[index] if isinstance(items, list) and index < len(items) else None


def _weather_description(code) -> str:
    try:
        return WEATHER_DESCRIPTIONS.get(int(code), "未知天氣狀況")
    except (TypeError, ValueError):
        return "未知天氣狀況"


def _format_forecast(payload: dict) -> dict:
    """將 Open-Meteo 回應整理成穩定且適合 Agent 使用的 JSON 結構。"""
    current = payload.get("current") or {}
    daily = payload.get("daily") or {}
    dates = daily.get("time") if isinstance(daily.get("time"), list) else []

    daily_forecast = []
    for index, forecast_date in enumerate(dates):
        weather_code = _value_at(daily, "weather_code", index)
        daily_forecast.append({
            "date": forecast_date,
            "weather": _weather_description(weather_code),
            "weather_code": weather_code,
            "temperature_max_c": _value_at(daily, "temperature_2m_max", index),
            "temperature_min_c": _value_at(daily, "temperature_2m_min", index),
            "apparent_temperature_max_c": _value_at(
                daily, "apparent_temperature_max", index),
            "apparent_temperature_min_c": _value_at(
                daily, "apparent_temperature_min", index),
            "precipitation_probability_max_percent": _value_at(
                daily, "precipitation_probability_max", index),
            "precipitation_sum_mm": _value_at(daily, "precipitation_sum", index),
            "wind_speed_max_kmh": _value_at(daily, "wind_speed_10m_max", index),
            "wind_gusts_max_kmh": _value_at(daily, "wind_gusts_10m_max", index),
            "sunrise": _value_at(daily, "sunrise", index),
            "sunset": _value_at(daily, "sunset", index),
        })

    current_code = current.get("weather_code")
    return {
        "source": "Open-Meteo Forecast API",
        "location": {
            "name": "木柵",
            "district": "臺北市文山區",
            "latitude": MUZHA_LATITUDE,
            "longitude": MUZHA_LONGITUDE,
        },
        "timezone": payload.get("timezone", TIMEZONE),
        "current": {
            "time": current.get("time"),
            "weather": _weather_description(current_code),
            "weather_code": current_code,
            "temperature_c": current.get("temperature_2m"),
            "apparent_temperature_c": current.get("apparent_temperature"),
            "relative_humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_direction_degrees": current.get("wind_direction_10m"),
        },
        "daily_forecast": daily_forecast,
    }


class WeatherForecastTool(BaseTool):
    """不使用模型，直接向固定 API 查詢木柵天氣。"""

    name: str = "MuzhaWeatherForecast"
    description: str = dedent("""\
        查詢臺北市文山區木柵的目前天氣與未來 1 到 7 天逐日預報。
        資料直接來自 Open-Meteo Forecast API，不使用語言模型產生或推測數值。

        適合查詢木柵的氣溫、體感溫度、降雨機率、雨量、風速、日出與日落。
        地點固定為木柵，不能用此工具查詢其他區域，也不提供歷史天氣。
    """)
    args_schema: type[BaseModel] = WeatherForecastInput

    timeout: float = 10.0
    session: Any = Field(default=None, exclude=True)

    def _run(self, forecast_days: int = 3) -> str:
        params = {
            "latitude": MUZHA_LATITUDE,
            "longitude": MUZHA_LONGITUDE,
            "timezone": TIMEZONE,
            "forecast_days": forecast_days,
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
            ]),
            "daily": ",".join([
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "precipitation_probability_max",
                "precipitation_sum",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "sunrise",
                "sunset",
            ]),
        }

        try:
            http_client = self.session or requests
            response = http_client.get(
                FORECAST_API_URL,
                params=params,
                headers={"User-Agent": "DL-Presentation-WeatherTool/1.0"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("error"):
                raise ValueError("unexpected weather API response")
            result = _format_forecast(payload)
        except (requests.RequestException, OSError, ValueError, TypeError) as exc:
            logging.getLogger(__name__).error(
                "木柵天氣 API 查詢失敗 (%s)", type(exc).__name__)
            result = {
                "error": "目前無法取得木柵天氣預報，請稍後重試。",
                "source": "Open-Meteo Forecast API",
                "location": "木柵",
            }
        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, forecast_days: int = 3) -> str:
        import asyncio

        return await asyncio.to_thread(self._run, forecast_days)
