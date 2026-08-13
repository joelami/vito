"""
Client for Open-Meteo's free historical weather archive (no API key, no
signup — `archive-api.open-meteo.com`). Used to join real game-day weather
onto historical games: temperature, wind, and precipitation are a validated
predictor of NFL totals (external research reviewed this session), and this
model previously had no weather signal at all.

One bulk request per unique (lat, lon), covering the FULL date range needed,
rather than one request per game — a handful of requests instead of
thousands, and it's the same archive endpoint already confirmed reachable
with no auth headers required.
"""

import json
import time
import urllib.request
import urllib.error

import pandas as pd

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily_weather(lat: float, lon: float, start_date: str, end_date: str,
                         max_retries: int = 5) -> pd.DataFrame:
    """Returns a DataFrame indexed by date (YYYY-MM-DD str) with columns
    temp_max_f, temp_min_f, wind_mph, precip_mm. Empty DataFrame only after
    exhausting retries — callers should treat that as "no weather data for
    this stadium" and fill neutral defaults, not crash. Retries with
    exponential backoff on HTTP 429: the free archive API rate-limits bursts
    (confirmed empirically — firing ~30 requests back-to-back for the NFL
    stadium set got most of them 429'd, which would have silently masked as
    "no wind ever" and corrupted the honest before/after measurement this
    feature is supposed to get)."""
    url = (
        f"{BASE_URL}?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,wind_speed_10m_max,precipitation_sum"
        f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
    )
    data = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16s
                print(f"[weather_client] 429 for ({lat},{lon}), retry {attempt + 1}/{max_retries} in {wait}s")
                time.sleep(wait)
                continue
            print(f"[weather_client] fetch failed for ({lat},{lon}) {start_date}..{end_date}: {e}")
            return pd.DataFrame(columns=["temp_max_f", "temp_min_f", "wind_mph", "precip_mm"])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"[weather_client] fetch failed for ({lat},{lon}) {start_date}..{end_date}: {e}")
            return pd.DataFrame(columns=["temp_max_f", "temp_min_f", "wind_mph", "precip_mm"])

    if data is None:
        return pd.DataFrame(columns=["temp_max_f", "temp_min_f", "wind_mph", "precip_mm"])

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        return pd.DataFrame(columns=["temp_max_f", "temp_min_f", "wind_mph", "precip_mm"])

    df = pd.DataFrame({
        "date": dates,
        "temp_max_f": daily.get("temperature_2m_max", [None] * len(dates)),
        "temp_min_f": daily.get("temperature_2m_min", [None] * len(dates)),
        "wind_mph": daily.get("wind_speed_10m_max", [None] * len(dates)),
        "precip_mm": daily.get("precipitation_sum", [None] * len(dates)),
    })
    return df.set_index("date")
