"""
Joins historical game-day weather onto the games DataFrame. Domed/retractable
stadiums get fixed neutral values (climate-controlled — the whole point is
that outside weather doesn't reach the field), not real outside weather,
since feeding real weather into a game that was actually played indoors
would inject wrong signal, not real signal.

One weather-archive request per unique stadium location (not per game) —
see `core/weather_client.py`. Results are cached to disk (`weather_cache.csv`,
keyed by game_id) since the fetch takes ~30-60s even with rate-limit-friendly
spacing — re-running it on every server boot and every harness invocation
would be both slow and an unnecessary hammer on a free public API. Only
game_ids missing from the cache (i.e. new games added since the last run)
trigger a real fetch.

CACHE_PATH resolves the same way DB_PATH does (database.py) — an optional
env var, defaulting to a local sibling-of-this-file path. Real bug this
fixes, confirmed directly against a real Railway boot: the old hardcoded
in-repo path was never on the persistent Volume (only DB_PATH/Datasets
were), so every redeploy started with an empty cache and re-fetched the
ENTIRE 5,431-game historical archive from scratch — not "no NFL games are
live so why fetch weather," but a full historical re-fetch needed for
model TRAINING features, repeated on every single boot. That's real,
avoidable time (~10 minutes, rate-limited 429s) and resource churn during
the highest-risk window of startup. Setting WEATHER_CACHE_PATH to a path on
the same Volume as DB_PATH (e.g. /data/weather_cache.csv) makes this a
one-time cost, ever, matching the discipline this module's own docstring
already describes but the path never actually delivered on.
"""

import os
import time
from pathlib import Path

import pandas as pd

from . import stadiums
from core import weather_client

NEUTRAL_TEMP_F = 70.0
NEUTRAL_WIND_MPH = 0.0
NEUTRAL_PRECIP_MM = 0.0

CACHE_PATH = Path(os.environ.get("WEATHER_CACHE_PATH", Path(__file__).parent / "weather_cache.csv"))
CACHE_COLS = ["game_id", "is_dome", "game_temp_f", "game_wind_mph", "game_precip_mm"]


def _load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH, dtype={"game_id": str})
    return pd.DataFrame(columns=CACHE_COLS)


def attach_weather(games: pd.DataFrame) -> pd.DataFrame:
    cache = _load_cache()
    cached_ids = set(cache["game_id"]) if not cache.empty else set()
    to_fetch = games[~games["game_id"].isin(cached_ids)]

    if to_fetch.empty:
        new_rows = pd.DataFrame(columns=CACHE_COLS)
    else:
        new_rows = _fetch_weather_rows(to_fetch)
        if not new_rows.empty:
            combined = pd.concat([cache, new_rows], ignore_index=True)
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(CACHE_PATH, index=False)
            print(f"[weather] fetched {len(new_rows)} new game(s), cache now has {len(combined)} entries")

    all_weather = pd.concat([cache, new_rows], ignore_index=True) if not new_rows.empty else cache
    out = games.merge(all_weather, on="game_id", how="left")
    # any game_id still missing (e.g. every fetch for it failed) gets neutral fallback, not NaN
    for col, default in [("is_dome", False), ("game_temp_f", NEUTRAL_TEMP_F),
                          ("game_wind_mph", NEUTRAL_WIND_MPH), ("game_precip_mm", NEUTRAL_PRECIP_MM)]:
        out[col] = out[col].fillna(default)
    return out


def _fetch_weather_rows(games: pd.DataFrame) -> pd.DataFrame:
    stadium = [stadiums.stadium_for(fr, season) for fr, season in zip(games["home_franchise"], games["season"])]
    is_dome = [s[2] for s in stadium]

    min_date = games["date"].min().strftime("%Y-%m-%d")
    max_date = games["date"].max().strftime("%Y-%m-%d")

    outdoor_locations = sorted({(round(lat, 3), round(lon, 3)) for lat, lon, dome in stadium if not dome})
    weather_by_loc = {}
    failed_locations = []
    for i, (lat, lon) in enumerate(outdoor_locations):
        if i > 0:
            time.sleep(15.0)  # spread requests out — avoid re-triggering the burst rate limit
            # (1.2s worked cold, 6s still left ~12/25 failing on a repeat attempt within the same
            # session — this looks like a longer-window/session-level limit, not just short-burst;
            # widened again after that)
        wdf = weather_client.fetch_daily_weather(lat, lon, min_date, max_date)
        weather_by_loc[(lat, lon)] = wdf
        if wdf.empty:
            failed_locations.append((lat, lon))
    if failed_locations:
        print(f"[weather] WARNING: {len(failed_locations)}/{len(outdoor_locations)} stadium "
              f"weather fetches failed even after retries: {failed_locations} — those games will "
              f"get neutral fallback values, not real weather.")

    rows = []
    for (idx, row), (lat, lon, dome) in zip(games.iterrows(), stadium):
        if dome:
            temp, wind, precip = NEUTRAL_TEMP_F, NEUTRAL_WIND_MPH, NEUTRAL_PRECIP_MM
        else:
            wdf = weather_by_loc.get((round(lat, 3), round(lon, 3)))
            date_str = row["date"].strftime("%Y-%m-%d")
            if wdf is None or wdf.empty or date_str not in wdf.index:
                temp, wind, precip = NEUTRAL_TEMP_F, NEUTRAL_WIND_MPH, NEUTRAL_PRECIP_MM
            else:
                w = wdf.loc[date_str]
                temp = w["temp_max_f"] if pd.notna(w["temp_max_f"]) else NEUTRAL_TEMP_F
                wind = w["wind_mph"] if pd.notna(w["wind_mph"]) else NEUTRAL_WIND_MPH
                precip = w["precip_mm"] if pd.notna(w["precip_mm"]) else NEUTRAL_PRECIP_MM
        rows.append({"game_id": row["game_id"], "is_dome": dome, "game_temp_f": temp,
                      "game_wind_mph": wind, "game_precip_mm": precip})
    return pd.DataFrame(rows)


def current_stadium_row(franchise: str, season: int, game_date) -> dict:
    """
    For a brand-new (manual/synced) upcoming game: historical archive weather
    doesn't exist for a date that hasn't happened yet. Rather than call a
    separate forecast API (not yet wired — a documented next step, not done
    here), new games get neutral defaults, same as a domed game. This means
    the weather feature currently only sharpens HISTORICAL training/backtest
    quality, not live picks — an honest limitation, not a silent gap.
    """
    lat, lon, is_dome = stadiums.stadium_for(franchise, season)
    return {"game_temp_f": NEUTRAL_TEMP_F, "game_wind_mph": NEUTRAL_WIND_MPH,
            "game_precip_mm": NEUTRAL_PRECIP_MM, "is_dome": is_dome}
