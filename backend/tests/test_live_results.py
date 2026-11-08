"""
Unit tests for core/live_results.py's with_live_results() -- specifically
a real, reproduced bug found and fixed 2026-10-15 while verifying the NHL
MoneyPuck rewrite end-to-end: fetch_completed_espn_results() always
returns a tz-NAIVE `date` column (correct for NFL/CFB/MLB/NBA, whose own
bulk loaders are all tz-naive) -- but NHL's bulk loader's own `date`
column is tz-AWARE. Concatenating tz-naive live rows onto a tz-aware
`games` frame silently downgraded the combined `date` column to `object`
dtype (a genuine mix of tz-aware and tz-naive Timestamps), which then
crashed core/power_ratings.py's own sort_values() with a real
`TypeError: Cannot compare tz-naive and tz-aware timestamps` -- meaning
NHL's Elo ratings were silently never seeing any of this season's real
completed games via this path at all.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from core import live_results


def _fake_live_df(tz_aware: bool):
    date = pd.Timestamp("2026-10-10")
    if tz_aware:
        date = date.tz_localize("UTC")
    return pd.DataFrame([{
        "home_franchise": "A", "away_franchise": "B", "home_score": 3, "away_score": 2,
        "date": date, "season": 2026, "is_neutral_venue": False, "home_win": 1, "game_id": "live_1",
    }])


class TestWithLiveResultsTzHandling:
    def test_tz_aware_target_gets_tz_naive_live_rows_localized_to_match(self):
        games = pd.DataFrame([{
            "game_id": "g0", "date": pd.Timestamp("2026-01-01", tz="UTC"), "home_franchise": "X", "away_franchise": "Y",
        }])
        with patch("core.live_results.fetch_completed_espn_results", return_value=_fake_live_df(tz_aware=False)):
            out = live_results.with_live_results(games, "NHL")
        assert out["date"].dt.tz is not None
        # Real regression check: this must not raise when sorted, the
        # exact operation that originally crashed.
        out.sort_values("date", kind="stable")

    def test_tz_naive_target_gets_tz_aware_live_rows_stripped_to_match(self):
        games = pd.DataFrame([{
            "game_id": "g0", "date": pd.Timestamp("2026-01-01"), "home_franchise": "X", "away_franchise": "Y",
        }])
        with patch("core.live_results.fetch_completed_espn_results", return_value=_fake_live_df(tz_aware=True)):
            out = live_results.with_live_results(games, "NFL")
        assert out["date"].dt.tz is None
        out.sort_values("date", kind="stable")

    def test_result_is_a_real_single_datetime64_dtype_not_object(self):
        # The actual symptom of the original bug: a mixed-tz concat
        # silently downgrades to `object` dtype instead of raising
        # immediately, so this is worth asserting explicitly, not just
        # checking that sort_values() happens to work.
        games = pd.DataFrame([{
            "game_id": "g0", "date": pd.Timestamp("2026-01-01", tz="UTC"), "home_franchise": "X", "away_franchise": "Y",
        }])
        with patch("core.live_results.fetch_completed_espn_results", return_value=_fake_live_df(tz_aware=False)):
            out = live_results.with_live_results(games, "NHL")
        assert out["date"].dtype != object
        assert str(out["date"].dtype).startswith("datetime64")

    def test_already_matching_tz_awareness_is_left_alone(self):
        games = pd.DataFrame([{
            "game_id": "g0", "date": pd.Timestamp("2026-01-01", tz="UTC"), "home_franchise": "X", "away_franchise": "Y",
        }])
        with patch("core.live_results.fetch_completed_espn_results", return_value=_fake_live_df(tz_aware=True)):
            out = live_results.with_live_results(games, "NHL")
        assert out["date"].dt.tz is not None
        assert len(out) == 2

    def test_empty_live_result_returns_games_unchanged(self):
        games = pd.DataFrame([{
            "game_id": "g0", "date": pd.Timestamp("2026-01-01", tz="UTC"), "home_franchise": "X", "away_franchise": "Y",
        }])
        with patch("core.live_results.fetch_completed_espn_results", return_value=pd.DataFrame()):
            out = live_results.with_live_results(games, "NHL")
        assert out is games
