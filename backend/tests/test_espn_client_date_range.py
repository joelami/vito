"""
Unit tests for core/espn_client.py's fetch_scoreboard() date-range
splitting -- a real, external incident, 2026-09-18 (app owner: "Vito has
crashed... it's not refreshing"): ESPN's scoreboard endpoint silently
stopped accepting the `YYYYMMDD-YYYYMMDD` range form (confirmed directly
via curl against the real endpoint, every sport, even a 1-week range --
not this project's request being malformed). Since harness.py's real
sync window is always a range, this zeroed out every sport's ESPN sync
at once. Fixed by transparently splitting a range into one single-date
request per day (still real, still working) and merging the results --
see fetch_scoreboard()'s own docstring for the full incident writeup.

Mocks _get_json (not the network) so these are fast and deterministic --
the real, live 400-on-range behavior was confirmed by hand via curl and
via a real harness.sync_espn_games('NFL') call during the actual fix,
not re-verified here on every test run.
"""

from unittest.mock import patch

from core import espn_client


class TestFetchScoreboardDateRangeSplitting:
    def test_a_range_is_split_into_one_request_per_day(self):
        calls = []

        def fake_get_json(url, timeout=10.0):
            calls.append(url)
            return {"events": [{"id": url}]}  # one fake event per call, tagged by its own url

        with patch.object(espn_client, "_get_json", side_effect=fake_get_json):
            result = espn_client.fetch_scoreboard("NFL", dates="20260913-20260915")

        # 3-day inclusive range -> 3 real requests, one per day.
        assert len(calls) == 3
        assert calls[0].endswith("dates=20260913")
        assert calls[1].endswith("dates=20260914")
        assert calls[2].endswith("dates=20260915")

    def test_merged_result_concatenates_every_days_events(self):
        def fake_get_json(url, timeout=10.0):
            return {"events": [{"id": url}]}

        with patch.object(espn_client, "_get_json", side_effect=fake_get_json):
            result = espn_client.fetch_scoreboard("NFL", dates="20260913-20260915")

        assert len(result["events"]) == 3

    def test_a_day_with_no_events_does_not_break_the_merge(self):
        def fake_get_json(url, timeout=10.0):
            if url.endswith("20260914"):
                return {"events": []}  # a real off day -- no games scheduled
            return {"events": [{"id": url}]}

        with patch.object(espn_client, "_get_json", side_effect=fake_get_json):
            result = espn_client.fetch_scoreboard("NFL", dates="20260913-20260915")

        assert len(result["events"]) == 2

    def test_a_day_that_returns_none_does_not_crash_the_merge(self):
        # _get_json's own real failure-after-retries return value is {}
        # (see its docstring) -- simulate the worst real case, None, too.
        def fake_get_json(url, timeout=10.0):
            if url.endswith("20260914"):
                return None
            return {"events": [{"id": url}]}

        with patch.object(espn_client, "_get_json", side_effect=fake_get_json):
            result = espn_client.fetch_scoreboard("NFL", dates="20260913-20260915")

        assert len(result["events"]) == 2

    def test_a_single_date_is_NOT_split_unchanged_behavior(self):
        calls = []

        def fake_get_json(url, timeout=10.0):
            calls.append(url)
            return {"events": []}

        with patch.object(espn_client, "_get_json", side_effect=fake_get_json):
            espn_client.fetch_scoreboard("NFL", dates="20260913")

        assert len(calls) == 1
        assert calls[0].endswith("dates=20260913")

    def test_no_dates_at_all_is_NOT_split_unchanged_behavior(self):
        calls = []

        def fake_get_json(url, timeout=10.0):
            calls.append(url)
            return {"events": []}

        with patch.object(espn_client, "_get_json", side_effect=fake_get_json):
            espn_client.fetch_scoreboard("NFL")

        assert len(calls) == 1
        assert "dates=" not in calls[0]
