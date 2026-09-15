"""
Unit tests for sports/mlb/park_factor.py -- hand-derived expected values,
not just "it ran" (see tests/test_nflfastr_epa_sos_adjusted.py for the
style this follows).

Four games, chronological, two parks (P1, P2):
  g1  d1  P1  total=10  home=TeamA
  g2  d2  P2  total=6   home=TeamB
  g3  d3  P1  total=8   home=TeamA
  g4  d4  P1  total=12  home=TeamA

Hand-derived trailing league average (shift(1), all games, min_periods=1):
  g1: no prior game at all -> NaN -> LEAGUE_AVG_TOTAL_RUNS fallback (9.17)
  g2: mean(prior: [10]) = 10.0
  g3: mean(prior: [10, 6]) = 8.0
  g4: mean(prior: [10, 6, 8]) = 8.0

Hand-derived trailing park average (shift(1), grouped by park_id, min_periods=1):
  g1 (P1): no prior P1 game -> NaN
  g2 (P2): no prior P2 game -> NaN
  g3 (P1): mean(prior P1: [10]) = 10.0
  g4 (P1): mean(prior P1: [10, 8]) = 9.0

park_scoring_factor = park_avg / league_avg, NaN (either side) -> 1.0:
  g1: NaN / 9.17            -> 1.0 (fallback)
  g2: NaN / 10.0             -> 1.0 (fallback)
  g3: 10.0 / 8.0             -> 1.25
  g4: 9.0 / 8.0              -> 1.125
"""

import pandas as pd
import pytest

from sports.mlb import park_factor as pf


def _synthetic_games():
    return pd.DataFrame([
        {"game_id": "g1", "date": pd.Timestamp("2020-04-01"), "park_id": "P1", "actual_total": 10,
         "home_franchise": "TeamA"},
        {"game_id": "g2", "date": pd.Timestamp("2020-04-02"), "park_id": "P2", "actual_total": 6,
         "home_franchise": "TeamB"},
        {"game_id": "g3", "date": pd.Timestamp("2020-04-03"), "park_id": "P1", "actual_total": 8,
         "home_franchise": "TeamA"},
        {"game_id": "g4", "date": pd.Timestamp("2020-04-04"), "park_id": "P1", "actual_total": 12,
         "home_franchise": "TeamA"},
    ])


class TestAttachParkFactor:
    def test_hand_derived_values_match_exactly(self):
        out = pf.attach_park_factor(_synthetic_games())
        by_id = out.set_index("game_id")["park_scoring_factor"]
        assert by_id["g1"] == pytest.approx(1.0, abs=1e-9)
        assert by_id["g2"] == pytest.approx(1.0, abs=1e-9)
        assert by_id["g3"] == pytest.approx(1.25, abs=1e-9)
        assert by_id["g4"] == pytest.approx(1.125, abs=1e-9)

    def test_walk_forward_safe_shuffled_input_order_gives_same_result(self):
        games = _synthetic_games().sample(frac=1, random_state=7).reset_index(drop=True)
        out = pf.attach_park_factor(games)
        by_id = out.set_index("game_id")["park_scoring_factor"]
        assert by_id["g3"] == pytest.approx(1.25, abs=1e-9)
        assert by_id["g4"] == pytest.approx(1.125, abs=1e-9)

    def test_never_drops_or_reorders_existing_rows(self):
        games = _synthetic_games()
        games["extra_col"] = ["keep0", "keep1", "keep2", "keep3"]
        out = pf.attach_park_factor(games)
        assert len(out) == 4
        assert list(out["game_id"]) == ["g1", "g2", "g3", "g4"]
        assert list(out["extra_col"]) == ["keep0", "keep1", "keep2", "keep3"]

    def test_a_parks_true_first_game_ever_falls_back_to_neutral(self):
        out = pf.attach_park_factor(_synthetic_games())
        assert out.set_index("game_id").loc["g1", "park_scoring_factor"] == 1.0
        assert out.set_index("game_id").loc["g2", "park_scoring_factor"] == 1.0


class TestCurrentParkFactorSnapshot:
    def test_hand_derived_snapshot_matches_tail_not_shifted(self):
        # Unlike attach_park_factor (walk-forward, shift(1)), the snapshot
        # INCLUDES each park's/the league's most recent game:
        #   P1 avg now = mean(10, 8, 12) = 10.0
        #   P2 avg now = mean(6) = 6.0
        #   league avg now = mean(10, 6, 8, 12) = 9.0
        #   TeamA's latest home park = P1 (g4) -> 10.0 / 9.0 = 1.11111...
        #   TeamB's latest home park = P2 (g2) -> 6.0 / 9.0 = 0.66667
        snap = pf.current_park_factor_snapshot(_synthetic_games())
        assert snap.loc["TeamA", "park_scoring_factor"] == pytest.approx(10.0 / 9.0, abs=1e-6)
        assert snap.loc["TeamB", "park_scoring_factor"] == pytest.approx(6.0 / 9.0, abs=1e-6)


class TestGetCurrentParkFactor:
    def test_uses_cache_and_matches_snapshot(self, monkeypatch):
        pf.reset_cache()
        monkeypatch.setattr(pf, "load_games", None, raising=False)
        import sports.mlb.loader as loader_mod
        monkeypatch.setattr(loader_mod, "load_games", lambda: _synthetic_games())
        try:
            result = pf.get_current_park_factor("TeamA")
            assert result == pytest.approx(10.0 / 9.0, abs=1e-6)
            # second call must reuse the cache, not re-invoke load_games
            monkeypatch.setattr(loader_mod, "load_games", lambda: (_ for _ in ()).throw(AssertionError("cache miss")))
            result2 = pf.get_current_park_factor("TeamA")
            assert result2 == pytest.approx(10.0 / 9.0, abs=1e-6)
        finally:
            pf.reset_cache()

    def test_unknown_franchise_gets_neutral_not_a_crash(self, monkeypatch):
        pf.reset_cache()
        import sports.mlb.loader as loader_mod
        monkeypatch.setattr(loader_mod, "load_games", lambda: _synthetic_games())
        try:
            assert pf.get_current_park_factor("Some Team Not In The Data") == 1.0
        finally:
            pf.reset_cache()
