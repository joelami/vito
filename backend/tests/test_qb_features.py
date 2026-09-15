"""
Unit tests for sports/nfl/qb_features.py -- the first PLAYER-category
feature module this project has built (see that module's own docstring
for its REJECTED hypothesis-test status: tested and not wired into
ML_FEATURE_COLS, kept as real, tested infrastructure for a future
re-test). Structurally mirrors tests/test_nflfastr_features.py.

Uses a small synthetic qb_starts CSV (monkeypatched DATA_PATH) rather
than the real ~11k-row fetched dataset.
"""

import pandas as pd
import pytest

from sports.nfl import qb_features as qf


@pytest.fixture
def synthetic_qb_csv(tmp_path, monkeypatch):
    """Kansas City Chiefs start the SAME QB (Q1) for 3 games, then a
    DIFFERENT QB (Q2) for a 4th -- makes both qb_epa_trail (mean of
    strictly-prior games) and qb_continuity_flag (same-starter-as-two-
    games-ago) trivially checkable by hand."""
    rows = [
        {"game_id": "g0", "season": 2024, "team": "KC", "starting_qb_id": "Q1",
         "starting_qb_name": "Q One", "qb_epa_per_dropback": 0.10, "qb_dropbacks": 30, "gameday": "2024-09-08"},
        {"game_id": "g1", "season": 2024, "team": "KC", "starting_qb_id": "Q1",
         "starting_qb_name": "Q One", "qb_epa_per_dropback": 0.20, "qb_dropbacks": 32, "gameday": "2024-09-15"},
        {"game_id": "g2", "season": 2024, "team": "KC", "starting_qb_id": "Q1",
         "starting_qb_name": "Q One", "qb_epa_per_dropback": 0.30, "qb_dropbacks": 28, "gameday": "2024-09-22"},
        {"game_id": "g3", "season": 2024, "team": "KC", "starting_qb_id": "Q2",
         "starting_qb_name": "Q Two", "qb_epa_per_dropback": -0.10, "qb_dropbacks": 35, "gameday": "2024-09-29"},
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "qb_starts.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(qf, "DATA_PATH", path)
    qf._qb_starts_cache = None  # reset the module-level cache between tests
    yield path
    qf._qb_starts_cache = None


class TestLoadQbStarts:
    def test_maps_team_code_to_real_franchise_name(self, synthetic_qb_csv):
        df = qf.load_qb_starts()
        assert (df["franchise"] == "Kansas City Chiefs").all()

    def test_missing_file_raises_a_clear_error_not_a_silent_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qf, "DATA_PATH", tmp_path / "does_not_exist.csv")
        with pytest.raises(FileNotFoundError):
            qf.load_qb_starts()


class TestBuildQbContinuityFeatures:
    def test_epa_trail_is_the_mean_of_STRICTLY_PRIOR_games_only(self, synthetic_qb_csv):
        # KC's 4th game (qb_epa_per_dropback=-0.10, a new QB): the trailing
        # feature for THAT game must be the mean of games 1-3 (0.10, 0.20,
        # 0.30 -> 0.20), never including -0.10 itself.
        games = pd.DataFrame([
            {"game_id": "vg0", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
             "date": pd.Timestamp("2024-09-08")},
            {"game_id": "vg1", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
             "date": pd.Timestamp("2024-09-15")},
            {"game_id": "vg2", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
             "date": pd.Timestamp("2024-09-22")},
            {"game_id": "vg3", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
             "date": pd.Timestamp("2024-09-29")},
        ])
        out = qf.build_qb_continuity_features(games)
        assert out["home_qb_epa_trail"].iloc[3] == pytest.approx(0.20)

    def test_continuity_flag_detects_a_real_qb_change(self, synthetic_qb_csv):
        # Game 4 (index 3): the prior two games (index 1, 2) both started
        # Q1 -- continuity_flag for game 4 should be 1 (stable going IN).
        # This is intentionally NOT about game 4's own starter (Q2) --
        # see this module's docstring for why that's never used.
        games = pd.DataFrame([
            {"game_id": f"vg{i}", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
             "date": pd.Timestamp(d)}
            for i, d in enumerate(["2024-09-08", "2024-09-15", "2024-09-22", "2024-09-29"])
        ])
        out = qf.build_qb_continuity_features(games)
        assert out["home_qb_continuity_flag"].iloc[3] == 1.0

    def test_first_two_games_get_a_neutral_fallback_not_a_fabricated_claim(self, synthetic_qb_csv):
        games = pd.DataFrame([
            {"game_id": "vg0", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
             "date": pd.Timestamp("2024-09-08")},
        ])
        out = qf.build_qb_continuity_features(games)
        assert out["home_qb_continuity_flag"].iloc[0] == 0.5

    def test_unmapped_team_gets_league_average_fallback_not_a_crash(self, synthetic_qb_csv):
        games = pd.DataFrame([
            {"game_id": "vg0", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
             "date": pd.Timestamp("2024-09-08")},
        ])
        out = qf.build_qb_continuity_features(games)
        league_avg = qf.load_qb_starts()["qb_epa_per_dropback"].mean()
        assert out["away_qb_epa_trail"].iloc[0] == pytest.approx(league_avg)

    def test_never_drops_or_reorders_existing_rows(self, synthetic_qb_csv):
        games = pd.DataFrame([
            {"game_id": "vg0", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
             "date": pd.Timestamp("2024-09-08"), "extra_col": "keep me"},
            {"game_id": "vg1", "home_franchise": "Denver Broncos", "away_franchise": "Kansas City Chiefs",
             "date": pd.Timestamp("2024-09-15"), "extra_col": "keep me too"},
        ])
        out = qf.build_qb_continuity_features(games)
        assert len(out) == 2
        assert list(out["game_id"]) == ["vg0", "vg1"]
        assert list(out["extra_col"]) == ["keep me", "keep me too"]


class TestGetCurrentQbForm:
    def test_returns_mean_of_most_recent_five_games(self, synthetic_qb_csv):
        result = qf.get_current_qb_form("Kansas City Chiefs")
        # All 4 real games -> mean of [0.10, 0.20, 0.30, -0.10] = 0.125
        assert result["qb_epa_trail"] == pytest.approx(0.125)

    def test_continuity_reflects_the_real_last_two_completed_games(self, synthetic_qb_csv):
        # Real last two games: Q1 (game 3), Q2 (game 4) -- different
        # starters -- continuity going into a hypothetical NEXT game is 0.
        result = qf.get_current_qb_form("Kansas City Chiefs")
        assert result["qb_continuity_flag"] == 0.0

    def test_unknown_franchise_gets_neutral_fallback_not_a_crash(self, synthetic_qb_csv):
        result = qf.get_current_qb_form("Some Team Not In The Data")
        league_avg = qf.load_qb_starts()["qb_epa_per_dropback"].mean()
        assert result["qb_epa_trail"] == pytest.approx(league_avg)
        assert result["qb_continuity_flag"] == 0.5
