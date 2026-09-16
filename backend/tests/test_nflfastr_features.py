"""
Unit tests for sports/nfl/nflfastr_features.py -- the real nflverse
trailing-EPA feature module adopted 2026-09-14. Two things get dedicated
coverage: the walk-forward safety of the trailing computation (the same
property core/power_ratings.py's own tests protect), and the team-code
mapping not silently going stale as nflverse's own data changes.

Uses a small synthetic team_game_epa CSV (monkeypatched DATA_PATH) rather
than the real ~11k-row fetched dataset -- fast, deterministic, and
doesn't require scripts/fetch_nflfastr_team_game_epa.py to have been run.
"""

import pandas as pd
import pytest

from sports.nfl import nflfastr_features as nf


@pytest.fixture
def synthetic_epa_csv(tmp_path, monkeypatch):
    """Team A plays 3 games with real, known EPA values (0.10, 0.20, 0.30
    off_epa_per_play in order) -- makes the trailing average trivially
    checkable by hand instead of just asserting "it ran."""
    rows = []
    for i, (date, off_epa) in enumerate([
        ("2024-09-08", 0.10), ("2024-09-15", 0.20), ("2024-09-22", 0.30),
    ]):
        rows.append({
            "game_id": f"g{i}", "season": 2024, "team": "KC", "is_home": True,
            "off_epa_per_play": off_epa, "off_success_rate": 0.40 + i * 0.01,
            "off_plays": 60, "def_epa_per_play_allowed": -0.05, "def_success_rate_allowed": 0.45,
            "def_plays": 60, "gameday": date,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "epa.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(nf, "DATA_PATH", path)
    nf._team_game_epa_cache = None  # reset the module-level cache between tests
    yield path
    nf._team_game_epa_cache = None


class TestLoadTeamGameEpa:
    def test_maps_team_code_to_real_franchise_name(self, synthetic_epa_csv):
        df = nf.load_team_game_epa()
        assert (df["franchise"] == "Kansas City Chiefs").all()

    def test_missing_file_raises_a_clear_error_not_a_silent_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nf, "DATA_PATH", tmp_path / "does_not_exist.csv")
        with pytest.raises(FileNotFoundError):
            nf.load_team_game_epa()

    def test_every_real_nflverse_code_this_session_pulled_has_a_mapping(self):
        # Real code list confirmed directly against the actual 2006-2025
        # pull (see scripts/fetch_nflfastr_team_game_epa.py's own module
        # docstring) -- catches TEAM_CODE_TO_FRANCHISE silently losing an
        # entry, not just "does the dict have 32 keys."
        real_codes = {"ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
                      "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
                      "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"}
        assert real_codes <= set(nf.TEAM_CODE_TO_FRANCHISE.keys())

    def test_franchise_names_are_unique(self):
        # A real bug this would catch: two different nflverse codes
        # accidentally mapped to the same franchise string would silently
        # merge two real teams' EPA histories together.
        names = list(nf.TEAM_CODE_TO_FRANCHISE.values())
        assert len(names) == len(set(names))


class TestBuildTrailingEpaFeatures:
    def test_trailing_value_is_the_mean_of_STRICTLY_PRIOR_games_only(self, synthetic_epa_csv):
        # Team A's 3rd game (off_epa_per_play=0.30): the trailing feature
        # for THAT game must be the mean of games 1-2 only (0.10, 0.20 ->
        # 0.15), never including 0.30 itself -- the exact walk-forward
        # property this whole feature category depends on.
        games = pd.DataFrame([{
            "game_id": "g2", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
            "date": pd.Timestamp("2024-09-22"),
        }])
        out = nf.build_trailing_epa_features(games, n_games=10)
        assert out["home_off_epa_per_play_trail"].iloc[0] == pytest.approx(0.15)

    def test_first_game_ever_gets_league_average_fallback(self, synthetic_epa_csv):
        # A team with zero prior games in the data -- must fall back to
        # the real league average, not crash or return NaN/0.0 unearned.
        games = pd.DataFrame([{
            "game_id": "g0", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
            "date": pd.Timestamp("2024-09-08"),
        }])
        out = nf.build_trailing_epa_features(games, n_games=10)
        league_avg = nf.load_team_game_epa()["off_epa_per_play"].mean()
        assert out["home_off_epa_per_play_trail"].iloc[0] == pytest.approx(league_avg)

    def test_never_drops_or_reorders_existing_rows(self, synthetic_epa_csv):
        games = pd.DataFrame([
            {"game_id": "g0", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
             "date": pd.Timestamp("2024-09-08"), "extra_col": "keep me"},
            {"game_id": "g1", "home_franchise": "Denver Broncos", "away_franchise": "Kansas City Chiefs",
             "date": pd.Timestamp("2024-09-15"), "extra_col": "keep me too"},
        ])
        out = nf.build_trailing_epa_features(games, n_games=10)
        assert len(out) == 2
        assert list(out["game_id"]) == ["g0", "g1"]
        assert list(out["extra_col"]) == ["keep me", "keep me too"]

    def test_adds_exactly_the_eight_columns_ml_feature_cols_expects(self, synthetic_epa_csv):
        from sports.nfl.features import ML_FEATURE_COLS
        games = pd.DataFrame([{
            "game_id": "g0", "home_franchise": "Kansas City Chiefs", "away_franchise": "Denver Broncos",
            "date": pd.Timestamp("2024-09-08"),
        }])
        out = nf.build_trailing_epa_features(games, n_games=10)
        epa_cols_in_ml_features = [c for c in ML_FEATURE_COLS if c.endswith("_trail")]
        assert len(epa_cols_in_ml_features) == 8
        for col in epa_cols_in_ml_features:
            assert col in out.columns, f"{col} is in ML_FEATURE_COLS but build_trailing_epa_features didn't produce it"


class TestGetCurrentTrailingEpa:
    def test_returns_mean_of_most_recent_n_games(self, synthetic_epa_csv):
        result = nf.get_current_trailing_epa("Kansas City Chiefs", n_games=2)
        # Most recent 2 of [0.10, 0.20, 0.30] -> [0.20, 0.30] -> mean 0.25
        assert result["off_epa_per_play"] == pytest.approx(0.25)

    def test_unknown_franchise_gets_league_average_not_a_crash(self, synthetic_epa_csv):
        result = nf.get_current_trailing_epa("Some Team Not In The Data")
        league_avg = nf.load_team_game_epa()["off_epa_per_play"].mean()
        assert result["off_epa_per_play"] == pytest.approx(league_avg)

    def test_returns_all_four_trailing_columns(self, synthetic_epa_csv):
        result = nf.get_current_trailing_epa("Kansas City Chiefs")
        assert set(result.keys()) == set(nf.TRAILING_COLS)

    def test_missing_dataset_returns_neutral_fallback_not_a_crash(self, tmp_path, monkeypatch):
        # Real incident, 2026-09-16: "/api/ratings?sport=NFL" was 404ing in
        # production because this dataset was never uploaded to R2 -- this
        # used to raise FileNotFoundError straight out of live matchup
        # scoring; now it degrades to a neutral 0.0 fallback instead.
        monkeypatch.setattr(nf, "DATA_PATH", tmp_path / "does_not_exist.csv")
        nf._team_game_epa_cache = None
        result = nf.get_current_trailing_epa("Kansas City Chiefs")
        assert result == {col: 0.0 for col in nf.TRAILING_COLS}
        nf._team_game_epa_cache = None
