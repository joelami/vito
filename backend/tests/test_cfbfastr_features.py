"""
Unit tests for sports/cfb/cfbfastr_features.py -- the real cfbfastR
trailing-success-rate feature module adopted 2026-09-14 (hypothesis test
"cfb_cfbfastr_trailing_success_rate_features", see decision_log.jsonl).
Mirrors tests/test_nflfastr_features.py's structure, plus dedicated
coverage for the one thing that's genuinely different here: the
school-name normalize/prefix-match mapping (NFL's is a flat, exact
32-code dict; this one is fuzzy and needs its own tests).

Uses a small synthetic team_game_success CSV (monkeypatched DATA_PATH)
rather than the real ~39k-row fetched dataset -- fast, deterministic, and
doesn't require scripts/fetch_cfbfastr_team_game_success.py to have been run.
"""

import pandas as pd
import pytest

from sports.cfb import cfbfastr_features as cf


@pytest.fixture
def synthetic_success_csv(tmp_path, monkeypatch):
    """"Ohio State" (cfbfastR's bare school name) plays 3 games with real,
    known success-rate values (0.40, 0.50, 0.60 off_success_rate in
    order) -- makes the trailing average trivially checkable by hand."""
    rows = []
    for i, off_sr in enumerate([0.40, 0.50, 0.60]):
        rows.append({
            "game_id": f"g{i}", "season": 2024, "team": "Ohio State", "is_home": True,
            "off_success_rate": off_sr, "off_plays": 65,
            "def_success_rate_allowed": 0.45, "def_plays": 65,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "success.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(cf, "DATA_PATH", path)
    cf._team_name_cache = None  # reset module-level caches between tests
    cf._success_by_franchise_cache = None
    yield path
    cf._team_name_cache = None
    cf._success_by_franchise_cache = None


class TestNormalizeAndMapping:
    def test_normalize_handles_st_state_and_apostrophes(self):
        assert cf._normalize("Ohio St.") == "ohio state"
        assert cf._normalize("Hawai'i") == "hawaii"

    def test_prefix_match_maps_franchise_with_mascot_to_bare_school_name(self, synthetic_success_csv):
        mapping = cf.build_team_name_mapping(["Ohio State Buckeyes"])
        assert mapping == {"Ohio State Buckeyes": "Ohio State"}

    def test_exact_match_also_works(self, synthetic_success_csv):
        mapping = cf.build_team_name_mapping(["Ohio State"])
        assert mapping == {"Ohio State": "Ohio State"}

    def test_unrelated_franchise_gets_no_mapping(self, synthetic_success_csv):
        mapping = cf.build_team_name_mapping(["Michigan Wolverines"])
        assert "Michigan Wolverines" not in mapping

    def test_missing_file_raises_a_clear_error_not_a_silent_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cf, "DATA_PATH", tmp_path / "does_not_exist.csv")
        with pytest.raises(FileNotFoundError):
            cf.build_team_name_mapping(["Ohio State Buckeyes"])


class TestBuildTrailingSuccessFeatures:
    def test_trailing_value_is_the_mean_of_STRICTLY_PRIOR_games_only(self, synthetic_success_csv):
        # Ohio State's 3rd game (off_success_rate=0.60): the trailing
        # feature for THAT game must be the mean of games 1-2 only
        # (0.40, 0.50 -> 0.45), never including 0.60 itself. `game_num`
        # is a real cumcount() over `games` itself (see this module's
        # docstring -- no date column in the raw cfbfastR release), so
        # all 3 of Ohio State's real games need to be present in the
        # test frame for the 3rd row to actually land on game_num=2,
        # exactly as it would in the real, full-history `games` this
        # function is actually called with in production.
        games = pd.DataFrame([
            {"game_id": "vg0", "season": 2024, "home_franchise": "Ohio State Buckeyes",
             "away_franchise": "Rutgers Scarlet Knights", "date": pd.Timestamp("2024-09-01")},
            {"game_id": "vg1", "season": 2024, "home_franchise": "Ohio State Buckeyes",
             "away_franchise": "Purdue Boilermakers", "date": pd.Timestamp("2024-11-23")},
            {"game_id": "vg2", "season": 2024, "home_franchise": "Ohio State Buckeyes",
             "away_franchise": "Michigan Wolverines", "date": pd.Timestamp("2024-11-30")},
        ])
        out = cf.build_trailing_success_features(games, n_games=10)
        assert out["home_off_success_rate_trail"].iloc[2] == pytest.approx(0.45)

    def test_first_game_ever_gets_league_average_fallback(self, synthetic_success_csv):
        games = pd.DataFrame([{
            "game_id": "vg0", "season": 2024, "home_franchise": "Ohio State Buckeyes",
            "away_franchise": "Michigan Wolverines", "date": pd.Timestamp("2024-09-01"),
        }])
        out = cf.build_trailing_success_features(games, n_games=10)
        success = cf.load_team_game_success(["Ohio State Buckeyes", "Michigan Wolverines"])
        league_avg = success["off_success_rate"].mean()
        assert out["home_off_success_rate_trail"].iloc[0] == pytest.approx(league_avg)

    def test_unmapped_team_gets_league_average_fallback_not_a_crash(self, synthetic_success_csv):
        # "Michigan Wolverines" has zero rows in this synthetic data at
        # all (real, honest ~74% coverage gap this module's docstring
        # documents) -- must fall back cleanly, not KeyError/NaN.
        games = pd.DataFrame([{
            "game_id": "vg0", "season": 2024, "home_franchise": "Ohio State Buckeyes",
            "away_franchise": "Michigan Wolverines", "date": pd.Timestamp("2024-09-01"),
        }])
        out = cf.build_trailing_success_features(games, n_games=10)
        success = cf.load_team_game_success(["Ohio State Buckeyes", "Michigan Wolverines"])
        league_avg = success["off_success_rate"].mean()
        assert out["away_off_success_rate_trail"].iloc[0] == pytest.approx(league_avg)

    def test_never_drops_or_reorders_existing_rows(self, synthetic_success_csv):
        games = pd.DataFrame([
            {"game_id": "vg0", "season": 2024, "home_franchise": "Ohio State Buckeyes",
             "away_franchise": "Michigan Wolverines", "date": pd.Timestamp("2024-09-01"), "extra_col": "keep me"},
            {"game_id": "vg1", "season": 2024, "home_franchise": "Michigan Wolverines",
             "away_franchise": "Ohio State Buckeyes", "date": pd.Timestamp("2024-09-08"), "extra_col": "keep me too"},
        ])
        out = cf.build_trailing_success_features(games, n_games=10)
        assert len(out) == 2
        assert list(out["game_id"]) == ["vg0", "vg1"]
        assert list(out["extra_col"]) == ["keep me", "keep me too"]

    def test_adds_exactly_the_four_columns_ml_feature_cols_expects(self, synthetic_success_csv):
        from sports.cfb.features import ML_FEATURE_COLS
        games = pd.DataFrame([{
            "game_id": "vg0", "season": 2024, "home_franchise": "Ohio State Buckeyes",
            "away_franchise": "Michigan Wolverines", "date": pd.Timestamp("2024-09-01"),
        }])
        out = cf.build_trailing_success_features(games, n_games=10)
        sr_cols_in_ml_features = [c for c in ML_FEATURE_COLS if c.endswith("_trail")]
        assert len(sr_cols_in_ml_features) == 4
        for col in sr_cols_in_ml_features:
            assert col in out.columns, f"{col} is in ML_FEATURE_COLS but build_trailing_success_features didn't produce it"


class TestGetCurrentTrailingSuccess:
    def test_raises_clear_error_before_any_pipeline_build_in_this_process(self, synthetic_success_csv):
        with pytest.raises(RuntimeError):
            cf.get_current_trailing_success("Ohio State Buckeyes")

    def test_returns_mean_of_most_recent_n_games(self, synthetic_success_csv):
        cf.load_team_game_success(["Ohio State Buckeyes"])  # populate the cache, as a real pipeline build would
        result = cf.get_current_trailing_success("Ohio State Buckeyes", n_games=2)
        # Most recent 2 of [0.40, 0.50, 0.60] -> [0.50, 0.60] -> mean 0.55
        assert result["off_success_rate"] == pytest.approx(0.55)

    def test_unknown_franchise_gets_league_average_not_a_crash(self, synthetic_success_csv):
        cf.load_team_game_success(["Ohio State Buckeyes"])
        result = cf.get_current_trailing_success("Some Team Not In The Data")
        league_avg = cf._success_by_franchise_cache["off_success_rate"].mean()
        assert result["off_success_rate"] == pytest.approx(league_avg)

    def test_returns_both_trailing_columns(self, synthetic_success_csv):
        cf.load_team_game_success(["Ohio State Buckeyes"])
        result = cf.get_current_trailing_success("Ohio State Buckeyes")
        assert set(result.keys()) == set(cf.TRAILING_COLS)
