"""
Unit tests for sports/nhl/moneypuck_xg_features.py -- the trailing
expected-goals (xG) features adopted 2026-09-15 (see decision_log.jsonl's
"nhl_moneypuck_trailing_xg_features" entry) and REWRITTEN 2026-10-15 to
read MoneyPuck's real TEAM-level per-game file instead of approximating
team xG as a per-skater mean (see moneypuck_xg_features.py's own top
docstring for the full incident -- the original skater files also had no
refresh mechanism at all, so live NHL scoring was silently using the
PREVIOUS season's trailing form the entire 2026-27 season).

Two synthetic teams (BOS, team_id 1; TOR, team_id 21 -- real codes from
MONEYPUCK_CODE_TO_TEAM_ID, chosen so the module's own code->team_id map
doesn't need monkeypatching), playing each other 3 times, with hand-
derived expected values -- same style as tests/test_nflfastr_epa_sos_
adjusted.py, simplified from the prior skater-level version since the
new data source is already team-level (no more per-skater mean needed).

Raw synthetic team-game xG (one row per team per game, already team-level):
  BOS g0 (2024-01-01): xg_for=1.5, xg_against=1.0
  BOS g1 (2024-01-08): xg_for=2.5, xg_against=0.75
  BOS g2 (2024-01-15): xg_for=1.2, xg_against=1.5

  TOR g0: xg_for=1.0, xg_against=1.5
  TOR g1: xg_for=1.25, xg_against=0.75
  TOR g2: xg_for=0.6, xg_against=0.7

League average (mean over all 6 team-game rows):
  xg_for:     (1.5+2.5+1.2+1.0+1.25+0.6)/6 = 8.05/6 = 1.3416666...
  xg_against: (1.0+0.75+1.5+1.5+0.75+0.7)/6 = 6.2/6 = 1.0333333...

Trailing (shift(1).rolling(10, min_periods=1).mean()) entering each game:
  BOS g0: no prior games -> fallback to league average (both cols)
  BOS g1: mean(g0) = xg_for=1.5, xg_against=1.0
  BOS g2: mean(g0,g1) = xg_for=(1.5+2.5)/2=2.0, xg_against=(1.0+0.75)/2=0.875
  TOR g0: no prior -> league average
  TOR g1: mean(g0) = xg_for=1.0, xg_against=1.5
  TOR g2: mean(g0,g1) = xg_for=(1.0+1.25)/2=1.125, xg_against=(1.5+0.75)/2=1.125
"""

import pandas as pd
import pytest

from sports.nhl import moneypuck_xg_features as mf

LEAGUE_AVG_XG_FOR = (1.5 + 2.5 + 1.2 + 1.0 + 1.25 + 0.6) / 6
LEAGUE_AVG_XG_AGAINST = (1.0 + 0.75 + 1.5 + 1.5 + 0.75 + 0.7) / 6


def _team_row(game_id, team, opp, home_or_away, date_str, xg_for, xg_against):
    return {
        "team": team, "season": 2024, "gameId": game_id, "playerTeam": team, "opposingTeam": opp,
        "home_or_away": home_or_away, "gameDate": date_str,
        "xg_for": xg_for, "xg_against": xg_against, "playoffGame": 0,
    }


@pytest.fixture
def synthetic_team_csv(tmp_path, monkeypatch):
    rows = [
        _team_row("mp_g0", "BOS", "TOR", "HOME", "20240101", 1.5, 1.0),
        _team_row("mp_g0", "TOR", "BOS", "AWAY", "20240101", 1.0, 1.5),
        _team_row("mp_g1", "BOS", "TOR", "HOME", "20240108", 2.5, 0.75),
        _team_row("mp_g1", "TOR", "BOS", "AWAY", "20240108", 1.25, 0.75),
        _team_row("mp_g2", "BOS", "TOR", "HOME", "20240115", 1.2, 1.5),
        _team_row("mp_g2", "TOR", "BOS", "AWAY", "20240115", 0.6, 0.7),
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "moneypuck_team_game_xg.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(mf, "DATA_PATH", path)
    mf._team_game_xg_cache = None
    yield path
    mf._team_game_xg_cache = None


@pytest.fixture
def our_games():
    """This project's own `games` shape (sports/nhl/loader.py's columns) --
    BOS (team_id 1) hosts TOR (team_id 21) all 3 times, same dates as the
    synthetic MoneyPuck rows, so the (home_team_id, away_team_id, date)
    join matches exactly (no merge_asof tolerance needed)."""
    return pd.DataFrame([
        {"game_id": "our_g0", "home_team_id": 1, "away_team_id": 21, "date": pd.Timestamp("2024-01-01", tz="UTC")},
        {"game_id": "our_g1", "home_team_id": 1, "away_team_id": 21, "date": pd.Timestamp("2024-01-08", tz="UTC")},
        {"game_id": "our_g2", "home_team_id": 1, "away_team_id": 21, "date": pd.Timestamp("2024-01-15", tz="UTC")},
    ])


class TestLoadTeamGameXg:
    def test_missing_file_raises_a_clear_error_not_a_silent_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mf, "DATA_PATH", tmp_path / "does_not_exist.csv")
        with pytest.raises(FileNotFoundError):
            mf.load_team_game_xg(pd.DataFrame())

    def test_real_team_codes_map_to_the_right_canonical_team_id(self, synthetic_team_csv, our_games):
        out = mf.load_team_game_xg(our_games)
        assert set(out["team_id"].unique()) == {1, 21}

    def test_joined_dtypes_match_our_own_int64_team_id_columns(self, synthetic_team_csv, our_games):
        # Real bug found and fixed 2026-10-15: .map() against a dict of
        # int values still upcasts to float64 the moment any row fails to
        # map, even before dropna() -- merge_asof's `by=` columns refuse a
        # float64-vs-int64 mismatch outright. This must not regress.
        out = mf.load_team_game_xg(our_games)
        assert out["team_id"].dtype == our_games["home_team_id"].dtype


class TestBuildTrailingXgFeatures:
    def test_trailing_value_is_the_mean_of_STRICTLY_PRIOR_games_only(self, synthetic_team_csv, our_games):
        out = mf.build_trailing_xg_features(our_games, n_games=10)

        g2 = out[out["game_id"] == "our_g2"].iloc[0]
        assert g2["home_xg_for_l10"] == pytest.approx(2.0, abs=1e-9)
        assert g2["home_xg_against_l10"] == pytest.approx(0.875, abs=1e-9)
        assert g2["away_xg_for_l10"] == pytest.approx(1.125, abs=1e-9)
        assert g2["away_xg_against_l10"] == pytest.approx(1.125, abs=1e-9)

        g1 = out[out["game_id"] == "our_g1"].iloc[0]
        assert g1["home_xg_for_l10"] == pytest.approx(1.5, abs=1e-9)
        assert g1["home_xg_against_l10"] == pytest.approx(1.0, abs=1e-9)
        assert g1["away_xg_for_l10"] == pytest.approx(1.0, abs=1e-9)
        assert g1["away_xg_against_l10"] == pytest.approx(1.5, abs=1e-9)

    def test_cold_start_falls_back_to_league_average(self, synthetic_team_csv, our_games):
        out = mf.build_trailing_xg_features(our_games, n_games=10)
        g0 = out[out["game_id"] == "our_g0"].iloc[0]
        assert g0["home_xg_for_l10"] == pytest.approx(LEAGUE_AVG_XG_FOR, abs=1e-9)
        assert g0["home_xg_against_l10"] == pytest.approx(LEAGUE_AVG_XG_AGAINST, abs=1e-9)
        assert g0["away_xg_for_l10"] == pytest.approx(LEAGUE_AVG_XG_FOR, abs=1e-9)
        assert g0["away_xg_against_l10"] == pytest.approx(LEAGUE_AVG_XG_AGAINST, abs=1e-9)

    def test_never_drops_or_reorders_existing_rows(self, synthetic_team_csv, our_games):
        games = our_games.copy()
        games["extra_col"] = ["keep me", "keep me too", "and me"]
        out = mf.build_trailing_xg_features(games, n_games=10)
        assert len(out) == 3
        assert list(out["game_id"]) == ["our_g0", "our_g1", "our_g2"]
        assert list(out["extra_col"]) == ["keep me", "keep me too", "and me"]

    def test_adds_exactly_the_four_columns_ml_feature_cols_expects(self, synthetic_team_csv, our_games):
        from sports.nhl.features import ML_FEATURE_COLS
        out = mf.build_trailing_xg_features(our_games, n_games=10)
        xg_cols = [c for c in ML_FEATURE_COLS if c.endswith("xg_for_l10") or c.endswith("xg_against_l10")]
        assert len(xg_cols) == 4
        for col in xg_cols:
            assert col in out.columns
            assert out[col].notna().all()  # cold-start fallback means this should NEVER be NaN


class TestGetCurrentTrailingXg:
    """Real bug found and fixed 2026-10-15 (discovered while verifying
    this rewrite): the ORIGINAL get_current_trailing_xg() called
    load_team_game_xg() with no `games` argument, which defaults to the
    STATIC BULK loader.load_games() -- a table with zero 2026-27 season
    rows. Since that function's gid_map join is an INNER join against
    that bulk table, every real current-season MoneyPuck row was
    silently dropped before ever reaching live scoring. The fix (mirrors
    get_current_trailing_epa()'s own real design): read the raw
    per-team rows directly, no join to our own games/game_id needed at
    all for "this team's own most recent real games." These tests use
    _load_raw_team_xg() directly (not load_team_game_xg()) to confirm
    that's really what's happening."""

    def test_matches_mean_of_all_real_games_not_just_the_shifted_trailing_value(self, synthetic_team_csv):
        # "Current form" = mean of the last n games INCLUDING the most
        # recent one (unlike training's shift(1), which always excludes
        # the game being predicted).
        result = mf.get_current_trailing_xg(1, n_games=10)  # team_id 1 = BOS
        assert result["xg_for_l10"] == pytest.approx((1.5 + 2.5 + 1.2) / 3, abs=1e-9)
        assert result["xg_against_l10"] == pytest.approx((1.0 + 0.75 + 1.5) / 3, abs=1e-9)

    def test_respects_n_games_window(self, synthetic_team_csv):
        result = mf.get_current_trailing_xg(1, n_games=2)  # last 2 of BOS's 3 games: g1, g2
        assert result["xg_for_l10"] == pytest.approx((2.5 + 1.2) / 2, abs=1e-9)
        assert result["xg_against_l10"] == pytest.approx((0.75 + 1.5) / 2, abs=1e-9)

    def test_unknown_team_id_gets_league_average_not_a_crash(self, synthetic_team_csv):
        result = mf.get_current_trailing_xg(999999999)
        assert result["xg_for_l10"] == pytest.approx(LEAGUE_AVG_XG_FOR, abs=1e-9)
        assert result["xg_against_l10"] == pytest.approx(LEAGUE_AVG_XG_AGAINST, abs=1e-9)

    def test_does_not_require_a_matching_game_id_in_our_own_games_table(self, synthetic_team_csv):
        # The real regression test for the bug above: deliberately do NOT
        # provide any `our_games`/loader monkeypatch at all. If this
        # function still secretly depended on joining against the bulk
        # loader, it would either crash (no games.csv fixture present) or
        # return the league-average fallback (0 real rows matched) instead
        # of BOS's own real games -- it must return the real per-team
        # values regardless.
        result = mf.get_current_trailing_xg(1, n_games=10)
        assert result["xg_for_l10"] == pytest.approx((1.5 + 2.5 + 1.2) / 3, abs=1e-9)

    def test_missing_dataset_returns_neutral_fallback_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mf, "DATA_PATH", tmp_path / "does_not_exist.csv")
        mf._team_game_xg_cache = None
        result = mf.get_current_trailing_xg(1)
        assert result == {f"{c}_l10": 0.0 for c in mf.TRAILING_COLS}
        mf._team_game_xg_cache = None

    def test_reset_caches_forces_a_fresh_read(self, synthetic_team_csv):
        mf.get_current_trailing_xg(1)
        assert mf._team_game_xg_cache is not None
        mf.reset_caches()
        assert mf._team_game_xg_cache is None


class TestExtraMatchupFeaturesWiring:
    def test_nhl_features_extra_matchup_features_returns_the_four_live_xg_columns(self, synthetic_team_csv):
        from sports.nhl.features import extra_matchup_features
        result = extra_matchup_features(1, 21, pd.Timestamp("2024-02-01"), None, None)
        assert set(result.keys()) == {
            "home_xg_for_l10", "home_xg_against_l10", "away_xg_for_l10", "away_xg_against_l10",
        }
        for v in result.values():
            assert isinstance(v, float)
