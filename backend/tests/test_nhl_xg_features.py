"""
Unit tests for sports/nhl/moneypuck_xg_features.py -- the trailing
expected-goals (xG) features adopted 2026-09-15 (see decision_log.jsonl's
"nhl_moneypuck_trailing_xg_features" entry).

Two synthetic teams (BOS, team_id 1; TOR, team_id 21 -- real codes from
MONEYPUCK_CODE_TO_TEAM_ID, chosen so the module's own code->team_id map
doesn't need monkeypatching), playing each other 3 times, 2 skaters per
team per game, with hand-derived expected values -- see the inline module
comment below for the full arithmetic (same style as
tests/test_nflfastr_epa_sos_adjusted.py).

Raw synthetic on-ice xG (situation=="all") by (team, game):
  BOS g0 (2024-01-01): skater1 OnIce_F=1.0/OnIce_A=0.5, skater2 OnIce_F=2.0/OnIce_A=1.5
    -> team-game mean: xg_for=1.5, xg_against=1.0
  BOS g1 (2024-01-08): skater1 2.0/1.0, skater2 3.0/0.5 -> xg_for=2.5, xg_against=0.75
  BOS g2 (2024-01-15): skater1 1.0/2.0, skater2 1.4/1.0 -> xg_for=1.2, xg_against=1.5

  TOR g0: skater3 0.8/1.2, skater4 1.2/1.8 -> xg_for=1.0, xg_against=1.5
  TOR g1: skater3 1.0/1.0, skater4 1.5/0.5 -> xg_for=1.25, xg_against=0.75
  TOR g2: skater3 0.5/0.5, skater4 0.7/0.9 -> xg_for=0.6, xg_against=0.7

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


def _skater_row(game_id, team, opp, home_or_away, date_str, player, onice_f, onice_a):
    return {
        "gameId": game_id, "season": 2024, "playerTeam": team, "opposingTeam": opp,
        "home_or_away": home_or_away, "gameDate": date_str, "position": "C", "situation": "all",
        "OnIce_F_xGoals": onice_f, "OnIce_A_xGoals": onice_a,
    }


@pytest.fixture
def synthetic_two_team_csv(tmp_path, monkeypatch):
    rows = [
        # game_id, home team, away team, date
        *[
            _skater_row("mp_g0", "BOS", "TOR", "HOME", "20240101", "b1", 1.0, 0.5),
            _skater_row("mp_g0", "BOS", "TOR", "HOME", "20240101", "b2", 2.0, 1.5),
            _skater_row("mp_g0", "TOR", "BOS", "AWAY", "20240101", "t1", 0.8, 1.2),
            _skater_row("mp_g0", "TOR", "BOS", "AWAY", "20240101", "t2", 1.2, 1.8),

            _skater_row("mp_g1", "BOS", "TOR", "HOME", "20240108", "b1", 2.0, 1.0),
            _skater_row("mp_g1", "BOS", "TOR", "HOME", "20240108", "b2", 3.0, 0.5),
            _skater_row("mp_g1", "TOR", "BOS", "AWAY", "20240108", "t1", 1.0, 1.0),
            _skater_row("mp_g1", "TOR", "BOS", "AWAY", "20240108", "t2", 1.5, 0.5),

            _skater_row("mp_g2", "BOS", "TOR", "HOME", "20240115", "b1", 1.0, 2.0),
            _skater_row("mp_g2", "BOS", "TOR", "HOME", "20240115", "b2", 1.4, 1.0),
            _skater_row("mp_g2", "TOR", "BOS", "AWAY", "20240115", "t1", 0.5, 0.5),
            _skater_row("mp_g2", "TOR", "BOS", "AWAY", "20240115", "t2", 0.7, 0.9),
        ],
    ]
    df = pd.DataFrame(rows)
    old_path = tmp_path / "old.csv"
    new_path = tmp_path / "new.csv"
    df.to_csv(old_path, index=False)
    df.iloc[0:0].to_csv(new_path, index=False)  # empty-but-valid, same shape as production's two-file split

    monkeypatch.setattr(mf, "MP_FILE_OLD", old_path)
    monkeypatch.setattr(mf, "MP_FILE_NEW", new_path)
    mf._team_game_xg_cache = None
    yield
    mf._team_game_xg_cache = None


@pytest.fixture
def our_games():
    return pd.DataFrame([
        {"game_id": "our_g0", "home_team_id": 1, "away_team_id": 21, "date": pd.Timestamp("2024-01-01")},
        {"game_id": "our_g1", "home_team_id": 1, "away_team_id": 21, "date": pd.Timestamp("2024-01-08")},
        {"game_id": "our_g2", "home_team_id": 1, "away_team_id": 21, "date": pd.Timestamp("2024-01-15")},
    ])


class TestLoadTeamGameXg:
    def test_aggregates_as_mean_not_sum_across_dressed_skaters(self, synthetic_two_team_csv, our_games):
        team_game = mf.load_team_game_xg(our_games)
        bos_g0 = team_game[(team_game["game_id"] == "our_g0") & (team_game["team_id"] == 1)].iloc[0]
        assert bos_g0["xg_for"] == pytest.approx(1.5, abs=1e-9)
        assert bos_g0["xg_against"] == pytest.approx(1.0, abs=1e-9)

    def test_joins_to_our_own_game_id_via_team_and_date(self, synthetic_two_team_csv, our_games):
        team_game = mf.load_team_game_xg(our_games)
        assert set(team_game["game_id"]) == {"our_g0", "our_g1", "our_g2"}
        assert "mp_g0" not in set(team_game["game_id"])  # never leaks MoneyPuck's own gameId through


class TestBuildTrailingXgFeatures:
    def test_hand_derived_trailing_values_match_exactly(self, synthetic_two_team_csv, our_games):
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

    def test_cold_start_falls_back_to_league_average(self, synthetic_two_team_csv, our_games):
        out = mf.build_trailing_xg_features(our_games, n_games=10)
        g0 = out[out["game_id"] == "our_g0"].iloc[0]
        assert g0["home_xg_for_l10"] == pytest.approx(LEAGUE_AVG_XG_FOR, abs=1e-9)
        assert g0["home_xg_against_l10"] == pytest.approx(LEAGUE_AVG_XG_AGAINST, abs=1e-9)
        assert g0["away_xg_for_l10"] == pytest.approx(LEAGUE_AVG_XG_FOR, abs=1e-9)
        assert g0["away_xg_against_l10"] == pytest.approx(LEAGUE_AVG_XG_AGAINST, abs=1e-9)

    def test_never_drops_or_reorders_existing_rows(self, synthetic_two_team_csv, our_games):
        games = our_games.copy()
        games["extra_col"] = ["keep me", "keep me too", "and me"]
        out = mf.build_trailing_xg_features(games, n_games=10)
        assert len(out) == 3
        assert list(out["game_id"]) == ["our_g0", "our_g1", "our_g2"]
        assert list(out["extra_col"]) == ["keep me", "keep me too", "and me"]

    def test_adds_exactly_the_four_columns_ml_feature_cols_expects(self, synthetic_two_team_csv, our_games):
        from sports.nhl.features import ML_FEATURE_COLS
        out = mf.build_trailing_xg_features(our_games, n_games=10)
        xg_cols = [c for c in ML_FEATURE_COLS if c.endswith("xg_for_l10") or c.endswith("xg_against_l10")]
        assert len(xg_cols) == 4
        for col in xg_cols:
            assert col in out.columns
            assert out[col].notna().all()  # cold-start fallback means this should NEVER be NaN


class TestGetCurrentTrailingXg:
    def test_matches_mean_of_all_real_games_not_just_the_shifted_trailing_value(
        self, synthetic_two_team_csv, our_games, monkeypatch
    ):
        # "Current form" = mean of the last n games INCLUDING the most
        # recent one (unlike training's shift(1), which always excludes
        # the game being predicted) -- same relationship
        # get_current_trailing_epa() has to build_trailing_epa_features()
        # in sports/nfl/nflfastr_features.py.
        monkeypatch.setattr("sports.nhl.loader.load_games", lambda: our_games)
        result = mf.get_current_trailing_xg(1, n_games=10)  # team_id 1 = BOS
        assert result["xg_for_l10"] == pytest.approx((1.5 + 2.5 + 1.2) / 3, abs=1e-9)
        assert result["xg_against_l10"] == pytest.approx((1.0 + 0.75 + 1.5) / 3, abs=1e-9)

    def test_respects_n_games_window(self, synthetic_two_team_csv, our_games, monkeypatch):
        monkeypatch.setattr("sports.nhl.loader.load_games", lambda: our_games)
        result = mf.get_current_trailing_xg(1, n_games=2)  # last 2 of BOS's 3 games: g1, g2
        assert result["xg_for_l10"] == pytest.approx((2.5 + 1.2) / 2, abs=1e-9)
        assert result["xg_against_l10"] == pytest.approx((0.75 + 1.5) / 2, abs=1e-9)

    def test_unknown_team_id_gets_league_average_not_a_crash(self, synthetic_two_team_csv, our_games, monkeypatch):
        monkeypatch.setattr("sports.nhl.loader.load_games", lambda: our_games)
        result = mf.get_current_trailing_xg(999999999)
        assert result["xg_for_l10"] == pytest.approx(LEAGUE_AVG_XG_FOR, abs=1e-9)
        assert result["xg_against_l10"] == pytest.approx(LEAGUE_AVG_XG_AGAINST, abs=1e-9)

    def test_reset_caches_forces_a_fresh_read(self, synthetic_two_team_csv, our_games, monkeypatch):
        monkeypatch.setattr("sports.nhl.loader.load_games", lambda: our_games)
        mf.get_current_trailing_xg(1)
        assert mf._team_game_xg_cache is not None
        mf.reset_caches()
        assert mf._team_game_xg_cache is None


class TestExtraMatchupFeaturesWiring:
    def test_nhl_features_extra_matchup_features_returns_the_four_live_xg_columns(
        self, synthetic_two_team_csv, our_games, monkeypatch
    ):
        monkeypatch.setattr("sports.nhl.loader.load_games", lambda: our_games)
        from sports.nhl.features import extra_matchup_features
        result = extra_matchup_features(1, 21, pd.Timestamp("2024-02-01"), None, None)
        assert set(result.keys()) == {
            "home_xg_for_l10", "home_xg_against_l10", "away_xg_for_l10", "away_xg_against_l10",
        }
        for v in result.values():
            assert isinstance(v, float)
