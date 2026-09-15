"""
Unit tests for sports/nba/research_ft_rate.py's `attach_ft_rate_features()` --
walk-forward-safe (shift(1)-before-rolling) trailing FT rate (offense,
FTA/FGA) and opponent FT rate allowed (defense) columns.

Two teams (ATL, BOS) playing each other 3 times, with HAND-DERIVED expected
values -- see the inline comments for the full arithmetic. Mirrors the style
of tests/test_nflfastr_epa_sos_adjusted.py (synthetic data, a manually
computed expected trailing value asserted directly, not just "it ran").

Synthetic per-team-game box-score totals (FGA/FTA only matter for FT rate;
FGM/3PM/TO/OREB/DREB are filled with harmless placeholder values since
research_four_factors.py's `_load_box_team_games` parses/aggregates them too):

  game e0 (2020-01-01): ATL home  FGA=80,  FTA=20  (ft_rate=0.25)
                        BOS away  FGA=85,  FTA=17  (ft_rate=0.2)
  game e1 (2020-01-03): BOS home  FGA=80,  FTA=8   (ft_rate=0.1)
                        ATL away  FGA=90,  FTA=27  (ft_rate=0.3)
  game e2 (2020-01-05): ATL home  FGA=100, FTA=10  (ft_rate=0.1)
                        BOS away  FGA=100, FTA=30  (ft_rate=0.3)

Hand-derived values entering game e2 (both ATL's and BOS's trailing windows
strictly exclude e2 itself -- shift(1)-before-rolling):

  ATL's own ft_rate_l10   = mean(e0 ATL=0.25, e1 ATL=0.30) = 0.275
  ATL's opp_ft_rate_l10   = mean(e0 opp(BOS)=0.20, e1 opp(BOS)=0.10) = 0.15
  BOS's own ft_rate_l10   = mean(e0 BOS=0.20, e1 BOS=0.10) = 0.15
  BOS's opp_ft_rate_l10   = mean(e0 opp(ATL)=0.25, e1 opp(ATL)=0.30) = 0.275

  ft_rate_diff (vg2, ATL home - BOS away)            = 0.275 - 0.15  = +0.125
  opp_ft_rate_allowed_diff (vg2, ATL home - BOS away) = 0.15 - 0.275 = -0.125
"""

import pandas as pd
import pytest

from sports.nba import research_four_factors as rff
from sports.nba import research_ft_rate as frr


def _fga_fta_string(fga: int, fta: int) -> tuple:
    """Builds ('M-A', 'M-A') strings for the fgm_a/ftm_a columns
    `_load_box_team_games` parses via `_parse_made_attempted` -- makes are
    irrelevant to FT rate (FTA/FGA), so 0 makes is used throughout, keeping
    the attempted counts exactly as designed above."""
    return f"0-{fga}", f"0-{fta}"


@pytest.fixture
def synthetic_box_csv(tmp_path, monkeypatch):
    rows = []
    specs = [
        ("e0", "2020-01-01", "Atlanta Hawks", "home", 80, 20),
        ("e0", "2020-01-01", "Boston Celtics", "away", 85, 17),
        ("e1", "2020-01-03", "Boston Celtics", "home", 80, 8),
        ("e1", "2020-01-03", "Atlanta Hawks", "away", 90, 27),
        ("e2", "2020-01-05", "Atlanta Hawks", "home", 100, 10),
        ("e2", "2020-01-05", "Boston Celtics", "away", 100, 30),
    ]
    for event_id, date, team, home_away, fga, fta in specs:
        fgm_a, ftm_a = _fga_fta_string(fga, fta)
        rows.append({
            "date": date, "event_id": event_id, "team": team, "home_away": home_away,
            "to": 10, "oreb": 10, "dreb": 30,
            "fgm_a": fgm_a, "threepm_a": "0-0", "ftm_a": ftm_a,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "box.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(rff, "BOX_SCORES_PATH", path)
    yield path


@pytest.fixture
def synthetic_games():
    return pd.DataFrame([
        {"game_id": "vg0", "home_franchise": "ATL", "away_franchise": "BOS",
         "date": pd.Timestamp("2020-01-01"), "season": 2020},
        {"game_id": "vg1", "home_franchise": "BOS", "away_franchise": "ATL",
         "date": pd.Timestamp("2020-01-03"), "season": 2020},
        {"game_id": "vg2", "home_franchise": "ATL", "away_franchise": "BOS",
         "date": pd.Timestamp("2020-01-05"), "season": 2020},
    ])


class TestAttachFtRateFeatures:
    def test_hand_derived_trailing_values_match_exactly(self, synthetic_box_csv, synthetic_games):
        out = frr.attach_ft_rate_features(synthetic_games)
        row = out[out["game_id"] == "vg2"].iloc[0]

        assert row["home_ft_rate_l10"] == pytest.approx(0.275, abs=1e-9)
        assert row["home_opp_ft_rate_allowed_l10"] == pytest.approx(0.15, abs=1e-9)
        assert row["away_ft_rate_l10"] == pytest.approx(0.15, abs=1e-9)
        assert row["away_opp_ft_rate_allowed_l10"] == pytest.approx(0.275, abs=1e-9)

    def test_diff_columns_are_home_minus_away(self, synthetic_box_csv, synthetic_games):
        out = frr.attach_ft_rate_features(synthetic_games)
        row = out[out["game_id"] == "vg2"].iloc[0]

        assert row["ft_rate_diff"] == pytest.approx(0.125, abs=1e-9)
        assert row["opp_ft_rate_allowed_diff"] == pytest.approx(-0.125, abs=1e-9)

    def test_first_game_falls_back_to_league_average_not_zero_or_nan(self, synthetic_box_csv, synthetic_games):
        out = frr.attach_ft_rate_features(synthetic_games)
        row = out[out["game_id"] == "vg0"].iloc[0]

        # vg0 is both ATL's and BOS's first tracked game -- no trailing
        # history exists yet, so this must be the computed league-average
        # fallback (a real, non-zero, non-NaN value derived from the same
        # synthetic data: mean of all 6 team-game ft_rates), never 0.0/NaN.
        league_avg = pd.Series([0.25, 0.20, 0.10, 0.30, 0.10, 0.30]).mean()
        assert row["home_ft_rate_l10"] == pytest.approx(league_avg, abs=1e-9)
        assert row["away_ft_rate_l10"] == pytest.approx(league_avg, abs=1e-9)
        assert not pd.isna(row["home_ft_rate_l10"])

    def test_never_drops_or_reorders_existing_rows(self, synthetic_box_csv, synthetic_games):
        games = synthetic_games.copy()
        games["extra_col"] = ["keep me", "keep me too", "and me"]
        out = frr.attach_ft_rate_features(games)
        assert len(out) == 3
        assert list(out["game_id"]) == ["vg0", "vg1", "vg2"]
        assert list(out["extra_col"]) == ["keep me", "keep me too", "and me"]

    def test_adds_exactly_the_expected_six_new_columns(self, synthetic_box_csv, synthetic_games):
        out = frr.attach_ft_rate_features(synthetic_games)
        for col in frr.FEATURE_COLS:
            assert col in out.columns
        assert len(frr.FEATURE_COLS) == 6

    def test_unmatched_game_gets_league_average_not_a_crash(self, synthetic_box_csv):
        # A team pair with no corresponding box-score rows at all (outside
        # the +/-1 day join tolerance / simply not present) must fall back
        # to the league average, not crash or silently propagate NaN.
        games = pd.DataFrame([{
            "game_id": "vg_unmatched", "home_franchise": "ATL", "away_franchise": "BOS",
            "date": pd.Timestamp("2019-01-01"), "season": 2019,
        }])
        out = frr.attach_ft_rate_features(games)
        row = out.iloc[0]
        assert not pd.isna(row["home_ft_rate_l10"])
        assert not pd.isna(row["home_opp_ft_rate_allowed_l10"])
