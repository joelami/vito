"""
Unit test for core/matchup.py's build_matchup_feature_row() -- a real gap
found and fixed 2026-11-07 (app owner: "take a look at this and make any
necessary changes to make the algo better"): NBA's home_b2b/away_b2b
(ML_FEATURE_COLS, derived at training time in sports/nba/features.py as
home_rest_days<=1) had NO live source at all -- confirmed directly via a
real harness.run("NBA") call logging the "no live source supplies it"
warning for both columns, meaning every live NBA pick defaulted to 0
(never back-to-back) regardless of the real schedule. Fixed generically
in the shared row dict (home_rest/away_rest are already computed there
for every sport) rather than in NBA's own module, since the derivation
only needs values every sport already has.

Uses NBA's real config/features modules (no mocking needed for those --
real, already-loaded production modules) with a minimal fake pipeline
dict, matching this project's existing lightweight-fixture style for
dispatch-based functions.
"""

import pandas as pd
import pytest

from core.matchup import build_matchup_feature_row


def _fake_pipeline(home_last_game_date, away_last_game_date):
    form = pd.DataFrame(
        [
            {"last_game_date": home_last_game_date, "pf_l10": 110.0, "pa_l10": 108.0,
             "ats_pct_l10": 0.5, "win_pct_l10": 0.5, "streak": 0},
            {"last_game_date": away_last_game_date, "pf_l10": 112.0, "pa_l10": 109.0,
             "ats_pct_l10": 0.5, "win_pct_l10": 0.5, "streak": 0},
        ],
        index=["BOS", "MIA"],  # NBA's real canonical franchise identity is the 3-letter code, not the display name
    )
    return {"current_ratings": {}, "current_form": form}


class TestHomeAwayB2bLiveDerivation:
    def test_a_real_back_to_back_is_correctly_flagged_live(self):
        # Home team's last game was literally yesterday -> rest_days=1 -> b2b=1.
        game_date = pd.Timestamp("2026-11-08", tz="UTC")
        pipeline = _fake_pipeline(
            home_last_game_date=pd.Timestamp("2026-11-07"),
            away_last_game_date=pd.Timestamp("2026-11-05"),
        )
        row, _, _, _ = build_matchup_feature_row("nba", pipeline, "Boston Celtics", "Miami Heat", game_date)
        assert row["home_b2b"].iloc[0] == 1
        assert row["away_b2b"].iloc[0] == 0

    def test_a_real_rested_team_is_correctly_flagged_not_b2b(self):
        game_date = pd.Timestamp("2026-11-08", tz="UTC")
        pipeline = _fake_pipeline(
            home_last_game_date=pd.Timestamp("2026-11-04"),  # 4 days rest
            away_last_game_date=pd.Timestamp("2026-11-03"),  # 5 days rest
        )
        row, _, _, _ = build_matchup_feature_row("nba", pipeline, "Boston Celtics", "Miami Heat", game_date)
        assert row["home_b2b"].iloc[0] == 0
        assert row["away_b2b"].iloc[0] == 0

    def test_matches_the_exact_training_time_derivation(self):
        # sports/nba/features.py computes this as (rest_days <= 1) at
        # training time -- the live path must use the identical threshold,
        # not an approximation, or live scoring would diverge from what
        # the model was actually trained to expect.
        from sports.nba.features import ML_FEATURE_COLS
        assert "home_b2b" in ML_FEATURE_COLS and "away_b2b" in ML_FEATURE_COLS

        game_date = pd.Timestamp("2026-11-08", tz="UTC")
        pipeline = _fake_pipeline(
            home_last_game_date=pd.Timestamp("2026-11-07"),  # exactly 1 day rest -- the real boundary case
            away_last_game_date=pd.Timestamp("2026-11-06"),  # 2 days rest -- just past the boundary
        )
        row, _, _, _ = build_matchup_feature_row("nba", pipeline, "Boston Celtics", "Miami Heat", game_date)
        assert row["home_b2b"].iloc[0] == 1  # rest_days==1 -> b2b
        assert row["away_b2b"].iloc[0] == 0  # rest_days==2 -> not b2b
