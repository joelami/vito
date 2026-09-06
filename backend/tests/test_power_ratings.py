"""
Unit tests for core/power_ratings.py -- the Elo-style engine every sport's
walk-forward backtest and live pick generation is built on.

Two things get real, dedicated coverage here:
1. The walk-forward invariant the module's own docstring guarantees ("the
   rating used to *predict* a game reflects only games strictly before
   it"). This is THE property that makes every backtest in this project
   honest rather than leaking future results -- it has never had a
   regression test despite being load-bearing for every sport.
2. The low_history_start_rating/low_history_game_threshold mechanism added
   this session (CFB rating-floor fix) -- brand new, zero prior coverage,
   and easy to silently break since it's additive/off-by-default for every
   other sport.
"""

import pandas as pd
import pytest

from core.power_ratings import compute_power_ratings, PowerRatingConfig


def make_games(rows):
    """Each row: (date, home, away, home_score, away_score, season)."""
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "season"])


def run(games_df, config=None):
    return compute_power_ratings(
        games_df, home_col="home_team", away_col="away_team",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", config=config,
    )


class TestWalkForwardInvariant:
    def test_first_game_for_every_team_starts_at_start_rating(self):
        games = make_games([
            ("2024-01-01", "A", "B", 24, 10, 2024),
        ])
        result = run(games)
        row = result.history.iloc[0]
        assert row["home_rating_pre"] == pytest.approx(1500.0)
        assert row["away_rating_pre"] == pytest.approx(1500.0)

    def test_second_game_pre_rating_reflects_only_the_first_game(self):
        # Team A blows out B in game 1, then plays C in game 2 -- A's
        # pre-game rating for game 2 must already reflect the game 1 win,
        # and must NOT reflect anything from game 2 itself (it hasn't
        # happened yet when the "prediction" is made).
        games = make_games([
            ("2024-01-01", "A", "B", 40, 0, 2024),
            ("2024-01-08", "A", "C", 20, 20, 2024),
        ])
        result = run(games)
        game2 = result.history.iloc[1]
        # A won game 1 as a coinflip (both started at 1500) -- its rating
        # must have moved UP from 1500 by the time game 2 is scored.
        assert game2["home_rating_pre"] > 1500.0
        # C is appearing for the first time -- still exactly start_rating,
        # completely unaffected by A's game 1 result.
        assert game2["away_rating_pre"] == pytest.approx(1500.0)

    def test_order_of_input_rows_does_not_matter_only_date_does(self):
        # games must be processed in DATE order regardless of row order in
        # the input DataFrame -- a real bug class if sort_values were ever
        # removed or the sort key changed.
        forward = make_games([
            ("2024-01-01", "A", "B", 40, 0, 2024),
            ("2024-01-08", "A", "C", 10, 10, 2024),
        ])
        shuffled = make_games([
            ("2024-01-08", "A", "C", 10, 10, 2024),
            ("2024-01-01", "A", "B", 40, 0, 2024),
        ])
        r1 = run(forward).history.sort_index()
        r2 = run(shuffled).history.sort_index()
        pd.testing.assert_frame_equal(r1, r2)

    def test_zero_sum_rating_change_within_one_game(self):
        games = make_games([
            ("2024-01-01", "A", "B", 24, 10, 2024),
        ])
        result = run(games)
        home_delta = result.final_ratings["A"] - 1500.0
        away_delta = result.final_ratings["B"] - 1500.0
        assert home_delta == pytest.approx(-away_delta)

    def test_winner_rating_goes_up_loser_goes_down(self):
        games = make_games([
            ("2024-01-01", "A", "B", 24, 10, 2024),
        ])
        result = run(games)
        assert result.final_ratings["A"] > 1500.0
        assert result.final_ratings["B"] < 1500.0


class TestSeasonRegression:
    def test_regresses_toward_start_rating_at_season_boundary(self):
        cfg = PowerRatingConfig(season_regression=0.5)
        games = make_games([
            ("2024-01-01", "A", "B", 40, 0, 2024),   # A well above 1500 by end of 2024
            ("2025-01-01", "A", "C", 10, 10, 2025),  # A's first 2025 game -- regression applies here
        ])
        result = run(games, cfg)
        rating_after_2024 = result.history.iloc[0]  # not directly the final value, just sanity
        pre_2025 = result.history.iloc[1]["home_rating_pre"]
        end_of_2024_rating = result.final_ratings  # after game 2 this includes game 2's own update too
        # Reconstruct: A's rating right after game 1 (before any 2025 regression or game 2 update)
        rating_post_game1 = run(make_games([("2024-01-01", "A", "B", 40, 0, 2024)])).final_ratings["A"]
        expected_pre_2025 = rating_post_game1 + 0.5 * (1500.0 - rating_post_game1)
        assert pre_2025 == pytest.approx(expected_pre_2025)

    def test_no_regression_within_the_same_season(self):
        games = make_games([
            ("2024-01-01", "A", "B", 40, 0, 2024),
            ("2024-02-01", "A", "C", 10, 10, 2024),
        ])
        result = run(games)
        rating_post_game1 = run(make_games([("2024-01-01", "A", "B", 40, 0, 2024)])).final_ratings["A"]
        pre_game2 = result.history.iloc[1]["home_rating_pre"]
        assert pre_game2 == pytest.approx(rating_post_game1)  # unchanged -- same season, no regression

    def test_ratings_entering_season_regresses_teams_absent_this_season(self):
        games = make_games([("2024-01-01", "A", "B", 40, 0, 2024)])
        result = run(games, PowerRatingConfig(season_regression=0.5))
        snap_2025 = result.ratings_entering_season(2025)
        rating_2024 = result.final_ratings["A"]
        assert snap_2025["A"] == pytest.approx(rating_2024 + 0.5 * (1500.0 - rating_2024))

    def test_ratings_entering_season_leaves_current_season_teams_alone(self):
        games = make_games([("2024-01-01", "A", "B", 40, 0, 2024)])
        result = run(games)
        snap_2024 = result.ratings_entering_season(2024)
        assert snap_2024["A"] == pytest.approx(result.final_ratings["A"])


class TestLowHistoryStartRating:
    def test_disabled_by_default_matches_start_rating_exactly(self):
        # Default PowerRatingConfig() must be byte-identical to a run with
        # low_history explicitly off -- every existing sport (NFL/MLB/NBA/
        # NHL) depends on this being a true no-op.
        games = make_games([
            ("2024-01-01", "SmallSchool", "BigProgram", 3, 45, 2024),
        ])
        default_result = run(games, PowerRatingConfig())
        explicit_off_result = run(games, PowerRatingConfig(low_history_start_rating=None, low_history_game_threshold=0))
        pd.testing.assert_frame_equal(default_result.history, explicit_off_result.history)

    def test_new_team_seeded_at_the_low_history_rating_when_enabled(self):
        cfg = PowerRatingConfig(low_history_start_rating=1300.0, low_history_game_threshold=8)
        games = make_games([
            ("2024-01-01", "SmallSchool", "BigProgram", 3, 45, 2024),
        ])
        result = run(games, cfg)
        assert result.history.iloc[0]["home_rating_pre"] == pytest.approx(1300.0)
        assert result.history.iloc[0]["away_rating_pre"] == pytest.approx(1300.0)

    def test_games_played_counter_is_walk_forward_safe(self):
        # A team's Nth game's pre-game rating must depend only on its own
        # PRIOR appearance count -- not on how many games it plays in total
        # across the whole dataset (that would be hindsight/leakage).
        cfg = PowerRatingConfig(low_history_start_rating=1300.0, low_history_game_threshold=2)
        games = make_games([
            ("2024-01-01", "Team", "Opp1", 10, 10, 2024),   # games_played=0 pre -> low history
            ("2024-01-08", "Team", "Opp2", 10, 10, 2024),   # games_played=1 pre -> still low history
            ("2024-01-15", "Team", "Opp3", 10, 10, 2024),   # games_played=2 pre -> threshold cleared
        ])
        result = run(games, cfg)
        # Game 3's pre-rating must NOT be reseeded to 1300 -- by then the
        # team has 2 prior games and graduates out of low-history handling.
        # (It won't be exactly 1500 either since real Elo updates from
        # games 1-2 already moved it -- just confirm it's not pinned to 1300.)
        assert result.history.iloc[2]["home_rating_pre"] != pytest.approx(1300.0)

    def test_season_boundary_regresses_toward_low_floor_while_still_low_history(self):
        cfg = PowerRatingConfig(season_regression=0.5, low_history_start_rating=1300.0, low_history_game_threshold=8)
        games = make_games([
            ("2024-01-01", "SmallSchool", "BigProgram", 3, 45, 2024),
            ("2025-01-01", "SmallSchool", "OtherProgram", 3, 45, 2025),
        ])
        result = run(games, cfg)
        rating_post_game1 = run(
            make_games([("2024-01-01", "SmallSchool", "BigProgram", 3, 45, 2024)]), cfg
        ).final_ratings["SmallSchool"]
        pre_game2 = result.history.iloc[1]["home_rating_pre"]
        # Regresses toward 1300 (the low-history floor), NOT the full 1500
        # mean -- that's the whole point of the fix (see PowerRatingConfig's
        # docstring: a single season boundary shouldn't erase the seeding).
        expected = rating_post_game1 + 0.5 * (1300.0 - rating_post_game1)
        assert pre_game2 == pytest.approx(expected)
