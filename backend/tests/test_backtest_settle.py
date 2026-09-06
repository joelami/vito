"""
Unit tests for core/backtest.py's settle_bet() -- shared, bit-for-bit,
between the historical backtest AND harness.py's real live settlement (its
own docstring's explicit guarantee). A sign error here would silently mark
real wins as losses (or vice versa) in both the backtest AND production --
exactly the class of bug this project has already found and fixed multiple
times in the confidence-tier layer; settle_bet itself has never had a
regression test guarding it.
"""

import pytest

from core.backtest import settle_bet


def row(**kwargs):
    """A dict is enough -- settle_bet only ever does row["home_win"] /
    row["actual_margin"] / row["actual_total"] style lookups."""
    return kwargs


class TestMoneyline:
    def test_home_win_home_side_wins(self):
        assert settle_bet("moneyline", "home", None, row(home_win=1)) == 1

    def test_home_win_away_side_loses(self):
        assert settle_bet("moneyline", "away", None, row(home_win=1)) == -1

    def test_away_win_away_side_wins(self):
        assert settle_bet("moneyline", "away", None, row(home_win=0)) == 1

    def test_away_win_home_side_loses(self):
        assert settle_bet("moneyline", "home", None, row(home_win=0)) == -1


class TestSpread:
    def test_home_favorite_covers(self):
        # home_line = -3.5 (home favored by 3.5), home wins by 10 -> home covers.
        assert settle_bet("spread", "home", -3.5, row(actual_margin=10)) == 1

    def test_home_favorite_fails_to_cover(self):
        # Home wins by only 1, needed to win by more than 3.5 -> loss.
        assert settle_bet("spread", "home", -3.5, row(actual_margin=1)) == -1

    def test_home_favorite_loses_outright(self):
        assert settle_bet("spread", "home", -3.5, row(actual_margin=-7)) == -1

    def test_home_push_exact_margin(self):
        # A real, common case -- home favored by exactly 3, wins by exactly 3.
        assert settle_bet("spread", "home", -3.0, row(actual_margin=3)) == 0

    def test_away_underdog_covers(self):
        # away's line is stored as -home_line (module convention) -- home
        # favored -3.5 means away's own line is +3.5. Home wins by only 1
        # (margin=1) -- away covered (lost by less than 3.5).
        assert settle_bet("spread", "away", 3.5, row(actual_margin=1)) == 1

    def test_away_underdog_fails_to_cover(self):
        # Home wins by 10, more than the 3.5 away was getting -> away loses.
        assert settle_bet("spread", "away", 3.5, row(actual_margin=10)) == -1

    def test_away_push(self):
        assert settle_bet("spread", "away", 3.0, row(actual_margin=3)) == 0

    def test_away_favorite(self):
        # away favored by 3.5 (home_line=+3.5, away's own line stored as -3.5),
        # away wins by 10 (actual_margin = home-away = -10) -> away covers.
        assert settle_bet("spread", "away", -3.5, row(actual_margin=-10)) == 1


class TestTotal:
    def test_over_wins_when_total_exceeds_line(self):
        assert settle_bet("total", "over", 48.5, row(actual_total=52)) == 1

    def test_over_loses_when_total_under_line(self):
        assert settle_bet("total", "over", 48.5, row(actual_total=40)) == -1

    def test_under_wins_when_total_below_line(self):
        assert settle_bet("total", "under", 48.5, row(actual_total=40)) == 1

    def test_under_loses_when_total_over_line(self):
        assert settle_bet("total", "under", 48.5, row(actual_total=52)) == -1

    def test_push_on_exact_total_over_side(self):
        assert settle_bet("total", "over", 48.0, row(actual_total=48)) == 0

    def test_push_on_exact_total_under_side(self):
        assert settle_bet("total", "under", 48.0, row(actual_total=48)) == 0


class TestUnknownMarket:
    def test_raises_value_error(self):
        with pytest.raises(ValueError):
            settle_bet("player_prop", "over", 10.5, row(actual_total=12))
