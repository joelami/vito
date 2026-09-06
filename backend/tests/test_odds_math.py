"""
Unit tests for core/odds_math.py -- the pure-math foundation every other
module (power ratings, ensemble, edge finder, backtest) builds on, per its
own module docstring: "keep them dependency-free and well-tested by hand,
since a bug here silently corrupts every downstream number." This session
found multiple real "backwards" bugs (confidence tiers, CLV sign) living
undetected for months because nothing regression-tested the primitives --
these tests exist to make that specific failure mode harder to repeat.
"""

import math

import pytest

from core import odds_math


# ---------- American <-> Decimal ----------

class TestAmericanToDecimal:
    def test_favorite(self):
        # -110 (bet $110 to win $100) -> decimal 1.909...
        assert odds_math.american_to_decimal(-110) == pytest.approx(1.9090909, abs=1e-6)

    def test_underdog(self):
        # +150 (bet $100 to win $150) -> decimal 2.50
        assert odds_math.american_to_decimal(150) == pytest.approx(2.50, abs=1e-9)

    def test_even_money_boundary(self):
        # +100 and -100 are both real, legal American prices for the same
        # true 50/50 probability -- must convert to the same decimal odds.
        assert odds_math.american_to_decimal(100) == pytest.approx(2.0)
        assert odds_math.american_to_decimal(-100) == pytest.approx(2.0)

    def test_round_trip_favorite(self):
        d = odds_math.american_to_decimal(-142)
        assert odds_math.decimal_to_american(d) == pytest.approx(-142, abs=1)

    def test_round_trip_underdog(self):
        d = odds_math.american_to_decimal(600)
        assert odds_math.decimal_to_american(d) == pytest.approx(600, abs=1)


class TestDecimalToAmerican:
    def test_invalid_odds_raises(self):
        with pytest.raises(ValueError):
            odds_math.decimal_to_american(1.0)
        with pytest.raises(ValueError):
            odds_math.decimal_to_american(0.5)


# ---------- Implied probability & vig removal ----------

class TestImpliedProb:
    def test_even_money(self):
        assert odds_math.implied_prob(2.0) == pytest.approx(0.5)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            odds_math.implied_prob(1.0)


class TestDevigTwoWay:
    def test_standard_vig_market_sums_to_one(self):
        # Two sides at standard -110/-110 (decimal 1.9091 each) -- real
        # vig market, ~4.5% overround before removal.
        fair_a, fair_b = odds_math.devig_two_way(1.9091, 1.9091)
        assert fair_a + fair_b == pytest.approx(1.0, abs=1e-6)
        assert fair_a == pytest.approx(0.5, abs=1e-3)

    def test_asymmetric_market(self):
        # A real favorite/underdog moneyline: -200/+170.
        dec_fav = odds_math.american_to_decimal(-200)   # 1.50
        dec_dog = odds_math.american_to_decimal(170)     # 2.70
        fair_fav, fair_dog = odds_math.devig_two_way(dec_fav, dec_dog)
        assert fair_fav + fair_dog == pytest.approx(1.0, abs=1e-9)
        assert fair_fav > fair_dog  # favorite's fair prob must stay higher after devig

    def test_missing_price_returns_none_none(self):
        assert odds_math.devig_two_way(None, 1.91) == (None, None)
        assert odds_math.devig_two_way(1.91, None) == (None, None)

    def test_invalid_price_returns_none_none(self):
        assert odds_math.devig_two_way(1.0, 1.91) == (None, None)
        assert odds_math.devig_two_way(0.5, 1.91) == (None, None)


class TestVigPct:
    def test_real_vig_is_positive_and_small(self):
        v = odds_math.vig_pct(1.9091, 1.9091)
        assert 0 < v < 10  # a real two-sided market's overround, not a raw 100%+ number

    def test_missing_price_returns_none(self):
        assert odds_math.vig_pct(None, 1.91) is None


# ---------- Expected value & Kelly ----------

class TestExpectedValue:
    def test_positive_edge_positive_ev(self):
        # Model says 60% win, market implies 50% (2.0 decimal) -- real edge, must be +EV.
        ev = odds_math.expected_value(0.60, 2.0)
        assert ev > 0

    def test_no_edge_is_zero_ev(self):
        # Model probability exactly equal to the break-even probability -> EV ~ 0.
        ev = odds_math.expected_value(0.5, 2.0)
        assert ev == pytest.approx(0.0, abs=1e-9)

    def test_negative_edge_negative_ev(self):
        ev = odds_math.expected_value(0.40, 2.0)
        assert ev < 0


class TestKellyFraction:
    def test_no_edge_returns_zero(self):
        # Never suggests betting against your own model -- the function's
        # own documented contract.
        assert odds_math.kelly_fraction(0.40, 2.0) == 0.0

    def test_real_edge_returns_positive_stake(self):
        k = odds_math.kelly_fraction(0.60, 2.0, fraction=0.25)
        assert k > 0

    def test_quarter_kelly_is_quarter_of_full_kelly(self):
        full = odds_math.kelly_fraction(0.60, 2.0, fraction=1.0)
        quarter = odds_math.kelly_fraction(0.60, 2.0, fraction=0.25)
        assert quarter == pytest.approx(full * 0.25)

    def test_zero_or_negative_net_odds_returns_zero(self):
        # decimal_odds <= 1.0 -> b <= 0 -- degenerate price, must not blow up.
        assert odds_math.kelly_fraction(0.9, 1.0) == 0.0


# ---------- Closing line value ----------

class TestClvPct:
    def test_beat_the_close_is_positive(self):
        # Took a worse-for-the-book price than where it closed (line moved
        # toward the bettor) -- real CLV, must be positive.
        clv = odds_math.clv_pct(price_taken=2.10, closing_price=1.91)
        assert clv > 0

    def test_worse_than_close_is_negative(self):
        clv = odds_math.clv_pct(price_taken=1.80, closing_price=1.91)
        assert clv < 0

    def test_same_price_is_zero(self):
        clv = odds_math.clv_pct(price_taken=1.91, closing_price=1.91)
        assert clv == pytest.approx(0.0, abs=1e-9)

    def test_missing_price_returns_none(self):
        assert odds_math.clv_pct(None, 1.91) is None
        assert odds_math.clv_pct(1.91, None) is None

    def test_invalid_price_returns_none(self):
        assert odds_math.clv_pct(1.0, 1.91) is None


# ---------- Probability from a point prediction ----------

class TestCoverProbSpread:
    def test_predicted_margin_equals_line_is_coinflip(self):
        # Predicted margin exactly matches the line -> the normal
        # distribution is centered exactly on the threshold -> ~50/50.
        p = odds_math.cover_prob_spread(predicted_margin=3.5, residual_std=13.5, line=-3.5)
        assert p == pytest.approx(0.5, abs=1e-6)

    def test_bigger_favorite_than_the_line_covers_more_than_half(self):
        # Model predicts a 10-point win against a -3.5 home line -> home
        # should be favored to cover (real, not backwards).
        p = odds_math.cover_prob_spread(predicted_margin=10.0, residual_std=13.5, line=-3.5)
        assert p > 0.5

    def test_smaller_favorite_than_the_line_covers_less_than_half(self):
        p = odds_math.cover_prob_spread(predicted_margin=1.0, residual_std=13.5, line=-3.5)
        assert p < 0.5

    def test_zero_std_is_deterministic(self):
        # A degenerate std shouldn't crash -- covers iff predicted beats the line outright.
        assert odds_math.cover_prob_spread(10.0, 0.0, -3.5) == 1.0
        assert odds_math.cover_prob_spread(1.0, 0.0, -3.5) == 0.0


class TestOverProbTotal:
    def test_predicted_equals_line_is_coinflip(self):
        p = odds_math.over_prob_total(predicted_total=48.5, residual_std=13.7, line=48.5)
        assert p == pytest.approx(0.5, abs=1e-6)

    def test_higher_predicted_total_favors_the_over(self):
        p = odds_math.over_prob_total(predicted_total=55.0, residual_std=13.7, line=48.5)
        assert p > 0.5

    def test_lower_predicted_total_favors_the_under(self):
        p = odds_math.over_prob_total(predicted_total=42.0, residual_std=13.7, line=48.5)
        assert p < 0.5
