"""
Unit tests for two real fixes shipped this session in core/edge_finder.py:
MAX_MONEYLINE_UNDERDOG_ODDS (the CFB extreme-underdog exclusion) and
reconcile_unvalidated_confidence() (the self-healing sweep for picks
logged before a gating rule existed). Both were validated against the real
CFB dataset at the time, but neither had a fast, deterministic regression
test -- this is what stops a future refactor from silently reintroducing
the exact staleness bug the reconcile function exists to prevent.
"""

import sqlite3

import pytest

from core import edge_finder, ensemble


def make_stds():
    return ensemble.ResidualStds(
        elo_margin_std=13.5, ml_margin_std=13.5, ml_total_std=13.7, naive_total_std=13.8,
    )


def moneyline_row(home_odds, away_odds, rating_diff_pre=0.0):
    """Minimal row: only moneyline columns populated, so evaluate_game's
    spread/total blocks harmlessly no-op (their own odds columns are
    missing) -- keeps these tests focused on the moneyline exclusion."""
    return {
        "rating_diff_pre": rating_diff_pre,
        "predicted_margin": 0.0,
        "predicted_total": 48.0,
        "naive_total": 48.0,
        "Home Odds Close": home_odds,
        "Away Odds Close": away_odds,
    }


class TestMaxMoneylineUnderdogOdds:
    def test_cfb_extreme_underdog_side_excluded(self):
        # A real shape: home is a big favorite (1.10), away is a real
        # extreme underdog (odds > the 10.0 CFB cap).
        row = moneyline_row(home_odds=1.10, away_odds=15.0)
        opps = edge_finder.evaluate_game(
            row, make_stds(), elo_points_per_margin=14.0, cfg=ensemble.EnsembleConfig(), sport="CFB",
        )
        sides = {(o.market, o.side) for o in opps}
        assert ("moneyline", "home") in sides   # favorite side still offered
        assert ("moneyline", "away") not in sides   # extreme underdog excluded

    def test_cfb_moderate_underdog_not_excluded(self):
        # Same shape but under the cap -- must still be offered.
        row = moneyline_row(home_odds=1.50, away_odds=8.0)
        opps = edge_finder.evaluate_game(
            row, make_stds(), elo_points_per_margin=14.0, cfg=ensemble.EnsembleConfig(), sport="CFB",
        )
        sides = {(o.market, o.side) for o in opps}
        assert ("moneyline", "away") in sides

    def test_cap_is_cfb_specific_not_global(self):
        # The exact same extreme-underdog price, for a sport NOT in
        # MAX_MONEYLINE_UNDERDOG_ODDS -- must NOT be excluded. Confirms the
        # cap doesn't leak into other sports.
        row = moneyline_row(home_odds=1.10, away_odds=15.0)
        opps = edge_finder.evaluate_game(
            row, make_stds(), elo_points_per_margin=14.0, cfg=ensemble.EnsembleConfig(), sport="NFL",
        )
        sides = {(o.market, o.side) for o in opps}
        assert ("moneyline", "away") in sides

    def test_no_cap_when_sport_is_none(self):
        # Confidence-blind callers (research scripts) pass sport=None --
        # must behave like an uncapped sport, not silently apply CFB's cap.
        row = moneyline_row(home_odds=1.10, away_odds=15.0)
        opps = edge_finder.evaluate_game(
            row, make_stds(), elo_points_per_margin=14.0, cfg=ensemble.EnsembleConfig(), sport=None,
        )
        sides = {(o.market, o.side) for o in opps}
        assert ("moneyline", "away") in sides


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("""
        CREATE TABLE forward_picks (
            id INTEGER PRIMARY KEY, sport TEXT, market TEXT,
            settled INTEGER, confidence TEXT, market_odds REAL
        )
    """)
    yield c
    c.close()


def insert_pick(conn, id, sport, market, settled, confidence, market_odds=1.9):
    conn.execute(
        "INSERT INTO forward_picks (id, sport, market, settled, confidence, market_odds) VALUES (?,?,?,?,?,?)",
        (id, sport, market, settled, confidence, market_odds),
    )
    conn.commit()


def confidence_of(conn, id):
    return conn.execute("SELECT confidence FROM forward_picks WHERE id=?", (id,)).fetchone()[0]


class TestReconcileUnvalidatedConfidence:
    def test_relabels_stale_unvalidated_sport_market(self, conn):
        # CFB is in SPREAD_UNVALIDATED_SPORTS -- a pending pick logged
        # before that gate shipped should get caught.
        insert_pick(conn, 1, "CFB", "spread", settled=0, confidence="High")
        changed = edge_finder.reconcile_unvalidated_confidence(conn)
        assert changed == 1
        assert confidence_of(conn, 1) == "Unvalidated"

    def test_leaves_already_unvalidated_alone_and_reports_zero_changed(self, conn):
        insert_pick(conn, 1, "CFB", "spread", settled=0, confidence="Unvalidated")
        changed = edge_finder.reconcile_unvalidated_confidence(conn)
        assert changed == 0
        assert confidence_of(conn, 1) == "Unvalidated"

    def test_never_touches_a_settled_row(self, conn):
        # Real historical record -- must never be silently rewritten,
        # even if it would violate a rule added after the fact.
        insert_pick(conn, 1, "CFB", "spread", settled=1, confidence="High")
        changed = edge_finder.reconcile_unvalidated_confidence(conn)
        assert changed == 0
        assert confidence_of(conn, 1) == "High"

    def test_never_touches_a_sport_market_not_in_either_gated_set(self, conn):
        insert_pick(conn, 1, "NHL", "moneyline", settled=0, confidence="High")
        changed = edge_finder.reconcile_unvalidated_confidence(conn)
        assert changed == 0
        assert confidence_of(conn, 1) == "High"

    def test_catches_extreme_underdog_moneyline_regardless_of_sport_set_membership(self, conn):
        # CFB moneyline itself is NOT in MONEYLINE_UNVALIDATED_SPORTS -- this
        # row is only caught via MAX_MONEYLINE_UNDERDOG_ODDS's own rule.
        insert_pick(conn, 1, "CFB", "moneyline", settled=0, confidence="Medium", market_odds=15.0)
        changed = edge_finder.reconcile_unvalidated_confidence(conn)
        assert changed == 1
        assert confidence_of(conn, 1) == "Unvalidated"

    def test_moneyline_under_the_odds_cap_is_left_alone(self, conn):
        insert_pick(conn, 1, "CFB", "moneyline", settled=0, confidence="Medium", market_odds=5.0)
        changed = edge_finder.reconcile_unvalidated_confidence(conn)
        assert changed == 0
        assert confidence_of(conn, 1) == "Medium"

    def test_idempotent_second_call_changes_nothing(self, conn):
        insert_pick(conn, 1, "CFB", "spread", settled=0, confidence="High")
        insert_pick(conn, 2, "CFB", "moneyline", settled=0, confidence="Medium", market_odds=20.0)
        first = edge_finder.reconcile_unvalidated_confidence(conn)
        second = edge_finder.reconcile_unvalidated_confidence(conn)
        assert first == 2
        assert second == 0
