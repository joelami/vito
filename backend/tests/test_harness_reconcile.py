"""
Regression test for a real, serious deployment bug found 2026-09-06: the
app owner spotted a stale, pre-fix CFB moneyline pick (Bowling Green
@ Nebraska, 18.00 odds -- well over the MAX_MONEYLINE_UNDERDOG_ODDS cap)
live in a production suggested parlay, days after that exclusion had
supposedly shipped and been verified.

Root cause: harness.reconcile_stale_confidence()'s self-heal logic
originally lived inline inside harness.py's `if __name__ == "__main__":`
block -- the standalone CLI path (`python3 harness.py`, used by local
launchd jobs). Every verification that session ran `python3 harness.py`
directly, which kept the LOCAL dev database self-healed and looked
completely fixed -- but Railway's actual scheduler (scheduler.py) only
ever `import harness` and calls specific functions; it never executes
that `__main__` block, so production's copy of the exact bug this
function exists to fix sat unreconciled the whole time regardless.

This test locks in the fix two ways: (1) the function itself works
correctly when called directly (already covered more thoroughly by
tests/test_edge_finder.py's TestReconcileUnvalidatedConfidence -- this
just confirms harness.reconcile_stale_confidence() is a real, callable
wrapper around it, not dead code); (2) a source-level check that BOTH
real call sites (scheduler.py's _run_full/_run_sync_only) actually call
it -- so a future refactor can't silently drop the scheduler.py wiring
again the same way and have every local test still pass.
"""

import re
from pathlib import Path

import pytest

import database
import harness


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    conn = __import__("sqlite3").connect(db_path)
    conn.execute("""
        CREATE TABLE forward_picks (
            id INTEGER PRIMARY KEY, sport TEXT, market TEXT,
            settled INTEGER, confidence TEXT, market_odds REAL
        )
    """)
    conn.commit()
    conn.close()
    return db_path


class TestReconcileStaleConfidenceIsWiredUp:
    def test_callable_directly_and_touches_the_real_database_module(self, temp_db):
        import sqlite3
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO forward_picks (id, sport, market, settled, confidence, market_odds) "
            "VALUES (1, 'CFB', 'moneyline', 0, 'High', 18.0)"
        )
        conn.commit()
        conn.close()

        changed = harness.reconcile_stale_confidence()
        assert changed == 1

        conn = sqlite3.connect(temp_db)
        confidence = conn.execute("SELECT confidence FROM forward_picks WHERE id=1").fetchone()[0]
        conn.close()
        assert confidence == "Unvalidated"

    def test_returns_zero_and_does_not_raise_on_an_empty_database(self, temp_db):
        assert harness.reconcile_stale_confidence() == 0

    def test_scheduler_run_full_calls_it_before_run_parlays(self):
        # Source-level check, deliberately not a mocked call-order test --
        # the real incident was this call being ABSENT from scheduler.py
        # entirely, which a mock-based test naturally can't catch (there'd
        # be nothing to assert against). Reading the real source is what
        # actually verifies the wiring exists.
        src = Path(__file__).parent.parent.joinpath("scheduler.py").read_text()
        full_run_body = src.split("def _run_full(")[1].split("\ndef ")[0]
        assert "reconcile_stale_confidence" in full_run_body, (
            "scheduler.py's _run_full() no longer calls harness.reconcile_stale_confidence() -- "
            "this is the exact real incident (2026-09-06) where the self-heal only ran locally, "
            "never in production, because Railway's scheduler doesn't go through harness.py's "
            "__main__ block. See this module's docstring."
        )
        # Must run BEFORE _run_parlays -- parlays are built from whatever
        # confidence values are on the pool at that moment.
        assert full_run_body.index("reconcile_stale_confidence") < full_run_body.index("_run_parlays"), (
            "reconcile_stale_confidence() must run before _run_parlays() in _run_full() -- "
            "otherwise a parlay can still be built from a not-yet-reconciled stale pick."
        )

    def test_scheduler_run_sync_only_calls_it_before_run_parlays(self):
        src = Path(__file__).parent.parent.joinpath("scheduler.py").read_text()
        sync_only_body = src.split("def _run_sync_only(")[1].split("\ndef ")[0]
        assert "reconcile_stale_confidence" in sync_only_body, (
            "scheduler.py's _run_sync_only() no longer calls harness.reconcile_stale_confidence() "
            "-- see this module's docstring for the real incident this guards against."
        )
        assert sync_only_body.index("reconcile_stale_confidence") < sync_only_body.index("_run_parlays")

    def test_harness_main_block_still_calls_it_too(self):
        # Local dev / launchd still goes through harness.py's own __main__
        # -- must not regress for that path either. Matches the real,
        # column-0 `if __name__ == "__main__":` statement specifically
        # (not this file's own docstring mentions of it, which also
        # contain that exact substring).
        src = Path(__file__).parent.parent.joinpath("harness.py").read_text()
        match = re.search(r'^if __name__ == "__main__":$', src, re.MULTILINE)
        assert match, "harness.py's __main__ guard not found in the expected form"
        main_block = src[match.end():]
        assert "reconcile_stale_confidence" in main_block
