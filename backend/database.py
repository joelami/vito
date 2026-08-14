"""
Plain sqlite3, no ORM, no migrations — same convention as the Hockey Scout
App and Photo Location Finder. Schema changes are additive edits to the
CREATE TABLE IF NOT EXISTS block below.

Holds only USER-GENERATED and APP-GENERATED persistent records (logged
bets, manually-entered upcoming games, a cached backtest run). The raw
historical odds/results data is never written here — it's loaded fresh
from the Datasets/*.xlsx files into pandas at startup every time, same
pattern as the Hockey app's CSV-to-DataFrame startup load.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.environ.get("DB_PATH", Path(__file__).parent / "sports_bet.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS manual_games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL DEFAULT 'NFL',
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_odds REAL,
    away_odds REAL,
    home_line REAL,
    home_line_odds REAL,
    away_line_odds REAL,
    total_line REAL,
    over_odds REAL,
    under_odds REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL DEFAULT 'NFL',
    game_label TEXT NOT NULL,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    line REAL,
    odds_taken REAL NOT NULL,
    stake REAL NOT NULL,
    placed_at TEXT NOT NULL,
    result TEXT NOT NULL DEFAULT 'pending',
    closing_odds REAL,
    clv_pct REAL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL DEFAULT 'NFL',
    run_at TEXT NOT NULL DEFAULT (datetime('now')),
    summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Real games/odds synced from ESPN's public scoreboard feed (core/espn_client.py).
-- "open" fields are locked in the first time we see an event (see the upsert in
-- harness.py) and never overwritten afterward, even as ESPN's own "close" fields
-- keep moving right up to kickoff — this is what lets forward_picks be priced at
-- a real, honest "opening" number, the same discipline the historical backtest
-- uses, rather than a price that only existed after the fact.
CREATE TABLE IF NOT EXISTS espn_games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL DEFAULT 'NFL',
    espn_event_id TEXT NOT NULL,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    completed INTEGER NOT NULL DEFAULT 0,
    is_neutral_venue INTEGER NOT NULL DEFAULT 0,
    is_playoff INTEGER NOT NULL DEFAULT 0,
    home_odds_open REAL, home_odds_close REAL,
    away_odds_open REAL, away_odds_close REAL,
    home_line_open REAL, home_line_close REAL,
    home_line_odds_open REAL, home_line_odds_close REAL,
    away_line_odds_open REAL, away_line_odds_close REAL,
    total_open REAL, total_close REAL,
    over_odds_open REAL, over_odds_close REAL,
    under_odds_open REAL, under_odds_close REAL,
    first_synced_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_synced_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(sport, espn_event_id)
);

-- Every qualifying edge the model found, auto-snapshotted at "Open" pricing the
-- moment a synced game first clears the same edge/confidence bar the backtest
-- uses — logged automatically, whether or not the user ever places the bet.
-- This is the forward-test track record: real future outcomes, not a backtest.
CREATE TABLE IF NOT EXISTS forward_picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL DEFAULT 'NFL',
    espn_event_id TEXT NOT NULL,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    line REAL,
    model_prob REAL NOT NULL,
    market_odds REAL NOT NULL,
    market_fair_prob REAL NOT NULL,
    edge_pct REAL NOT NULL,
    confidence TEXT NOT NULL,
    kelly_stake REAL NOT NULL,
    snapshotted_at TEXT NOT NULL DEFAULT (datetime('now')),
    settled INTEGER NOT NULL DEFAULT 0,
    result TEXT,
    profit_units REAL,
    clv_pct REAL,
    settled_at TEXT,
    UNIQUE(sport, espn_event_id, market, side, line)
);

-- The parlay counterpart to forward_picks: every combo the app actually
-- surfaced as a "Suggested Parlay" (same pool, same suggest_parlays() call,
-- same per-leg-count trim as /api/suggestions/daily), snapshotted once so it
-- can be graded later against the real outcome instead of only ever existing
-- as a live, disposable API response. Cross-league by nature (parlay legs
-- are pooled from every sport's pending picks at once, same as the
-- Suggestions page), so there's no `sport` column here the way forward_picks
-- has one — see harness.py's snapshot_new_parlays().
CREATE TABLE IF NOT EXISTS forward_parlays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    leg_count INTEGER NOT NULL,
    theme TEXT,
    leg_signature TEXT NOT NULL UNIQUE,  -- sorted forward_picks.id list, joined -- dedupes re-suggestion of an unchanged combo
    pick_ids TEXT NOT NULL,              -- JSON array of forward_picks.id, one per leg
    legs_json TEXT NOT NULL,             -- human-readable snapshot of each leg (sport/matchup/market/side/line/odds), so the UI never has to re-join forward_picks
    combined_decimal_odds REAL NOT NULL,
    combined_model_prob REAL,
    combined_market_fair_prob REAL,
    edge_pct REAL,
    kelly_stake REAL,
    snapshotted_at TEXT NOT NULL DEFAULT (datetime('now')),
    settled INTEGER NOT NULL DEFAULT 0,
    result TEXT,
    profit_units REAL,
    settled_at TEXT
);
"""


def _fix_forward_picks_null_line_dupes(conn):
    """
    Migration, safe to run on every startup: SQLite's plain `UNIQUE(...)`
    table constraint treats NULL as distinct from every other NULL (standard
    SQL semantics), so the `UNIQUE(sport, espn_event_id, market, side, line)`
    constraint on `forward_picks` never actually caught duplicates for
    moneyline picks — `line` is always NULL for moneyline, so two rows both
    having `line IS NULL` were never considered a collision, and
    `INSERT OR IGNORE` had nothing to ignore. Confirmed as a real, live bug
    (a genuine duplicate moneyline pick was found via the API, not
    hypothesized) rather than assumed from reading the schema.

    Fixed with a COALESCE-based unique index (line -> a sentinel that no
    real spread/total line would ever equal, so it can't collide with a
    genuine pick'em spread of 0) — SQLite's standard way to make NULL behave
    as "equal to itself" for uniqueness. The plain UNIQUE constraint above
    is left in place (harmless, still catches every non-moneyline case) —
    recreating the table just to drop it isn't worth the churn.

    Existing duplicate rows are deduped first (keeping the settled copy if
    one exists, else the earliest by id) since SQLite can't build a unique
    index over data that already violates it.
    """
    conn.execute("""
        DELETE FROM forward_picks
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY sport, espn_event_id, market, side, COALESCE(line, -999999)
                    ORDER BY settled DESC, id ASC
                ) AS rn
                FROM forward_picks
            ) WHERE rn = 1
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_forward_picks_unique
        ON forward_picks(sport, espn_event_id, market, side, COALESCE(line, -999999))
    """)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    _fix_forward_picks_null_line_dupes(conn)
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_setting(key: str, default=None):
    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
