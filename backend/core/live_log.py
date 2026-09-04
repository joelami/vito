"""
Real-time, per-row backstop for forward_picks -- closes the gap
core/db_backup.py's periodic whole-database snapshot leaves open. That
snapshot only runs on the 09:00 UTC full-run cycle (plus once on every
boot/redeploy) -- a Volume wipe any other time loses everything logged or
settled since the last snapshot, up to most of a day. That's exactly the
data this app can never regenerate (unlike Datasets/, which just re-syncs
from source): a real pick the model made, or a real settled result.

Real incident this responds to: after the second Volume-wipe recovery
still came back showing "a limited amount of data... just a few of
tonight's games," the app owner asked directly for a way to hold onto
game data better. A periodic full-DB snapshot alone isn't tight enough on
its own -- this adds a second, much finer-grained layer underneath it.

DESIGN: one small JSON object per pick in R2, keyed by the pick's own
UNIQUE constraint columns (sport, espn_event_id, market, side, line) --
deliberately NOT a true append-only log. R2's S3-compatible API has no
atomic append, so a real append-log would need read-modify-write on every
single pick (slower, and racy under concurrent writers). One object per
pick means a second write to the same pick (e.g. once it settles) just
overwrites the same key with a strictly more complete version -- nothing
to reconcile, no ordering logic needed anywhere. Every recovery operation
is a plain list -> download -> upsert.

Every call here is wrapped and non-fatal, matching db_backup.py's own
discipline -- a live_log write failing must NEVER block a real pick from
being logged into the actual database, which stays the single source of
truth for everything the app reads at request time. This is a disaster-
recovery backstop, not a second live datastore -- nothing ever reads from
R2 except restore_gap(), and only at boot.
"""

import json

from core.db_backup import _r2_client

LIVE_LOG_PREFIX = "live_log/forward_picks/"

# The exact columns forward_picks' schema defines, in the exact order
# restore_gap()'s INSERT below expects -- kept as one explicit list (not
# `dict(row).keys()`) so a future schema change that adds a column is a
# loud KeyError here, not a silent gap in what gets protected.
PICK_COLUMNS = [
    "sport", "espn_event_id", "date", "home_team", "away_team", "market", "side", "line",
    "model_prob", "market_odds", "market_fair_prob", "edge_pct", "confidence", "kelly_stake",
    "snapshotted_at", "settled", "result", "profit_units", "clv_pct", "settled_at",
]


def _pick_key(row) -> str:
    # Matches forward_picks' own UNIQUE(sport, espn_event_id, market, side, line)
    # exactly, so re-uploading the same pick later (e.g. once it settles)
    # naturally overwrites the same object instead of creating a duplicate.
    line = row["line"]
    line_str = "none" if line is None else f"{line:g}"
    return f"{LIVE_LOG_PREFIX}{row['sport']}_{row['espn_event_id']}_{row['market']}_{row['side']}_{line_str}.json"


def append_pick(row) -> bool:
    """
    Uploads the current full state of one forward_picks row to R2. `row`
    is anything dict-like (a sqlite3.Row from a fresh SELECT of the row
    that was just written is what both call sites in harness.py use --
    re-reading the real row rather than reconstructing one from in-memory
    values, so this can never drift from what's actually in the database).
    Safe to call on both initial creation and later settlement. Returns
    False (never raises) on any failure, including missing R2 credentials
    -- same graceful no-op as every other R2 call in this codebase.
    """
    try:
        client, bucket = _r2_client()
        if client is None:
            return False
        body = json.dumps({c: row[c] for c in PICK_COLUMNS}, default=str).encode("utf-8")
        client.put_object(Bucket=bucket, Key=_pick_key(row), Body=body, ContentType="application/json")
        return True
    except Exception as e:
        print(f"[live_log] append FAILED (non-fatal, real DB row is unaffected): {e}")
        return False


def restore_gap(conn) -> int:
    """
    Call once on every boot, unconditionally -- whether or not
    core.db_backup.restore_latest_backup() ran this time. Upserts every
    live_log object into forward_picks, keyed on the table's own real
    UNIQUE constraint (sport, espn_event_id, market, side, line), so this
    is fully idempotent: a normal boot with an intact, current database
    just re-applies values that are already identical (a no-op in effect),
    while a boot after a Volume wipe fills in everything logged or settled
    after the last periodic snapshot. `conn` must already have forward_picks
    created (call after database.init_db()). Returns the number of rows
    processed (not necessarily changed) for the startup log line.
    """
    try:
        client, bucket = _r2_client()
        if client is None:
            return 0
    except Exception as e:
        print(f"[live_log] restore_gap FAILED to connect to R2 (non-fatal): {e}")
        return 0

    processed = 0
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=LIVE_LOG_PREFIX):
            for obj in page.get("Contents", []):
                try:
                    body = client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                    row = json.loads(body)
                    placeholders = ",".join(f":{c}" for c in PICK_COLUMNS)
                    updates = ",".join(f"{c}=excluded.{c}" for c in PICK_COLUMNS if c not in
                                        ("sport", "espn_event_id", "market", "side", "line"))
                    # Conflict target must match database.py's REAL unique
                    # index exactly (idx_forward_picks_unique, keyed on
                    # COALESCE(line, -999999)) -- not the plain bare-columns
                    # UNIQUE() table constraint. SQLite treats NULL as
                    # distinct from every other NULL, so a plain
                    # ON CONFLICT(...,line) target would never actually
                    # match for moneyline picks (line is always NULL there)
                    # and would silently INSERT a duplicate row on every
                    # second write instead of upserting -- exactly the same
                    # real bug _fix_forward_picks_null_line_dupes() already
                    # exists to fix for the table's own writes, caught here
                    # by testing the round-trip before shipping rather than
                    # assuming the naive conflict target would work.
                    conn.execute(
                        f"INSERT INTO forward_picks ({','.join(PICK_COLUMNS)}) VALUES ({placeholders}) "
                        f"ON CONFLICT(sport, espn_event_id, market, side, COALESCE(line, -999999)) "
                        f"DO UPDATE SET {updates}",
                        row,
                    )
                    processed += 1
                except Exception as e:
                    print(f"[live_log] failed to restore {obj['Key']} (skipped, non-fatal): {e}")
        conn.commit()
    except Exception as e:
        print(f"[live_log] restore_gap listing FAILED (non-fatal): {e}")
    return processed
