"""
Standing store for subgroup effects that are directionally real (stable
sign across a season split-half) but not yet strong enough, on the
evidence collected so far, to clear this project's adoption bar.

Why this exists, distinct from decision_log.jsonl: the decision log is a
narrative history -- "here's what we tried and what happened" -- and it's
deliberately append-only prose, not a queryable state store. A "watch, not
adopt yet" verdict needs one more thing the log alone can't give you: a
place that gets RE-CHECKED automatically as more live data accumulates,
so a real-but-underpowered signal doesn't just get typed into a log entry
and then forgotten. That re-check is what turns "notice it, but don't bet
the model on it yet" into an ongoing process instead of a one-time note.

This is intentionally a thin, structured JSONL sibling to decision_log --
same append-only shape, one line per event, but keyed so a later pass can
find "is nba_spread_favorite_underdog already on the list, and what did
it look like last time" without parsing free-text reasoning strings.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
WATCHLIST_PATH = BASE_DIR / "subgroup_watchlist.jsonl"


def add_or_update(name: str, sport: str, market: str, shrinkage_dict: dict,
                   z_score: float, stable_direction: bool, note: str) -> None:
    """One line per check -- history is kept (never overwritten), so a
    later read can see the effect's shrinkage weight and z-score trending
    up (more evidence, more trust) or down (regressing to noise) over
    successive research passes rather than just its current snapshot."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "name": name, "sport": sport, "market": market,
        "shrinkage": shrinkage_dict, "z_score": z_score,
        "stable_direction": stable_direction, "note": note,
    }
    with open(WATCHLIST_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_all() -> list:
    if not WATCHLIST_PATH.exists():
        return []
    return [json.loads(line) for line in WATCHLIST_PATH.read_text().splitlines() if line.strip()]


def latest_by_name() -> dict:
    """Most recent entry per `name` -- what a re-check pass should compare
    its new numbers against to see if an effect is strengthening,
    weakening, or holding steady."""
    latest = {}
    for entry in read_all():
        latest[entry["name"]] = entry
    return latest


def stale_items(max_age_days: int = 30) -> list:
    """
    Entries not re-derived in over `max_age_days` -- surfaced by the
    harness as a reminder, same spirit as `confidence_audit`'s standing
    checks. Deliberately does NOT auto-re-run the underlying research
    script or auto-promote anything: every adoption decision this project
    has made required a real, reviewed re-derivation (population-
    contamination bugs were caught exactly this way, twice, this session)
    -- a watchlist item earning "adopt" is a real event worth a human
    looking at the fresh numbers, not something to happen silently in a
    background cron job. This only flags that it's time to look again.
    """
    stale = []
    cutoff = datetime.now() - timedelta(days=max_age_days)
    for name, e in latest_by_name().items():
        checked = datetime.fromisoformat(e["timestamp"])
        if checked < cutoff:
            stale.append(e)
    return stale


def print_watchlist() -> None:
    latest = latest_by_name()
    if not latest:
        print("Subgroup watchlist is empty.")
        return
    print(f"\n=== Subgroup watchlist ({len(latest)} tracked effect(s)) ===")
    for name, e in sorted(latest.items()):
        sh = e["shrinkage"]
        print(f"  [{e['sport']}/{e['market']}] {name}: z={e['z_score']:+.2f}, "
              f"shrunk_effect={sh['shrunk_effect']:+.3f} (subgroup weight {sh['subgroup_weight']*100:.0f}%), "
              f"last checked {e['timestamp'][:10]} -- {e['note']}")
