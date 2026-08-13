"""
Daily markdown reports — a durable, human-readable record of what the
harness saw and did each day, alongside the DB-backed `forward_picks`
tracking. This is part of the observability story (see `versioning.py`)
but scoped differently: versioning/decision-log/traces are about the
SYSTEM's own changes over time, this is about the SPORT's actual daily
activity — games synced, picks logged, picks settled, and a rolling
grade — in a format that's readable without querying the database.

One file per (sport, date) in `logs/daily_reports/{sport}/{date}.md`,
appended to / rewritten each time the harness runs that day (idempotent —
re-running the same day's report just regenerates it from current DB
state, never duplicates).
"""

from datetime import datetime
from pathlib import Path

import database

REPORTS_DIR = Path(__file__).parent.parent / "logs" / "daily_reports"


def _fmt_pick_row(p: dict) -> str:
    line = p["line"] if p["line"] is not None else "—"
    return (f"| {p['market']} | {p['side']} | {line} | {p['market_odds']:.2f} | "
            f"{p['edge_pct']:.1f}% | {p['confidence']} |")


def _fmt_settled_row(p: dict) -> str:
    line = p["line"] if p["line"] is not None else "—"
    clv = f"{p['clv_pct']:+.2f}%" if p["clv_pct"] is not None else "—"
    return (f"| {p['home_team']} vs {p['away_team']} | {p['market']} | {p['side']} | {line} | "
            f"{p['result']} | {clv} |")


def write_daily_report(sport: str, report_date: datetime, events: list,
                        new_picks: list, settled_today: list) -> Path:
    """
    `events` = parsed ESPN events synced this run (espn_client.parse_events
    shape). `new_picks`/`settled_today` = the forward_picks rows (as dicts)
    logged / settled during THIS harness run — pass what the run actually
    did, not a re-derived guess, so the report reflects reality exactly.
    """
    sport_dir = REPORTS_DIR / sport.lower()
    sport_dir.mkdir(parents=True, exist_ok=True)
    date_str = report_date.strftime("%Y-%m-%d")
    path = sport_dir / f"{date_str}.md"

    lines = [f"# {sport.upper()} — {date_str}", ""]

    lines.append(f"## Games Synced ({len(events)})")
    if events:
        for e in events:
            status = "FINAL" if e.get("completed") else "scheduled"
            score = f"{e['home_score']}-{e['away_score']}" if e.get("completed") else ""
            lines.append(f"- {e['away_team']} @ {e['home_team']} — {e['date']} ({status} {score})".rstrip())
    else:
        lines.append("_No games synced this run._")
    lines.append("")

    lines.append(f"## Picks Logged Today ({len(new_picks)})")
    if new_picks:
        lines.append("| Market | Side | Line | Odds | Edge | Confidence |")
        lines.append("|---|---|---|---|---|---|")
        lines.extend(_fmt_pick_row(p) for p in new_picks)
    else:
        lines.append("_No new qualifying picks this run._")
    lines.append("")

    lines.append(f"## Picks Settled Today ({len(settled_today)})")
    if settled_today:
        lines.append("| Game | Market | Side | Line | Result | CLV |")
        lines.append("|---|---|---|---|---|---|")
        lines.extend(_fmt_settled_row(p) for p in settled_today)
    else:
        lines.append("_No picks settled this run._")
    lines.append("")

    lines.append("## Rolling Track Record (all-time, through today)")
    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM forward_picks WHERE sport = ? AND settled = 1", (sport.upper(),)
        ).fetchall()]
    if rows:
        decided = [r for r in rows if r["result"] != "push"]
        wins = sum(1 for r in decided if r["result"] == "win")
        total_profit = sum(r["profit_units"] or 0.0 for r in rows)
        roi = total_profit / len(rows) * 100.0
        clv_vals = [r["clv_pct"] for r in rows if r["clv_pct"] is not None]
        hit_rate = wins / len(decided) * 100.0 if decided else float("nan")
        avg_clv = sum(clv_vals) / len(clv_vals) if clv_vals else float("nan")
        lines.append(f"- **{len(rows)}** settled picks | hit rate **{hit_rate:.1f}%** | "
                      f"ROI **{roi:+.2f}%** | avg CLV **{avg_clv:+.2f}%**")
        lines.append("")
        lines.append("_This is the honest number — forward-tested, not backtested. "
                      "See docs/METHODOLOGY.md for what would actually indicate a real edge._")
    else:
        lines.append("_No settled picks yet — grading starts once the first tracked game finishes._")
    lines.append("")

    lines.extend(_line_movement_section(rows))

    path.write_text("\n".join(lines))
    return path


def _line_movement_section(settled_rows: list) -> list:
    """
    Self-measured line movement — how far the price we actually took at
    "Open" typically drifted from the real closing price, using CLV data
    the harness already computes on every settled pick (`settle_bet`'s
    `clv_pct`). Costs nothing new: no extra ESPN calls, no extra sync
    frequency — this is purely a reporting pass over data collected during
    the existing once-nightly run.

    Framed after the same question a bought reference dataset
    (Datasets/Misc./Line Movement/betbetter_distance_to_close.csv) answers
    for other books/sports: "if you only see a game once, hours before it
    starts, how far off is that from the real close?" Our `clv_pct` is a
    relative-probability-percent measure, not literally the same unit as
    that file's `mean_distance_to_close_pp` (a raw probability-point
    difference) — so treat this as our own comparable-in-spirit measure,
    not a plug-in match to those numbers. The point of tracking it here is
    to eventually answer, with OUR OWN evidence instead of a guess: is a
    second, later daily sync (closer to game time) worth the added
    engineering cost for this specific sport? Right now the answer is
    "probably, for sports/markets where |avg CLV| stays consistently large"
    — this section exists so that's a measured conclusion, not a hunch.
    """
    clv_vals = [r["clv_pct"] for r in settled_rows if r["clv_pct"] is not None]
    lines = ["## Line Movement (self-measured — Open price vs. last known price before settlement)"]
    if not clv_vals:
        lines.append("_No settled picks with a captured closing price yet — this fills in as picks settle. "
                      "See harness.py's sync_espn_games() docstring: ESPN drops a game's odds block almost "
                      "immediately after completion, so this reflects the last price seen before that, not "
                      "always the true final closing line._")
        lines.append("")
        return lines
    n = len(clv_vals)
    mean_abs = sum(abs(v) for v in clv_vals) / n
    pct_gt2pp = sum(1 for v in clv_vals if abs(v) > 2.0) / n * 100.0
    lines.append(f"- **{n}** settled picks with a captured closing price | "
                 f"mean |distance to close| **{mean_abs:.2f}%** | "
                 f"**{pct_gt2pp:.1f}%** moved more than 2 percentage points")
    lines.append("")
    return lines
