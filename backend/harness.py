"""
Standalone evaluation harness — no FastAPI server required, run directly
(`python3 harness.py`) or via the scheduled launchd job (see
`scripts/run_harness.sh`). This is the "constantly evaluate and get better"
loop the app owner asked for. Each run:

  1. Builds the model exactly the way the live app does (`pipeline.py` — one
     shared source of truth, so this can never silently drift from what the
     app actually shows).
  2. Syncs real NFL games/odds from ESPN for a window around today.
  3. Snapshots every qualifying edge (same threshold/confidence bar as the
     historical backtest) into `forward_picks`, priced at "Open" — the same
     honesty discipline as the backtest: never priced at a number that only
     existed after the fact. Idempotent — re-running never double-logs a pick.
  4. Settles any forward_picks whose game has since finished, using the exact
     same win/loss/push logic as the historical backtest (`settle_bet`), and
     computes real closing-line value now that a true close price exists.
  5. Prints a plain-language summary — this is what shows up in the cron log.

Explicitly excludes preseason games (ESPN season.type == 1) — the model has
never seen a preseason game in training and has no business scoring one.

Every run also writes a step-by-step execution trace (with per-step timing)
to `logs/harness_traces/` via `versioning.RunTrace` — one of three
observability pillars for the system's own evolution, alongside version
manifests (`harness_versions/`) and the decision log (`decision_log.jsonl`);
see `versioning.py`'s module docstring for the full rationale.
"""

import sys
from datetime import datetime, timedelta

import database
import reports
from core import espn_client, odds_math
from core.backtest import settle_bet
from core.dispatch import build_pipeline as _build_pipeline, score_matchup as _score_matchup, LIVE_SPORTS
from versioning import RunTrace

# NFL keeps its dedicated matchup scorer (weather, Pythagorean features, its
# own tested history — see sports/nfl/matchup.py); every other sport goes
# through the generic dispatcher (core/matchup.py). Same reasoning as
# pipeline.py's build_nfl_pipeline staying separate from build_pipeline().
# See core/dispatch.py — the single place this decision is made, shared
# with main.py so the live app and this offline job can never drift apart.
NFL_SEASON_TYPE_FILTER = (2, 3)  # regular season, postseason — excludes preseason (1)

SYNC_PAST_DAYS = 5      # re-check games that may have finished since the last run
SYNC_FUTURE_DAYS = 14   # look far enough ahead to catch next week's slate early

# espn_games column (snake_case) each market/side's CLOSING price lives in,
# used to compute CLV once a game is settled.
CLOSE_COL = {
    ("moneyline", "home"): "home_odds_close", ("moneyline", "away"): "away_odds_close",
    ("spread", "home"): "home_line_odds_close", ("spread", "away"): "away_line_odds_close",
    ("total", "over"): "over_odds_close", ("total", "under"): "under_odds_close",
}

# espn_games column -> the "Open"-prefixed key score_matchup()/edge_finder expect
OPEN_KEY_TO_COL = {
    "Home Odds Open": "home_odds_open", "Away Odds Open": "away_odds_open",
    "Home Line Open": "home_line_open", "Home Line Odds Open": "home_line_odds_open",
    "Away Line Odds Open": "away_line_odds_open",
    "Total Score Open": "total_open", "Total Score Over Open": "over_odds_open",
    "Total Score Under Open": "under_odds_open",
}


def _date_range() -> str:
    start = datetime.now() - timedelta(days=SYNC_PAST_DAYS)
    end = datetime.now() + timedelta(days=SYNC_FUTURE_DAYS)
    return f"{start:%Y%m%d}-{end:%Y%m%d}"


def sync_espn_games(sport: str = "NFL") -> list:
    """Upserts synced events into espn_games. Returns the parsed event list
    (regular/postseason only) for immediate use by the caller — no need to
    re-read what was just written."""
    raw = espn_client.fetch_scoreboard(sport, dates=_date_range())
    events = [e for e in espn_client.parse_events(raw) if e.get("season_type") in (2, 3)]

    with database.get_db() as conn:
        for e in events:
            conn.execute("""
                INSERT INTO espn_games (
                    sport, espn_event_id, date, home_team, away_team, home_score, away_score,
                    completed, is_neutral_venue, is_playoff,
                    home_odds_open, home_odds_close, away_odds_open, away_odds_close,
                    home_line_open, home_line_close, home_line_odds_open, home_line_odds_close,
                    away_line_odds_open, away_line_odds_close,
                    total_open, total_close, over_odds_open, over_odds_close,
                    under_odds_open, under_odds_close, last_synced_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, datetime('now'))
                ON CONFLICT(sport, espn_event_id) DO UPDATE SET
                    home_score=excluded.home_score, away_score=excluded.away_score,
                    completed=excluded.completed, is_playoff=excluded.is_playoff,
                    -- "open" fields lock in on first sight and never move again, even if
                    -- ESPN's own feed changes its retroactive "open" value later
                    home_odds_open=COALESCE(espn_games.home_odds_open, excluded.home_odds_open),
                    away_odds_open=COALESCE(espn_games.away_odds_open, excluded.away_odds_open),
                    home_line_open=COALESCE(espn_games.home_line_open, excluded.home_line_open),
                    home_line_odds_open=COALESCE(espn_games.home_line_odds_open, excluded.home_line_odds_open),
                    away_line_odds_open=COALESCE(espn_games.away_line_odds_open, excluded.away_line_odds_open),
                    total_open=COALESCE(espn_games.total_open, excluded.total_open),
                    over_odds_open=COALESCE(espn_games.over_odds_open, excluded.over_odds_open),
                    under_odds_open=COALESCE(espn_games.under_odds_open, excluded.under_odds_open),
                    -- "close" fields are the opposite of "open" above (update to whatever
                    -- ESPN shows now, don't lock in) with one real fix: prefer the NEW value
                    -- but fall back to the last known one instead of overwriting with NULL.
                    -- Real bug found via a live MLB CLV check: ESPN drops a game's entire
                    -- `odds` block almost immediately once it's marked completed (confirmed
                    -- directly — a game synced with real odds one day had zero odds providers
                    -- the very next), and the old `excluded.home_odds_close` here blindly
                    -- overwrote a real, already-captured close price with NULL the moment
                    -- that happened — silently erasing CLV for every settled pick. This won't
                    -- capture the TRUE closing line (that would need syncing more often than
                    -- once/day, right up to first pitch) but it stops throwing away whatever
                    -- close price the most recent pre-completion sync actually saw.
                    home_odds_close=COALESCE(excluded.home_odds_close, espn_games.home_odds_close),
                    away_odds_close=COALESCE(excluded.away_odds_close, espn_games.away_odds_close),
                    home_line_close=COALESCE(excluded.home_line_close, espn_games.home_line_close),
                    home_line_odds_close=COALESCE(excluded.home_line_odds_close, espn_games.home_line_odds_close),
                    away_line_odds_close=COALESCE(excluded.away_line_odds_close, espn_games.away_line_odds_close),
                    total_close=COALESCE(excluded.total_close, espn_games.total_close),
                    over_odds_close=COALESCE(excluded.over_odds_close, espn_games.over_odds_close),
                    under_odds_close=COALESCE(excluded.under_odds_close, espn_games.under_odds_close),
                    last_synced_at=datetime('now')
            """, (
                sport, e["espn_event_id"], e["date"], e["home_team"], e["away_team"],
                e["home_score"], e["away_score"], int(e["completed"]), int(e["is_neutral_venue"]),
                int(e["is_playoff"]),
                e["Home Odds Open"], e["Home Odds Close"], e["Away Odds Open"], e["Away Odds Close"],
                e["Home Line Open"], e["Home Line Close"], e["Home Line Odds Open"], e["Home Line Odds Close"],
                e["Away Line Odds Open"], e["Away Line Odds Close"],
                e["Total Score Open"], e["Total Score Close"], e["Total Score Over Open"], e["Total Score Over Close"],
                e["Total Score Under Open"], e["Total Score Under Close"],
            ))
    return events


def snapshot_new_picks(pipeline: dict, sport: str, events: list) -> int:
    """For every synced, not-yet-completed game with an opening line, scores it
    through the model at Open pricing and logs any qualifying edge as a
    forward_pick. Duplicate-safe via the UNIQUE constraint (INSERT OR IGNORE)."""
    bcfg = pipeline["backtest_cfg"]
    logged = 0
    with database.get_db() as conn:
        for e in events:
            if e["completed"]:
                continue
            market_odds = {k: e.get(k) for k in OPEN_KEY_TO_COL}
            if not any(v is not None for v in market_odds.values()):
                continue  # market hasn't opened yet for this game

            opps = _score_matchup(
                sport, pipeline, e["home_team"], e["away_team"], e["date"], market_odds,
                is_playoff=e["is_playoff"], is_neutral_venue=e["is_neutral_venue"], price_point="Open",
            )
            for o in opps:
                if o.edge_pct < bcfg.min_edge_pct or o.confidence not in bcfg.allowed_confidence:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO forward_picks "
                    "(sport, espn_event_id, date, home_team, away_team, market, side, line, "
                    "model_prob, market_odds, market_fair_prob, edge_pct, confidence, kelly_stake) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sport, e["espn_event_id"], e["date"], e["home_team"], e["away_team"],
                     o.market, o.side, o.line, o.model_prob, o.market_odds, o.market_fair_prob,
                     o.edge_pct, o.confidence, o.kelly_stake),
                )
                if cur.rowcount:
                    logged += 1
    return logged


def resync_and_settle(sport: str) -> dict:
    """
    Lightweight companion to `run()` — refreshes odds (just the ESPN
    scoreboard call; no pipeline rebuild, no walk-forward retraining, no new
    picks snapshotted) and re-settles any picks whose game has since
    completed. Real problem this exists to fix: `run()` only executes once a
    day, and ESPN drops a game's entire odds block almost immediately after
    it completes (confirmed directly — see the CLV bug documented in
    docs/METHODOLOGY.md's Phase 3 section) — so a game that starts AND
    finishes between one day's `run()` and the next is synced exactly once,
    always after its odds are already gone, and can never get a real
    captured closing price no matter how well `sync_espn_games`'s upsert
    behaves.

    This is the deliberately cheap, "smarter, not costlier" answer to that —
    not continuous polling (expensive, and the app owner explicitly doesn't
    want it), just one well-timed SECOND daily pass, run later in the day
    when more of today's games are near or past their start time, that only
    ever does the two genuinely inexpensive steps (`sync_espn_games`,
    `settle_finished_picks` — both pure DB/HTTP, no model fitting). See
    `reports.py`'s `_line_movement_section` for how this gets measured over
    time: the two together give real, empirical evidence for whether an
    even-more-targeted schedule is worth building later, instead of a guess.
    """
    events = sync_espn_games(sport)
    settled = settle_finished_picks(sport)
    return {"games_synced": len(events), "picks_settled": settled}


def settle_finished_picks(sport: str) -> int:
    """Settles any pending forward_picks whose game is now completed, computing
    real win/loss/push + CLV against the true closing line."""
    settled = 0
    with database.get_db() as conn:
        pending = conn.execute(
            "SELECT * FROM forward_picks WHERE sport = ? AND settled = 0", (sport,)
        ).fetchall()
        for p in pending:
            game = conn.execute(
                "SELECT * FROM espn_games WHERE sport = ? AND espn_event_id = ?",
                (sport, p["espn_event_id"]),
            ).fetchone()
            if not game or not game["completed"] or game["home_score"] is None:
                continue

            row = {
                "home_win": 1 if game["home_score"] > game["away_score"] else 0,
                "actual_margin": game["home_score"] - game["away_score"],
                "actual_total": game["home_score"] + game["away_score"],
            }
            result_code = settle_bet(p["market"], p["side"], p["line"], row)
            result = {1: "win", 0: "push", -1: "loss"}[result_code]
            profit = 0.0 if result_code == 0 else (
                (p["market_odds"] - 1.0) if result_code == 1 else -1.0
            )

            close_col = CLOSE_COL.get((p["market"], p["side"]))
            close_price = game[close_col] if close_col else None
            clv = odds_math.clv_pct(p["market_odds"], close_price) if close_price else None

            conn.execute(
                "UPDATE forward_picks SET settled = 1, result = ?, profit_units = ?, clv_pct = ?, "
                "settled_at = datetime('now') WHERE id = ?",
                (result, profit, clv, p["id"]),
            )
            settled += 1
    return settled


def print_report(sport: str = "NFL"):
    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM forward_picks WHERE sport = ?", (sport,)
        ).fetchall()]
    settled_rows = [r for r in rows if r["settled"]]
    pending = len(rows) - len(settled_rows)
    print(f"[harness] {sport} forward-test track record: {len(settled_rows)} settled, {pending} pending.")
    if settled_rows:
        decided = [r for r in settled_rows if r["result"] != "push"]
        wins = sum(1 for r in decided if r["result"] == "win")
        total_profit = sum(r["profit_units"] or 0 for r in settled_rows)
        roi = total_profit / len(settled_rows) * 100.0
        clv_vals = [r["clv_pct"] for r in settled_rows if r["clv_pct"] is not None]
        hit_rate = wins / len(decided) * 100.0 if decided else float("nan")
        avg_clv = sum(clv_vals) / len(clv_vals) if clv_vals else float("nan")
        pos_clv = sum(1 for v in clv_vals if v > 0) / len(clv_vals) * 100.0 if clv_vals else float("nan")
        print(f"          hit rate {hit_rate:.1f}% | ROI {roi:+.2f}% | avg CLV {avg_clv:+.2f}% "
              f"| {pos_clv:.0f}% beat the close")
    else:
        print("          no settled picks yet — check back once this week's games finish.")


def run(sport: str = "NFL"):
    run_started_at = datetime.now()
    print(f"[harness] run started {run_started_at:%Y-%m-%d %H:%M:%S} for {sport}")
    trace = RunTrace(run_name=f"harness_{sport.lower()}")
    database.init_db()

    with trace.step("build_pipeline", sport=sport):
        pipeline = _build_pipeline(sport)

    with trace.step("sync_espn", sport=sport) as _:
        events = sync_espn_games(sport)
    trace.steps[-1]["details"]["games_synced"] = len(events)
    print(f"[harness] synced {len(events)} regular/postseason games from ESPN "
          f"(window {_date_range()})")

    with trace.step("snapshot_picks"):
        logged = snapshot_new_picks(pipeline, sport, events)
    trace.steps[-1]["details"]["picks_logged"] = logged
    print(f"[harness] logged {logged} new forward picks")

    with trace.step("settle_picks"):
        settled = settle_finished_picks(sport)
    trace.steps[-1]["details"]["picks_settled"] = settled
    print(f"[harness] settled {settled} picks from completed games")

    print_report(sport)

    with trace.step("write_daily_report"):
        new_picks, settled_today = _picks_since(sport, run_started_at)
        report_path = reports.write_daily_report(sport, run_started_at, events, new_picks, settled_today)
    print(f"[harness] daily report written to {report_path}")

    trace_path = trace.save()
    print(f"[harness] trace written to {trace_path} "
          f"(total {trace.steps and sum(s['duration_s'] for s in trace.steps):.1f}s across "
          f"{len(trace.steps)} steps)")


def _picks_since(sport: str, since: datetime) -> tuple:
    """Rows this run actually logged/settled, for the daily markdown report —
    queried by timestamp rather than threading return values through
    snapshot_new_picks/settle_finished_picks, so those functions' tested
    signatures don't need to change for a reporting-only need."""
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    with database.get_db() as conn:
        new_picks = [dict(r) for r in conn.execute(
            "SELECT * FROM forward_picks WHERE sport = ? AND snapshotted_at >= ?", (sport, since_str)
        ).fetchall()]
        settled_today = [dict(r) for r in conn.execute(
            "SELECT * FROM forward_picks WHERE sport = ? AND settled = 1 AND settled_at >= ?",
            (sport, since_str),
        ).fetchall()]
    return new_picks, settled_today


def _notify_failure(message: str):
    """
    Real gap this closes: the harness runs unattended via launchd, with
    stdout/stderr redirected to a log file (see scripts/run_harness.sh) that
    nobody actively watches — a failed run just sits there silently until
    someone happens to check. This surfaces it immediately as a native
    macOS notification instead. Best-effort only: if `osascript` isn't
    available (no GUI session, different OS, sandboxed environment) this
    silently does nothing rather than crashing the harness over a
    notification failing — the exit code (see below) is still the real,
    durable signal launchd/cron surfaces either way.
    """
    try:
        import subprocess
        script = f'display notification "{message}" with title "Sports Bet harness" sound name "Basso"'
        subprocess.run(["osascript", "-e", script], timeout=5, capture_output=True)
    except Exception:
        pass


if __name__ == "__main__":
    args = sys.argv[1:]
    sync_only = "--sync-only" in args
    sports_to_run = [a for a in args if not a.startswith("--")] or LIVE_SPORTS

    # --sync-only never touches Datasets/ (pure DB/ESPN operations, no pipeline
    # rebuild — see resync_and_settle()'s docstring), so skip the download
    # entirely on that path. The full run() path needs it (build_pipeline()
    # reads the historical dataset files) — same no-op-if-present sync main.py
    # uses at startup, since harness.py runs as its own standalone process
    # (e.g. a separate Railway Cron service) with no shared state with the app.
    if not sync_only:
        from core.dataset_sync import sync_datasets
        sync_datasets()

    failures = []
    for sport in sports_to_run:
        sport = sport.upper()
        try:
            if sync_only:
                result = resync_and_settle(sport)
                print(f"[harness] {sport} sync-only: {result['games_synced']} games refreshed, "
                      f"{result['picks_settled']} picks settled")
            else:
                run(sport)
        except Exception as e:
            print(f"[harness] {sport} FAILED: {e}", file=sys.stderr)
            failures.append(sport)
    if failures:
        # one sport's failure (e.g. a season with zero synced games right now)
        # shouldn't stop the others from running and being graded — but the
        # run as a whole still needs to exit non-zero so cron/launchd surfaces it.
        msg = f"completed with failures in: {', '.join(failures)}"
        print(f"[harness] {msg}", file=sys.stderr)
        _notify_failure(msg)
        sys.exit(1)
