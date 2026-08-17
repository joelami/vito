"""
FastAPI app: wires the NFL data pipeline (loader -> features -> power
ratings -> ML -> ensemble -> backtest) together at startup and exposes it
through the routes documented for the frontend. All routes live flat in
this file, comment-banner grouped — same convention as the Hockey Scout App
and Photo Location Finder.
"""

import importlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
from core import odds_math, ensemble, edge_finder, parlay
from core.dispatch import build_pipeline as build_sport_pipeline, score_matchup as dispatch_score_matchup, LIVE_SPORTS
from sports.nfl import config as nfl_config
from sports.nfl.matchup import score_matchup

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="Vito")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

_data = {}


# ---------------------------------------------------------------------------
# Startup: build the entire NFL model pipeline once, in memory. This is the
# same pattern the Hockey Scout App uses (CSV -> DataFrames at boot) rather
# than materializing the odds history into SQLite.
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    from core.dataset_sync import sync_datasets
    sync_datasets()  # no-op if Datasets/ is already present (local dev, or a persisted volume)

    database.init_db()
    nfl_pipeline = build_sport_pipeline("NFL", persist_backtest=True)
    _data.update(nfl_pipeline)  # legacy flat access — every existing NFL-only route reads _data directly

    # precompute opportunities for every historical game once (pure math, cheap) —
    # an app-level view over the pipeline, not part of the shared pipeline itself
    # since harness.py has no use for the full history browse list.
    history_df = _data["history_df"]
    stds, ecfg = _data["stds"], _data["ensemble_cfg"]
    history_opportunities = {}
    for gid, row in history_df.iterrows():
        if pd.isna(row.get("predicted_margin")):
            history_opportunities[gid] = []
            continue
        opps = edge_finder.evaluate_game(row, stds, nfl_config.ELO_POINTS_PER_MARGIN, ecfg, price_point="Close")
        history_opportunities[gid] = [o.to_dict() for o in opps]
        if opps:
            ml = ensemble.moneyline_prob(row, stds, nfl_config.ELO_POINTS_PER_MARGIN, ecfg)
            history_df.loc[gid, "model_home_win_prob"] = ml["blended_prob"]
    _data["history_opportunities"] = history_opportunities

    print(f"[startup] NFL: {len(_data['games'])} games loaded, {len(_data['oos_df'])} walk-forward "
          f"predictions, {len(_data['current_ratings'])} teams rated.")

    # Every other live sport (see core/dispatch.py's LIVE_SPORTS) — built the
    # same way harness.py builds them, kept in a separate per-sport dict
    # rather than flattened into `_data` so each league's model stays
    # completely isolated from the others (no shared/overwritten keys). A
    # sport that fails to build (e.g. missing dataset) is logged and skipped
    # rather than taking the whole app down — every other tab/sport should
    # still work.
    _data["pipelines"] = {"NFL": nfl_pipeline}
    for sport in LIVE_SPORTS:
        if sport == "NFL":
            continue
        try:
            _data["pipelines"][sport] = build_sport_pipeline(sport, persist_backtest=True)
            p = _data["pipelines"][sport]
            print(f"[startup] {sport}: {len(p['games'])} games loaded, {len(p['oos_df'])} walk-forward "
                  f"predictions, {len(p['current_ratings'])} teams rated.")
        except Exception as e:
            print(f"[startup] {sport} pipeline FAILED to build, skipping: {e}")

    # CFB: ratings-only, deliberately NOT in LIVE_SPORTS — its historical odds
    # coverage is too sparse/short-window for a trustworthy live edge-finder
    # (see docs/METHODOLOGY.md's NCAAF section), so it stays out of the ESPN
    # sync/harness/Suggestions path entirely. The power-rating engine itself
    # doesn't depend on odds at all though, so there's no reason to withhold
    # it from Ratings specifically — same pipeline, just never synced live or
    # exposed anywhere a bet would actually be suggested.
    try:
        _data["pipelines"]["CFB"] = build_sport_pipeline("CFB", persist_backtest=False)
        p = _data["pipelines"]["CFB"]
        print(f"[startup] CFB: {len(p['games'])} games loaded, {len(p['oos_df'])} walk-forward "
              f"predictions, {len(p['current_ratings'])} teams rated (ratings-only, not live-synced).")
    except Exception as e:
        print(f"[startup] CFB ratings pipeline FAILED to build, skipping: {e}")

    # Started last, after every pipeline above is already built — its own
    # immediate boot-time harness pass (see scheduler.py) is a second, full
    # pipeline rebuild, and letting the app's own startup (what the
    # healthcheck is waiting on) finish first avoids competing for CPU
    # during that already-slow window. No-op locally / anywhere
    # ENABLE_SCHEDULER isn't set to "1" — see scheduler.py's docstring.
    from scheduler import start_background_scheduler
    start_background_scheduler()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ManualGameCreate(BaseModel):
    date: str
    home_team: str
    away_team: str
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    home_line: Optional[float] = None
    home_line_odds: Optional[float] = None
    away_line_odds: Optional[float] = None
    total_line: Optional[float] = None
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None


class BetCreate(BaseModel):
    sport: str = "NFL"
    game_label: str
    market: str
    side: str
    line: Optional[float] = None
    odds_taken: float
    stake: float
    placed_at: str
    notes: Optional[str] = None


class BetUpdate(BaseModel):
    result: Optional[str] = None
    closing_odds: Optional[float] = None
    notes: Optional[str] = None


class ParlayLegIn(BaseModel):
    game_key: str
    market: str
    side: str
    line: Optional[float] = None
    model_prob: float
    market_odds: float
    market_fair_prob: float
    confidence: Optional[str] = None


class ParlayRequest(BaseModel):
    legs: list[ParlayLegIn]
    kelly_frac: float = 0.25


# ---------------------------------------------------------------------------
# API: Ratings
# ---------------------------------------------------------------------------
@app.get("/api/ratings")
def get_ratings(sport: str = "NFL"):
    sport = sport.upper()
    pipeline = _data["pipelines"].get(sport) if "pipelines" in _data else (_data if sport == "NFL" else None)
    if pipeline is None:
        raise HTTPException(404, f"no live pipeline built for sport={sport!r}")

    # Some sports declare a display filter for the Ratings tab because their
    # raw data mixes tiers that don't belong in one ranked list together —
    # currently just CFB (see its config.py's FBS_TEAMS docstring: no
    # conference/division column in the data at all, so small FCS/D-II
    # programs on a hot streak can otherwise rank above real blue-bloods).
    # FBS_TEAMS (an explicit allow-list) is preferred when a sport declares
    # one; MIN_GAMES_FOR_RATING_DISPLAY is a coarser fallback for a sport
    # that hasn't got a hand-built list. Both opt-in, off by default.
    sport_config = importlib.import_module(f"sports.{sport.lower()}.config")
    allow_list = getattr(sport_config, "FBS_TEAMS", None)
    min_games = getattr(sport_config, "MIN_GAMES_FOR_RATING_DISPLAY", None)

    ranked = sorted(pipeline["current_ratings"].items(), key=lambda x: -x[1])
    if allow_list:
        ranked = [(team, rating) for team, rating in ranked if team in allow_list]
    elif min_games:
        games_df = pipeline["games"]
        games_played = pd.concat([games_df["home_franchise"], games_df["away_franchise"]]).value_counts()
        ranked = [(team, rating) for team, rating in ranked if games_played.get(team, 0) >= min_games]

    return [
        {"rank": i + 1, "team": pipeline["latest_names"].get(team, team), "rating": round(rating, 1)}
        for i, (team, rating) in enumerate(ranked)
    ]


# ---------------------------------------------------------------------------
# API: Historical games
# ---------------------------------------------------------------------------
@app.get("/api/games/history")
def get_games_history(season: Optional[int] = None, team: Optional[str] = None,
                       market: Optional[str] = None, min_edge: Optional[float] = None,
                       limit: int = 50, offset: int = 0):
    df = _data["history_df"]
    mask = pd.Series(True, index=df.index)
    if season is not None:
        mask &= df["season"] == season
    if team is not None:
        mask &= (df["Home Team"] == team) | (df["Away Team"] == team)
    filtered = df[mask].sort_values("date", ascending=False)

    matches = []
    for gid, row in filtered.iterrows():
        opps = _data["history_opportunities"].get(gid, [])
        if market is not None:
            opps = [o for o in opps if o["market"] == market]
        if min_edge is not None and not any(o["edge_pct"] >= min_edge for o in opps):
            continue
        matches.append((gid, row, opps))

    total = len(matches)
    page = matches[offset:offset + limit]
    games_json = []
    for gid, row, opps in page:
        games_json.append({
            "game_id": gid, "date": row["date"].strftime("%Y-%m-%d"), "season": int(row["season"]),
            "home_team": row["Home Team"], "away_team": row["Away Team"],
            "home_score": int(row["home_score"]), "away_score": int(row["away_score"]),
            "is_playoff": bool(row["is_playoff"]), "is_neutral_venue": bool(row["is_neutral_venue"]),
            "home_odds_close": _safe(row.get("Home Odds Close")), "away_odds_close": _safe(row.get("Away Odds Close")),
            "home_line_close": _safe(row.get("Home Line Close")), "total_score_close": _safe(row.get("Total Score Close")),
            "model_home_win_prob": _safe(row.get("model_home_win_prob")),
            "model_predicted_margin": _safe(row.get("predicted_margin")),
            "model_predicted_total": _safe(row.get("predicted_total")),
            "opportunities": opps,
        })
    return {"total": total, "games": games_json}


def _safe(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return v


# ---------------------------------------------------------------------------
# API: Manually-entered upcoming games
# ---------------------------------------------------------------------------
def _row_to_manual_dict(row: dict) -> dict:
    return {
        "id": row["id"], "date": row["date"], "home_team": row["home_team"], "away_team": row["away_team"],
        "home_odds": row["home_odds"], "away_odds": row["away_odds"],
        "home_line": row["home_line"], "home_line_odds": row["home_line_odds"], "away_line_odds": row["away_line_odds"],
        "total_line": row["total_line"], "over_odds": row["over_odds"], "under_odds": row["under_odds"],
        "created_at": row["created_at"],
    }


def _score_manual_game(g: dict) -> list:
    """Scores a manually-entered game via the shared `score_matchup` helper
    (also used by harness.py for ESPN-synced real games) — manual entries only
    ever have one odds snapshot, treated as "Close" pricing."""
    market_odds = {
        "Home Odds Close": g["home_odds"], "Away Odds Close": g["away_odds"],
        "Home Line Close": g["home_line"], "Home Line Odds Close": g["home_line_odds"],
        "Away Line Odds Close": g["away_line_odds"],
        "Total Score Close": g["total_line"], "Total Score Over Close": g["over_odds"],
        "Total Score Under Close": g["under_odds"],
    }
    opps = score_matchup(_data, g["home_team"], g["away_team"], g["date"], market_odds, price_point="Close")
    return [o.to_dict() for o in opps]


@app.get("/api/games/manual")
def list_manual_games():
    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM manual_games ORDER BY date").fetchall()]
    return [dict(_row_to_manual_dict(r), opportunities=_score_manual_game(r)) for r in rows]


@app.post("/api/games/manual", status_code=201)
def create_manual_game(g: ManualGameCreate):
    with database.get_db() as conn:
        cur = conn.execute(
            "INSERT INTO manual_games (date, home_team, away_team, home_odds, away_odds, home_line, "
            "home_line_odds, away_line_odds, total_line, over_odds, under_odds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (g.date, g.home_team, g.away_team, g.home_odds, g.away_odds, g.home_line,
             g.home_line_odds, g.away_line_odds, g.total_line, g.over_odds, g.under_odds),
        )
        row = dict(conn.execute("SELECT * FROM manual_games WHERE id = ?", (cur.lastrowid,)).fetchone())
    return dict(_row_to_manual_dict(row), opportunities=_score_manual_game(row))


@app.delete("/api/games/manual/{game_id}", status_code=204)
def delete_manual_game(game_id: int):
    with database.get_db() as conn:
        conn.execute("DELETE FROM manual_games WHERE id = ?", (game_id,))
    return None


# ---------------------------------------------------------------------------
# API: Injuries — starting-QB availability, lazy-fetched (not at startup,
# since a 32-team ESPN roster pull takes real time) and cached in memory for
# an hour. This is a CONTEXT FLAG shown next to picks, not a trained model
# input — see sports/nfl/injuries.py's module docstring for why.
# ---------------------------------------------------------------------------
@app.get("/api/injuries/qb-status")
def get_qb_statuses(refresh: bool = False):
    from datetime import datetime, timedelta
    from sports.nfl.injuries import fetch_all_qb_statuses

    fetched_at = _data.get("qb_statuses_fetched_at")
    stale = fetched_at is None or (datetime.now() - fetched_at) > timedelta(hours=1)
    if refresh or stale:
        _data["qb_statuses"] = fetch_all_qb_statuses()
        _data["qb_statuses_fetched_at"] = datetime.now()

    return {
        "fetched_at": _data["qb_statuses_fetched_at"].isoformat(),
        "statuses": {
            _data["latest_names"].get(fr, fr): s for fr, s in _data.get("qb_statuses", {}).items()
        },
    }


# ---------------------------------------------------------------------------
# API: MLB probable starting pitchers — lazy-fetched, cached 1hr, same
# pattern as QB status above. Different shape on purpose: WHO starts is
# itself the live question in MLB (unlike QB), so this returns the
# announced probable + season ERA/W-L per side per game, not an
# out/questionable flag — see sports/mlb/probables.py's module docstring.
# ---------------------------------------------------------------------------
@app.get("/api/mlb/probable-pitchers")
def get_mlb_probable_pitchers(refresh: bool = False):
    from datetime import datetime, timedelta
    from sports.mlb.probables import fetch_probable_pitchers

    fetched_at = _data.get("mlb_probables_fetched_at")
    stale = fetched_at is None or (datetime.now() - fetched_at) > timedelta(hours=1)
    if refresh or stale:
        start = datetime.now()
        end = start + timedelta(days=5)
        _data["mlb_probables"] = fetch_probable_pitchers(dates=f"{start:%Y%m%d}-{end:%Y%m%d}")
        _data["mlb_probables_fetched_at"] = datetime.now()

    return {
        "fetched_at": _data["mlb_probables_fetched_at"].isoformat(),
        "probables": _data.get("mlb_probables", {}),
    }


# ---------------------------------------------------------------------------
# API: Upcoming games (real schedule + odds, synced from ESPN by harness.py —
# see backend/harness.py and scripts/run_harness.sh). Scored at the LATEST
# synced price ("Close" columns, which keep moving right up to kickoff) so
# this reflects "the situation right now," distinct from forward_picks below
# which are permanently snapshotted at the honest opening price.
# ---------------------------------------------------------------------------
@app.get("/api/games/upcoming")
def get_upcoming_games(sport: str = "NFL"):
    sport = sport.upper()
    pipeline = _data["pipelines"].get(sport) if "pipelines" in _data else (_data if sport == "NFL" else None)
    if pipeline is None:
        raise HTTPException(404, f"no live pipeline built for sport={sport!r}")

    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM espn_games WHERE sport = ? AND completed = 0 ORDER BY date", (sport,)
        ).fetchall()]

    out = []
    for r in rows:
        market_odds = {
            "Home Odds Close": r["home_odds_close"], "Away Odds Close": r["away_odds_close"],
            "Home Line Close": r["home_line_close"], "Home Line Odds Close": r["home_line_odds_close"],
            "Away Line Odds Close": r["away_line_odds_close"],
            "Total Score Close": r["total_close"], "Total Score Over Close": r["over_odds_close"],
            "Total Score Under Close": r["under_odds_close"],
        }
        opps = []
        if any(v is not None for v in market_odds.values()):
            # NFL keeps its own dedicated scorer (weather, Pythagorean features);
            # every other sport goes through the generic dispatcher — same
            # decision harness.py makes, made in exactly one place (core/dispatch.py)
            # so the live app and the offline job can never silently disagree
            # about which sport uses which scorer.
            opps = [o.to_dict() for o in dispatch_score_matchup(
                sport, pipeline, r["home_team"], r["away_team"], r["date"], market_odds,
                is_playoff=bool(r["is_playoff"]), is_neutral_venue=bool(r["is_neutral_venue"]),
                price_point="Close",
            )]
        out.append({
            "espn_event_id": r["espn_event_id"], "date": r["date"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "is_neutral_venue": bool(r["is_neutral_venue"]), "is_playoff": bool(r["is_playoff"]),
            "home_odds": r["home_odds_close"], "away_odds": r["away_odds_close"],
            "home_line": r["home_line_close"], "total_line": r["total_close"],
            "last_synced_at": r["last_synced_at"],
            "opportunities": opps,
        })
    return out


# ---------------------------------------------------------------------------
# API: Backtest
# ---------------------------------------------------------------------------
@app.get("/api/backtest")
def get_backtest():
    return _data["backtest_summary"]


# ---------------------------------------------------------------------------
# API: Forward test — the harness's real, ongoing track record. Every
# qualifying edge the model found in a real (not historical) game, logged
# automatically at the honest opening price the moment it was synced,
# settled against the real result once the game finished. This is NOT the
# backtest (which replays history) and NOT the user's bet log (which only
# has bets actually placed) — it's what the model would have done, for real,
# going forward. See docs/METHODOLOGY.md for why this is the bar that
# actually matters more than the backtest.
# ---------------------------------------------------------------------------
@app.get("/api/forward-test/summary")
def get_forward_test_summary():
    """
    The cross-league version of /api/forward-test below — "Vito's record
    since going live," aggregated across every sport instead of one at a
    time. Same honest-numbers discipline: real settled picks only, priced
    at the open, nothing backtested or hypothetical.
    """
    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM forward_picks").fetchall()]

    settled = [r for r in rows if r["settled"]]
    decided = [r for r in settled if r["result"] != "push"]
    wins = sum(1 for r in decided if r["result"] == "win")
    clv_vals = [r["clv_pct"] for r in settled if r["clv_pct"] is not None]
    total_profit = sum(r["profit_units"] or 0.0 for r in settled)
    live_since = min((r["snapshotted_at"] for r in rows), default=None)

    def sport_summary(sp):
        s = [r for r in settled if r["sport"] == sp]
        sd = [r for r in s if r["result"] != "push"]
        sw = sum(1 for r in sd if r["result"] == "win")
        return {
            "sport": sp, "bets": len(s),
            "pending": len([r for r in rows if r["sport"] == sp and not r["settled"]]),
            "hit_rate": (sw / len(sd)) if sd else None,
            "roi_pct": (sum(r["profit_units"] or 0.0 for r in s) / len(s) * 100.0) if s else None,
        }

    return {
        "live_since": live_since,
        "overall": {
            "bets": len(settled), "pending": len(rows) - len(settled),
            "hit_rate": (wins / len(decided)) if decided else None,
            "roi_pct": (total_profit / len(settled) * 100.0) if settled else None,
            "avg_clv_pct": (sum(clv_vals) / len(clv_vals)) if clv_vals else None,
            "pct_positive_clv": (sum(1 for v in clv_vals if v > 0) / len(clv_vals)) if clv_vals else None,
        },
        "by_sport": [sport_summary(sp) for sp in LIVE_SPORTS],
    }


@app.get("/api/forward-test")
def get_forward_test(sport: str = "NFL"):
    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM forward_picks WHERE sport = ? ORDER BY date DESC", (sport,)
        ).fetchall()]

    settled = [r for r in rows if r["settled"]]
    decided = [r for r in settled if r["result"] != "push"]
    wins = sum(1 for r in decided if r["result"] == "win")
    clv_vals = [r["clv_pct"] for r in settled if r["clv_pct"] is not None]
    total_profit = sum(r["profit_units"] or 0.0 for r in settled)

    def market_summary(market):
        m = [r for r in settled if r["market"] == market]
        md = [r for r in m if r["result"] != "push"]
        mw = sum(1 for r in md if r["result"] == "win")
        mclv = [r["clv_pct"] for r in m if r["clv_pct"] is not None]
        return {
            "market": market, "bets": len(m),
            "hit_rate": (mw / len(md)) if md else None,
            "roi_pct": (sum(r["profit_units"] or 0.0 for r in m) / len(m) * 100.0) if m else None,
            "avg_clv_pct": (sum(mclv) / len(mclv)) if mclv else None,
        }

    overall = {
        "bets": len(settled), "pending": len(rows) - len(settled),
        "hit_rate": (wins / len(decided)) if decided else None,
        "roi_pct": (total_profit / len(settled) * 100.0) if settled else None,
        "avg_clv_pct": (sum(clv_vals) / len(clv_vals)) if clv_vals else None,
        "pct_positive_clv": (sum(1 for v in clv_vals if v > 0) / len(clv_vals)) if clv_vals else None,
    }
    return {
        "overall": overall,
        "by_market": [market_summary(m) for m in ("moneyline", "spread", "total")],
        "picks": rows,
    }


@app.get("/api/forward-test/parlays")
def get_forward_test_parlays():
    """
    The parlay counterpart to /api/forward-test above — grades exactly the
    combos the harness snapshotted as "Suggested Parlays" (see harness.py's
    snapshot_new_parlays), the same honest-numbers discipline: real settled
    parlays only, priced off each leg's own real opening odds. Cross-league
    by nature (a parlay's legs can span sports), so unlike /api/forward-test
    this isn't filtered by `sport`.
    """
    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM forward_parlays ORDER BY snapshotted_at DESC"
        ).fetchall()]
    for r in rows:
        r["legs"] = json.loads(r.pop("legs_json"))
        r["pick_ids"] = json.loads(r["pick_ids"])

    settled = [r for r in rows if r["settled"]]
    decided = [r for r in settled if r["result"] != "push"]
    wins = sum(1 for r in decided if r["result"] == "win")
    total_profit = sum(r["profit_units"] or 0.0 for r in settled)

    def size_summary(n):
        s = [r for r in settled if r["leg_count"] == n]
        sd = [r for r in s if r["result"] != "push"]
        sw = sum(1 for r in sd if r["result"] == "win")
        return {
            "leg_count": n, "bets": len(s),
            "hit_rate": (sw / len(sd)) if sd else None,
            "roi_pct": (sum(r["profit_units"] or 0.0 for r in s) / len(s) * 100.0) if s else None,
        }

    return {
        "overall": {
            "bets": len(settled), "pending": len(rows) - len(settled),
            "hit_rate": (wins / len(decided)) if decided else None,
            "roi_pct": (total_profit / len(settled) * 100.0) if settled else None,
        },
        "by_leg_count": [size_summary(n) for n in range(2, 6)],
        "parlays": rows,
    }


# ---------------------------------------------------------------------------
# API: Suggestions of the Day — the main-page view: every currently-live,
# already-qualifying edge across EVERY league in one place, plus cross-game
# parlay suggestions built from that same pool. "Currently qualifying" means
# an unsettled row in `forward_picks` — the harness only ever writes a row
# there once a real synced game clears the exact same edge/confidence bar
# the backtest uses (see harness.py's `snapshot_new_picks`), so nothing here
# is filtered or re-scored on the fly; this route only reads and groups what
# the harness already decided, honestly, at the real opening price.
#
# Each league's picks stay in their own bucket (`sports.{SPORT}.picks`) —
# never merged into one undifferentiated list — because the whole point of
# keeping every sport's model fully separate (see core/dispatch.py, sports/*)
# is that a viewer needs to be able to tell which league a pick came from
# and trust each one independently.
# ---------------------------------------------------------------------------
def _pending_picks(sport: str = None) -> list:
    with database.get_db() as conn:
        if sport:
            rows = conn.execute(
                "SELECT * FROM forward_picks WHERE sport = ? AND settled = 0 ORDER BY edge_pct DESC",
                (sport,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM forward_picks WHERE settled = 0 ORDER BY edge_pct DESC"
            ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/suggestions/daily")
def get_daily_suggestions(top_n: int = 20, max_parlay_legs: int = 5, parlays_per_size: int = 3):
    all_pending = _pending_picks()

    # Surfaced per-league so a viewer can tell how fresh each sport's data
    # actually is — the in-process scheduler (see scheduler.py) only syncs
    # once or twice a day now that this runs on Railway instead of a
    # constantly-open local dev session, so "when was this last pulled"
    # stopped being an implicit "just now" and needs to be an honest,
    # visible number. MAX(last_synced_at) across a sport's espn_games is the
    # real signal — it moves on every sync, full run or sync-only alike.
    with database.get_db() as conn:
        sync_rows = conn.execute(
            "SELECT sport, MAX(last_synced_at) AS last_synced_at FROM espn_games GROUP BY sport"
        ).fetchall()
    last_synced_by_sport = {r["sport"]: r["last_synced_at"] for r in sync_rows}

    by_sport = {}
    for sport in LIVE_SPORTS:
        picks = [p for p in all_pending if p["sport"] == sport]
        by_sport[sport] = {
            "count": len(picks), "picks": picks,
            "last_synced_at": last_synced_by_sport.get(sport),
        }

    top_picks = sorted(all_pending, key=lambda p: p["edge_pct"], reverse=True)[:top_n]

    # Parlay pool: one ParlayLeg per pending pick, game_key namespaced by
    # sport so two different leagues' games never collide on a coincidentally
    # equal espn_event_id. suggest_parlays() already restricts itself to the
    # single best-edge leg per game and never combines legs from the same
    # game — see core/parlay.py's module docstring for why.
    legs = [
        parlay.ParlayLeg(
            game_key=f"{p['sport']}:{p['espn_event_id']}", market=p["market"], side=p["side"],
            line=p["line"], model_prob=p["model_prob"], market_odds=p["market_odds"],
            market_fair_prob=p["market_fair_prob"], confidence=p["confidence"],
        )
        for p in all_pending
    ]
    # Grouped by leg count rather than one flat top-N-by-edge cut. Real
    # problem that fixed: a flat sort lets whichever leg-count happens to
    # produce the biggest compounded edge today crowd out every other size
    # — verified directly, even back when max_legs was already 4, the top-5
    # never once included a single 4-leg combo, purely because compounded
    # edge doesn't scale predictably with leg count (it depends on which
    # specific games/edges happen to be available, not the leg count
    # itself). `top_n` here is generous on purpose (not the final count
    # shown) so every size category that has ANY qualifying combination is
    # actually represented before `parlays_per_size` trims each one down.
    raw_parlays = parlay.suggest_parlays(legs, max_legs=max_parlay_legs, top_n=500)
    by_leg_count = {}
    for p in raw_parlays:
        by_leg_count.setdefault(len(p["legs"]), []).append(p)
    suggested_parlays = []
    for n in sorted(by_leg_count):
        suggested_parlays.extend(by_leg_count[n][:parlays_per_size])
    # attach display context (sport/matchup) back onto each leg in the response,
    # since ParlayLeg itself only carries the game_key, not human-readable labels
    label_by_key = {f"{p['sport']}:{p['espn_event_id']}":
                     f"{p['sport']} — {p['away_team']} @ {p['home_team']} ({p['date'][:10]})"
                     for p in all_pending}
    for parlay_dict in suggested_parlays:
        for leg in parlay_dict["legs"]:
            leg["game_label"] = label_by_key.get(leg["game_key"], leg["game_key"])

    return {
        "generated_at": datetime.now().isoformat(),
        "sports": by_sport,
        "top_picks": top_picks,
        "parlays": suggested_parlays,
    }


# ---------------------------------------------------------------------------
# API: Bet log
# ---------------------------------------------------------------------------
@app.get("/api/bets")
def list_bets(status: Optional[str] = None):
    with database.get_db() as conn:
        if status:
            rows = conn.execute("SELECT * FROM bets WHERE result = ? ORDER BY placed_at DESC", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM bets ORDER BY placed_at DESC").fetchall()
        return [dict(r) for r in rows]


@app.post("/api/bets", status_code=201)
def create_bet(b: BetCreate):
    with database.get_db() as conn:
        cur = conn.execute(
            "INSERT INTO bets (sport, game_label, market, side, line, odds_taken, stake, placed_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (b.sport, b.game_label, b.market, b.side, b.line, b.odds_taken, b.stake, b.placed_at, b.notes),
        )
        row = conn.execute("SELECT * FROM bets WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


@app.put("/api/bets/{bet_id}")
def update_bet(bet_id: int, b: BetUpdate):
    with database.get_db() as conn:
        existing = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
        if not existing:
            raise HTTPException(404, "bet not found")
        result = b.result if b.result is not None else existing["result"]
        closing_odds = b.closing_odds if b.closing_odds is not None else existing["closing_odds"]
        notes = b.notes if b.notes is not None else existing["notes"]
        clv = existing["clv_pct"]
        if closing_odds is not None:
            clv = odds_math.clv_pct(existing["odds_taken"], closing_odds)
        conn.execute(
            "UPDATE bets SET result = ?, closing_odds = ?, clv_pct = ?, notes = ? WHERE id = ?",
            (result, closing_odds, clv, notes, bet_id),
        )
        return dict(conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone())


@app.delete("/api/bets/{bet_id}", status_code=204)
def delete_bet(bet_id: int):
    with database.get_db() as conn:
        conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))
    return None


# ---------------------------------------------------------------------------
# API: Parlay builder — combines legs the user picks (typically from the
# Dashboard's opportunities). Cross-game legs get a real combined edge;
# same-game legs are correlated and intentionally do NOT get a fabricated
# combined probability — see core/parlay.py's module docstring for why.
# ---------------------------------------------------------------------------
@app.post("/api/parlay/combine")
def combine_parlay(req: ParlayRequest):
    if not req.legs:
        raise HTTPException(400, "at least one leg is required")
    legs = [parlay.ParlayLeg(**leg.dict()) for leg in req.legs]
    result = parlay.build_parlay(legs, kelly_frac=req.kelly_frac)
    return result.to_dict()


# ---------------------------------------------------------------------------
# API: Bankroll (derived entirely from settled bets — no separate snapshots
# table, so there's only one source of truth)
# ---------------------------------------------------------------------------
@app.get("/api/bankroll")
def get_bankroll(starting_bankroll: float = 1000.0):
    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM bets WHERE result != 'pending' ORDER BY placed_at"
        ).fetchall()]

    bankroll = starting_bankroll
    total_staked = 0.0
    total_profit = 0.0
    curve = [{"date": rows[0]["placed_at"] if rows else None, "bankroll": bankroll}]
    for r in rows:
        stake = r["stake"]
        total_staked += stake
        if r["result"] == "win":
            profit = stake * (r["odds_taken"] - 1.0)
        elif r["result"] == "loss":
            profit = -stake
        else:  # push
            profit = 0.0
        total_profit += profit
        bankroll += profit
        curve.append({"date": r["placed_at"], "bankroll": bankroll})

    roi_pct = (total_profit / total_staked * 100.0) if total_staked else 0.0
    return {
        "starting_bankroll": starting_bankroll, "current_bankroll": bankroll,
        "total_staked": total_staked, "total_profit": total_profit, "roi_pct": roi_pct,
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# Frontend serving
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import os
    import uvicorn
    # Railway (and most PaaS hosts) assign the port dynamically via $PORT and
    # route external traffic to whatever it is — a hardcoded port means the
    # platform can never actually reach the app. Falls back to 8010 for local
    # dev, where nothing sets $PORT.
    port = int(os.environ.get("PORT", 8010))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
