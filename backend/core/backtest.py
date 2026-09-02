"""
Walk-forward backtest: replays historical games through the EXACT SAME
pipeline used for live picks (loader -> features -> power ratings -> ML ->
ensemble -> edge_finder), so there is no separate "backtest-only" shortcut
logic that could quietly diverge from what a live pick actually does.

Two honesty choices drive this module:
  1. Bets are evaluated against the OPENING line, not the closing line —
     you can never actually place a bet at a game's closing price (it's
     only known once betting closes), so pricing every backtested bet at
     open is what a real bettor could have gotten.
  2. Closing-line value (CLV) is tracked for every bet placed: how the
     price taken compares to where the market ultimately settled. Positive
     average CLV is a much more reliable long-run signal than a single
     ROI number computed on a small sample, since it doesn't depend on
     variance in who actually won each game.

Every summary returned here reports sample size and a rough standard error
on ROI alongside the point estimate — a positive ROI on 40 bets is not
evidence of anything on its own.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import edge_finder, ensemble


CLOSE_ODDS_COL = {
    ("moneyline", "home"): "Home Odds Close",
    ("moneyline", "away"): "Away Odds Close",
    ("spread", "home"): "Home Line Odds Close",
    ("spread", "away"): "Away Line Odds Close",
    ("total", "over"): "Total Score Over Close",
    ("total", "under"): "Total Score Under Close",
}


@dataclass
class BacktestConfig:
    min_edge_pct: float = 3.0
    allowed_confidence: tuple = ("Medium", "High")
    kelly_frac: float = 0.25
    starting_bankroll: float = 100.0
    price_point: str = "Open"   # the price a bet is actually placed at
    flat_stake: float = 1.0     # units, for the flat-staking ROI view


def settle_bet(market: str, side: str, line, row) -> int:
    """Returns 1 = win, 0 = push, -1 = loss, for a single bet against the actual
    result. Public so `harness.py` can settle real forward_picks against ESPN
    results with bit-for-bit identical logic to the historical backtest."""
    if market == "moneyline":
        won_home = row["home_win"] == 1
        return 1 if (won_home if side == "home" else not won_home) else -1
    if market == "spread":
        margin = row["actual_margin"]
        if side == "home":
            if margin == -line:
                return 0
            return 1 if margin > -line else -1
        else:  # away; line already stored as -home_line
            if margin == line:
                return 0
            return 1 if margin < line else -1
    if market == "total":
        total = row["actual_total"]
        if total == line:
            return 0
        if side == "over":
            return 1 if total > line else -1
        else:
            return 1 if total < line else -1
    raise ValueError(f"unknown market {market}")


def run_backtest(oos_df: pd.DataFrame, stds: ensemble.ResidualStds, elo_points_per_margin: float,
                  ensemble_cfg: ensemble.EnsembleConfig, cfg: BacktestConfig = None,
                  sport: str = None) -> pd.DataFrame:
    """
    `oos_df` = walk-forward feature+prediction DataFrame (feats joined with
    ml_models' out-of-sample predictions), one row per game. Returns one row
    per BET PLACED (not per game — a game can produce zero, one, or several
    qualifying bets), with everything needed for the summary tables and the
    bankroll curve.

    `sport`, if given, is passed straight through to edge_finder.evaluate_game()
    -- gates which moneyline confidence function is used (see edge_finder.py's
    MARKET_AGREEMENT_CONFIDENCE_SPORTS). Every existing caller that doesn't
    pass this keeps getting the always-safe old confidence_tier() behavior,
    unchanged -- this is additive, not a silent behavior change for anyone
    who hasn't opted in.
    """
    cfg = cfg or BacktestConfig()
    records = []

    for game_id, row in oos_df.iterrows():
        opps = edge_finder.evaluate_game(row, stds, elo_points_per_margin, ensemble_cfg,
                                          kelly_frac=cfg.kelly_frac, price_point=cfg.price_point,
                                          sport=sport)
        for o in opps:
            if o.edge_pct < cfg.min_edge_pct or o.confidence not in cfg.allowed_confidence:
                continue
            result = settle_bet(o.market, o.side, o.line, row)
            flat_profit = 0.0 if result == 0 else (cfg.flat_stake * (o.market_odds - 1.0) if result == 1
                                                     else -cfg.flat_stake)
            close_col = CLOSE_ODDS_COL[(o.market, o.side)]
            close_price = row.get(close_col) if hasattr(row, "get") else row[close_col]
            clv = None
            if close_price is not None and close_price == close_price:
                from . import odds_math
                clv = odds_math.clv_pct(o.market_odds, close_price)

            records.append({
                "game_id": game_id, "date": row["date"], "season": row["season"],
                "market": o.market, "side": o.side, "line": o.line,
                "market_odds": o.market_odds, "model_prob": o.model_prob,
                "market_fair_prob": o.market_fair_prob, "edge_pct": o.edge_pct,
                "confidence": o.confidence, "kelly_stake": o.kelly_stake,
                "result": result, "flat_profit": flat_profit, "clv_pct": clv,
            })

    bets = pd.DataFrame.from_records(records)
    if bets.empty:
        return bets
    return bets.sort_values("date", kind="stable").reset_index(drop=True)


def _roi_stderr(profits: pd.Series, stakes: float) -> float:
    """Rough standard error of ROI, treating each bet's profit as an i.i.d. sample."""
    n = len(profits)
    if n < 2:
        return float("nan")
    return float(profits.std(ddof=1) / math.sqrt(n) / stakes * 100.0)


def summarize(bets: pd.DataFrame, group_cols: list = None) -> pd.DataFrame:
    """
    Aggregate a bet log into ROI/hit-rate/CLV metrics, optionally grouped
    (e.g. by ['market'], by ['market', 'season'], by ['market', 'edge_bucket']).
    Always includes sample size and ROI standard error so small buckets read
    as noisy rather than as false confidence.
    """
    if bets.empty:
        return pd.DataFrame()

    df = bets.copy()
    decided = df[df["result"] != 0]  # exclude pushes from hit rate

    def _agg(g):
        n = len(g)
        wins = (g["result"] == 1).sum() if n else 0
        decided_n = (g["result"] != 0).sum()
        total_staked = n * 1.0  # flat_stake normalized to 1 unit for this view
        roi = g["flat_profit"].sum() / total_staked * 100.0 if n else float("nan")
        return pd.Series({
            "bets": n,
            "hit_rate": wins / decided_n if decided_n else float("nan"),
            "roi_pct": roi,
            "roi_stderr_pct": _roi_stderr(g["flat_profit"], 1.0),
            "avg_edge_pct": g["edge_pct"].mean(),
            "avg_clv_pct": g["clv_pct"].mean(skipna=True),
            "pct_positive_clv": (g["clv_pct"] > 0).mean(skipna=True) if g["clv_pct"].notna().any() else float("nan"),
        })

    if group_cols:
        return df.groupby(group_cols, observed=True).apply(_agg, include_groups=False).reset_index()
    return _agg(df).to_frame().T


def edge_bucket(edge_pct: float) -> str:
    if edge_pct < 5:
        return "3-5%"
    if edge_pct < 10:
        return "5-10%"
    if edge_pct < 20:
        return "10-20%"
    return "20%+"


def bankroll_curve(bets: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Simulates bankroll growth betting FRACTIONAL KELLY stakes in chronological
    order (one shared bankroll across all bet types/markets, since that's how
    a real bankroll works). Returns date + bankroll after each bet.
    """
    if bets.empty:
        return pd.DataFrame(columns=["date", "bankroll"])
    bankroll = cfg.starting_bankroll
    rows = []
    for _, b in bets.sort_values("date", kind="stable").iterrows():
        stake = bankroll * b["kelly_stake"]
        if b["result"] == 1:
            bankroll += stake * (b["market_odds"] - 1.0)
        elif b["result"] == -1:
            bankroll -= stake
        rows.append({"date": b["date"], "bankroll": bankroll})
    return pd.DataFrame(rows)
