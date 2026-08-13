"""
Runs the one-time holdout check (see core/research.py's module section,
"Held-out validation window") against every sport's CURRENT production
pipeline. Not a hypothesis test — there's no baseline/variant here, nothing
to adopt or reject. It's a report: does the model, exactly as it stands
today with everything already adopted, perform on its most recent season
(never individually used as a baseline/variant in any hypothesis test this
session) the way it performs on the seasons that WERE used for iteration?

Meant to be run occasionally (after a batch of research passes, not after
every single one) — that's what makes it a genuine check rather than just
another data point the process learns to game over time.

Usage: python3 -m core.holdout_check [SPORT ...]   (defaults to all 5)
"""

import importlib
import sys

from core import backtest, dispatch, research


def _sport_metrics(df, stds, elo_ppm, ecfg, bcfg) -> dict:
    if df.empty:
        return {"margin_corr": None, "total_corr": None, "roi_pct": 0.0, "roi_stderr_pct": 0.0, "bets": 0}
    bets_df = backtest.run_backtest(df, stds, elo_ppm, ecfg, bcfg)
    summary = backtest.summarize(bets_df)
    margin_corr = df["predicted_margin"].corr(df["actual_margin"])
    total_corr = df["predicted_total"].corr(df["actual_total"])
    if summary.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr, "roi_pct": 0.0, "roi_stderr_pct": 0.0, "bets": 0}
    row = summary.iloc[0]
    return {
        "margin_corr": margin_corr, "total_corr": total_corr,
        "roi_pct": float(row["roi_pct"]), "roi_stderr_pct": float(row["roi_stderr_pct"]), "bets": int(row["bets"]),
    }


def check_sport(sport: str) -> dict:
    sport = sport.upper()
    config = importlib.import_module(f"sports.{sport.lower()}.config")
    p = dispatch.build_pipeline(sport, persist_backtest=False)

    oos_df = p["oos_df"]
    stds, ecfg, bcfg = p["stds"], p["ensemble_cfg"], p["backtest_cfg"]
    elo_ppm = config.ELO_POINTS_PER_MARGIN

    # The holdout season must be picked from seasons with real, qualifying
    # bets — not just walk-forward predictions. Real bug caught running this
    # for the first time: MLB's real-odds archive only covers 2012-2021, so
    # `research.split_holdout()`'s naive "most recent season with OOS
    # predictions" landed on 2025 — a season with predictions but zero real
    # market data, giving a meaningless 0-bet "holdout." Determine the
    # holdout from `bets_df`'s own season column instead (a full backtest
    # run on the whole oos_df first), which only ever contains seasons that
    # actually produced a qualifying bet.
    full_bets = backtest.run_backtest(oos_df, stds, elo_ppm, ecfg, bcfg)
    if full_bets.empty:
        raise ValueError(f"{sport}: zero qualifying bets in the full backtest — nothing to hold out")
    bettable_seasons = sorted(full_bets["season"].unique())
    holdout_season = {bettable_seasons[-1]}

    iteration_df = oos_df[~oos_df["season"].isin(holdout_season)]
    holdout_df = oos_df[oos_df["season"].isin(holdout_season)]
    iteration_metrics = _sport_metrics(iteration_df, stds, elo_ppm, ecfg, bcfg)
    holdout_metrics = _sport_metrics(holdout_df, stds, elo_ppm, ecfg, bcfg)

    result = research.evaluate_holdout(sport, holdout_metrics, iteration_metrics)
    result["holdout_season"] = sorted(holdout_season)
    return result


def _fmt(m: dict) -> str:
    mc = f"{m['margin_corr']:.4f}" if m["margin_corr"] is not None else "—"
    tc = f"{m['total_corr']:.4f}" if m["total_corr"] is not None else "—"
    return f"margin_corr={mc} total_corr={tc} roi={m['roi_pct']:+.2f}% (±{m['roi_stderr_pct']:.2f}pp, n={m['bets']})"


if __name__ == "__main__":
    sports = [s.upper() for s in sys.argv[1:]] or ["NFL", "CFB", "MLB", "NBA", "NHL"]
    print("=" * 78)
    print("HOLDOUT CHECK — one-time report, not a hypothesis test")
    print("=" * 78)
    for sport in sports:
        try:
            r = check_sport(sport)
        except Exception as e:
            print(f"\n{sport}: FAILED — {e}")
            continue
        print(f"\n{sport} (holdout season: {r['holdout_season']})")
        print(f"  iteration : {_fmt(r['iteration_metrics'])}")
        print(f"  holdout   : {_fmt(r['holdout_metrics'])}")
        print(f"  delta     : {r['roi_delta_pct']:+.2f}pp — "
              f"{'within noise' if r['within_noise'] else 'OUTSIDE noise'}"
              f"{'  ** WARNING: holdout meaningfully worse **' if r['warning'] else ''}")
    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
