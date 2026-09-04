"""
Standing audit of whether confidence badges actually mean what they claim --
built in direct response to a real finding, not a hypothetical: the
original agreement-based confidence_tier() was shown to be BACKWARDS for
NFL and MLB moneyline (High confidence underperformed Medium, in real
historical backtests of thousands of bets each) and was never checked
against real outcomes until someone asked directly. That's the actual gap
this module exists to close -- not "fix confidence once," but make sure
this specific kind of unvalidated, plausible-sounding assumption can never
again sit unverified in this system for months without anyone noticing.

Real, honest limitation up front: a monotonic ordering (High hit-rate/ROI
>= Medium >= Low) on a HISTORICAL backtest is necessary but not sufficient
evidence a confidence scheme is trustworthy -- it's still one fixed sample,
same caveat this project applies everywhere else. This module reports the
real number either way (including "can't tell, sample too small") rather
than declaring victory the moment an ordering looks right once.
"""

import pandas as pd

# Canonical trust ordering -- used only to check whether real performance
# actually falls in this order, never assumed correct on its own.
TIER_RANK = {"High": 2, "Medium": 1, "Low": 0}


def audit_market(bets: pd.DataFrame, market: str, min_bets_for_verdict: int = 30) -> dict:
    """
    `bets` = a backtest.run_backtest()-shaped DataFrame (or any real,
    settled forward_picks/forward-test dataframe with the same `market`,
    `confidence`, `result`/`hit`, and profit columns). Returns a dict with
    per-tier hit-rate/ROI, whether the observed ordering is monotonic in
    the expected direction, and an honest verdict string -- never silently
    skips a market just because the current numbers are inconvenient.
    """
    sub = bets[bets["market"] == market].copy()
    if sub.empty:
        return {"market": market, "verdict": "no_data", "tiers": []}

    if "flat_profit" in sub.columns:
        profit_col = "flat_profit"
    elif "profit_units" in sub.columns:
        profit_col = "profit_units"
    else:
        profit_col = None

    tiers = []
    for tier in ("High", "Medium", "Low"):
        tb = sub[sub["confidence"] == tier]
        if tb.empty:
            continue
        hit_rate = None
        if "result" in tb.columns:
            if tb["result"].dtype == object:
                # string convention (e.g. live forward_picks): "win"/"loss"/"push"
                decided = tb[tb["result"] != "push"]
                if len(decided):
                    hit_rate = (decided["result"] == "win").mean()
            else:
                # numeric convention (core/backtest.py's run_backtest() output,
                # matching core/backtest.py's settle_bet(): 1=win, 0=push, -1=loss)
                decided = tb[tb["result"] != 0]
                if len(decided):
                    hit_rate = (decided["result"] == 1).mean()
        roi_pct = (tb[profit_col].sum() / len(tb) * 100.0) if profit_col and len(tb) else None
        tiers.append({
            "tier": tier, "bets": int(len(tb)), "hit_rate": hit_rate, "roi_pct": roi_pct,
            "enough_sample": len(tb) >= min_bets_for_verdict,
        })

    if len(tiers) < 2:
        verdict = "insufficient_tiers"
    else:
        ranked = sorted(tiers, key=lambda t: TIER_RANK.get(t["tier"], -1), reverse=True)
        comparable = [t for t in ranked if t["roi_pct"] is not None and t["enough_sample"]]
        if len(comparable) < 2:
            verdict = "insufficient_sample"
        else:
            monotonic = all(comparable[i]["roi_pct"] >= comparable[i + 1]["roi_pct"] - 1e-9
                             for i in range(len(comparable) - 1))
            verdict = "monotonic" if monotonic else "BACKWARDS"

    return {"market": market, "verdict": verdict, "tiers": tiers}


def audit_all_markets(bets: pd.DataFrame, min_bets_for_verdict: int = 30) -> list:
    """One audit_market() result per market actually present in `bets`."""
    return [audit_market(bets, m, min_bets_for_verdict) for m in sorted(bets["market"].unique())]


def audit_live_forward_test(sport: str, min_bets_for_verdict: int = 15) -> list:
    """
    Same audit, but against REAL settled forward_picks for `sport` -- the
    actual growing live track record, not the fixed historical backtest.
    This is the real point of "have the harness do part of this as we gain
    more data over time": a confidence scheme validated on a 2012-2021 (or
    similar) historical archive is still just a claim about the PAST until
    it's also checked against picks the app is ACTUALLY making right now.

    forward_picks already has the exact same shape audit_market() expects
    (`market`, `confidence`, `result` as a 'win'/'loss'/'push' string,
    `profit_units`) -- no reshaping needed, confirmed directly against the
    real schema before writing this rather than assumed.

    `min_bets_for_verdict` defaults lower than the historical version (15
    vs 30) because live samples are, by construction, much smaller and
    will be for a long time -- this doesn't lower the bar for what counts
    as a real signal, it just means most live audits will honestly report
    "insufficient_sample" for a while rather than never being checked at
    all. That's the correct, honest state, not a bug.
    """
    import database
    with database.get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM forward_picks WHERE sport = ? AND settled = 1", (sport,)
        ).fetchall()]
    if not rows:
        return []
    bets = pd.DataFrame(rows)
    return audit_all_markets(bets, min_bets_for_verdict)


def print_audit(sport: str, results: list) -> None:
    print(f"\n=== Confidence audit: {sport} ===")
    for r in results:
        flag = " <-- REAL PROBLEM, not validated" if r["verdict"] == "BACKWARDS" else ""
        print(f"  {r['market']}: {r['verdict']}{flag}")
        for t in r["tiers"]:
            hr = f"{t['hit_rate']*100:.1f}%" if t["hit_rate"] is not None else "n/a"
            roi = f"{t['roi_pct']:+.2f}%" if t["roi_pct"] is not None else "n/a"
            sample_note = "" if t["enough_sample"] else "  (small sample, not a verdict)"
            print(f"    {t['tier']:8s} n={t['bets']:5d}  hit={hr:7s}  roi={roi:8s}{sample_note}")
