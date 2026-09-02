"""
Standing confidence-calibration check, runnable on demand (`python3
run_confidence_audit.py`) or from the harness's daily cycle. Real reason
this exists as durable infrastructure and not a one-off script: the
original agreement-based confidence_tier() was shown to be BACKWARDS for
NFL/MLB moneyline and, checked across every sport/market on 2026-09,
backwards for SPREAD in all five sports and for TOTAL in two of five --
and none of that was caught until someone asked directly, because nothing
in this system checked it automatically. This script is that check, so a
future regression (or a currently-unfixed market/sport combination) shows
up here instead of requiring another manual audit.

See core/confidence_audit.py for the actual computation; this is just the
runnable entry point across every live sport.
"""

import sys
import warnings

import pipeline
from core import backtest
from core.confidence_audit import audit_all_markets, print_audit
from core.dispatch import LIVE_SPORTS

warnings.filterwarnings("ignore", category=FutureWarning)

_BUILDERS = {
    "NFL": (lambda: pipeline.build_nfl_pipeline(persist_backtest=False), "sports.nfl.config"),
    "MLB": (lambda: pipeline.build_pipeline("MLB", persist_backtest=False), "sports.mlb.config"),
    "NBA": (lambda: pipeline.build_pipeline("NBA", persist_backtest=False), "sports.nba.config"),
    "NHL": (lambda: pipeline.build_pipeline("NHL", persist_backtest=False), "sports.nhl.config"),
    "CFB": (lambda: pipeline.build_pipeline("CFB", persist_backtest=False), "sports.cfb.config"),
}


def run(sports=None) -> list:
    """Returns the full list of per-sport, per-market audit results (see
    core/confidence_audit.py's audit_market() for the shape) -- callable
    from other code (e.g. a future scheduler hook), not just as a script."""
    sports = sports or LIVE_SPORTS
    all_results = []
    any_backwards = False
    for sport in sports:
        if sport not in _BUILDERS:
            continue
        build_fn, config_mod = _BUILDERS[sport]
        p = build_fn()
        elo_ppm = __import__(config_mod, fromlist=["x"]).ELO_POINTS_PER_MARGIN
        bets = backtest.run_backtest(p["oos_df"], p["stds"], elo_ppm, p["ensemble_cfg"], p["backtest_cfg"], sport=sport)
        results = audit_all_markets(bets)
        print_audit(sport, results)
        for r in results:
            r["sport"] = sport
            if r["verdict"] == "BACKWARDS":
                any_backwards = True
        all_results.extend(results)

    print()
    if any_backwards:
        print("[confidence_audit] AT LEAST ONE MARKET IS BACKWARDS -- see 'REAL PROBLEM' lines above. "
              "Do not trust that market's confidence badge until it's fixed.", file=sys.stderr)
    else:
        print("[confidence_audit] no backwards orderings found this run.")
    return all_results


if __name__ == "__main__":
    run()
