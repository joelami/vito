"""
The one place that decides "NFL uses its own dedicated pipeline/matchup
scorer, every other live sport goes through the generic multi-sport path."
Both `harness.py` (the offline cron job) and `main.py` (the live FastAPI
app) import from here — this used to be duplicated logic living only in
harness.py, which is exactly the kind of drift `pipeline.py`'s own
docstring warns about ("extracting it here removes that risk entirely").
Keeping the dispatch itself in one place applies that same principle one
level up.
"""

import pipeline as pipeline_mod
from core import matchup as generic_matchup
from sports.nfl.matchup import score_matchup as score_matchup_nfl

# Every sport with a live ESPN scoreboard feed and a real backtested model
# behind it. CFB added 2026-08-25 at the app owner's explicit request, going
# in with eyes open on a real, documented caveat: unlike NFL/MLB/NBA/NHL,
# CFB's historical backtest is statistically inconclusive (-0.58% ROI ±
# 2.53% standard error — the data can't distinguish this model from having
# no edge at all — see docs/METHODOLOGY.md's NCAAF section), because the
# historical odds coverage is sparse/short-window and spreads/totals are a
# flat assumed -110/-110 rather than real two-sided market prices (CLV is
# therefore not meaningful for CFB the way it is for the other sports here).
# CFB suggestions/parlays/forward-test now work identically to every other
# live sport — same edge-finding, same confidence badges, same UI — but
# that backtest caveat is real and doesn't go away just because it's wired
# in; it's on the app owner's record as a known, accepted tradeoff, not a
# claim that CFB predictions are as trustworthy as NFL/MLB's.
LIVE_SPORTS = ["NFL", "MLB", "NBA", "NHL", "CFB"]


def build_pipeline(sport: str, persist_backtest: bool = False) -> dict:
    sport = sport.upper()
    return pipeline_mod.build_nfl_pipeline(persist_backtest=persist_backtest) if sport == "NFL" \
        else pipeline_mod.build_pipeline(sport, persist_backtest=persist_backtest)


def score_matchup(sport: str, pipeline: dict, home_team, away_team, game_date,
                   market_odds, is_playoff=False, is_neutral_venue=False, price_point="Close"):
    sport = sport.upper()
    if sport == "NFL":
        return score_matchup_nfl(pipeline, home_team, away_team, game_date, market_odds,
                                  is_playoff=is_playoff, is_neutral_venue=is_neutral_venue,
                                  price_point=price_point)
    return generic_matchup.score_matchup(sport, pipeline, home_team, away_team, game_date, market_odds,
                                          is_playoff=is_playoff, is_neutral_venue=is_neutral_venue,
                                          price_point=price_point)
