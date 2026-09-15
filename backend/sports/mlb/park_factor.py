"""
Ballpark scoring-environment factor ("park factor"), joined onto the MLB
games frame. Real, decades-old sabermetric adjustment (Bill James
popularized it; every modern site -- FanGraphs, Baseball Reference,
Statcast -- publishes one): two teams can show identical trailing scoring
stats while playing in parks with genuinely different real run
environments (Coors Field vs. a pitcher's park), and nothing in the
current MLB feature set adjusts for that at all.

WHAT THIS IS: `park_scoring_factor` = (this park's trailing average total
runs per game) / (the league's trailing average total runs per game),
centered at 1.0 (>1.0 = hitter's park, <1.0 = pitcher's park). A single,
game-level value (not a home_/away_ pair) since a park's run environment
applies identically to BOTH offenses in a given game -- same shape as
`is_interleague`/`is_night` in features.py, not the home_/away_-split
shape starter-quality features use.

SOURCE: needs NO new data acquisition. `park_id` is already a column on
sports/mlb/loader.py's output (confirmed in decision_log.jsonl's
`travel_fatigue_short_rest_park_change` entry, which already keys off it).
This module is a straightforward empirical computation from games already
loaded -- the same "compute a trailing rate, shift/recenter" pattern every
other rolling feature in this project already uses.

RELOCATION SAFETY, checked directly (not assumed) before trusting `park_id`
as the grouping key -- the same discipline features.py's TEAM_LEAGUE/
TEAM_TIMEZONE tables apply for AL/NL and timezone continuity across team
relocations: Retrosheet already assigns a NEW `park_id` to a genuinely new
physical venue, so no separate park-id-to-venue mapping table is needed
here at all (unlike a team CODE, a park_id already IS the specific venue).
Verified against this dataset directly: the Yankees' home `park_id` is
NYC16 for 1990-04-12 through 2008-09-21 (the old Yankee Stadium) and NYC21
for 2009-04-16 onward (the new one) -- two distinct codes, clean handoff,
no overlap. The Braves show the same pattern across THREE parks (ATL01
1990-1996, ATL02 1997-2016, ATL03 2017-2025). A handful of `park_id`s
(WIL02, TOK01, LON01, SJU01, MEX02, and a few others) are shared across
MULTIPLE franchises' "home" games -- spot-checked and confirmed these are
genuine one-off/neutral-site/international games (London Series, Tokyo
Dome openers, hurricane-displacement games at a shared minor-league park,
etc.) at a real shared physical venue, which is exactly the correct
grouping for a scoring-ENVIRONMENT stat (the park's own effect on
scoring, independent of which franchise's game it technically was) --
not a data-quality problem to work around.

COVERAGE: every park_id in the full 1990-2025 dataset gets a real value
once it has at least one prior game recorded there; a park's true first-
ever game (or the very first games in the dataset overall, before the
league-wide baseline has any history) falls back to a neutral 1.0 factor
-- documented, not fabricated.
"""

import numpy as np
import pandas as pd

# Trailing games AT ONE PARK considered. 243 = 3 seasons x 81 home games/
# season, the standard multi-year window published park factors use
# (FanGraphs/Baseball Reference both compute multi-year, not single-season,
# park factors) -- a single season's ~81 games at one park is already a
# fairly thin sample for a rate this granular, and averaging across
# multiple seasons is standard practice specifically to smooth that out.
ROLL_N_PARK_GAMES = 243

# Trailing games LEAGUE-WIDE considered for the denominator. 2430 = one
# full 30-team season's worth of games (30*162/2) -- deliberately a
# ROLLING window, not an all-time expanding average, so the league
# baseline stays responsive to real era shifts (rule changes, the ball,
# the humidor, the 2023 shift ban) within about a year, instead of
# smearing 35 years of a materially different game into one flat number.
ROLL_N_LEAGUE_GAMES = 2430

# League-average total runs per game, computed directly from every
# completed game in this dataset (1990-2025, 83,215 games; mean
# actual_total = 9.166, rounded here same convention as features.py's
# LEAGUE_AVG_TEAM_SCORE). Used ONLY as the fallback for the very first
# games in the dataset overall, before ROLL_N_LEAGUE_GAMES worth of
# league-wide history exists at all.
LEAGUE_AVG_TOTAL_RUNS = 9.17


def attach_park_factor(games: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins one new column onto `games` (sports/mlb/loader.py's output,
    or anything carrying game_id/date/park_id/actual_total):
    `park_scoring_factor`, walk-forward-safe (current game's own runs
    never included in either the park or league trailing average that
    feature it).
    """
    work = games[["game_id", "date", "park_id", "actual_total"]].sort_values("date", kind="stable").copy()

    # League-wide trailing scoring baseline -- shift(1) so today's own
    # game is never in its own baseline, same walk-forward discipline as
    # every rolling feature in this project.
    league_avg = work["actual_total"].shift(1).rolling(ROLL_N_LEAGUE_GAMES, min_periods=1).mean()
    work["_league_avg_total_trailing"] = league_avg.fillna(LEAGUE_AVG_TOTAL_RUNS)

    # Park-level trailing scoring average, grouped by park_id (see module
    # docstring for why this needs no separate relocation-mapping table).
    # groupby preserves `work`'s own date-sorted order within each group,
    # so shift(1)+rolling here is exactly as walk-forward-safe as
    # features.py's team-grouped _rolling_form.
    grp = work.groupby("park_id", group_keys=False)
    park_avg = grp["actual_total"].apply(lambda s: s.shift(1).rolling(ROLL_N_PARK_GAMES, min_periods=1).mean())
    work["_park_avg_total_trailing"] = park_avg

    # A park's true first-ever game in this dataset has no prior game
    # there at all (shift(1) on a brand-new group is NaN) -- neutral 1.0
    # fallback, not an invented number.
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = work["_park_avg_total_trailing"] / work["_league_avg_total_trailing"]
    work["park_scoring_factor"] = factor.fillna(1.0)

    out = games.merge(work[["game_id", "park_scoring_factor"]], on="game_id", how="left")
    assert len(out) == len(games), "attach_park_factor must return exactly one row per input game"
    out["park_scoring_factor"] = out["park_scoring_factor"].fillna(1.0)
    return out


def current_park_factor_snapshot(games: pd.DataFrame) -> pd.DataFrame:
    """
    Each park's trailing scoring factor as of RIGHT NOW (including its
    most recent game, unlike the walk-forward columns in
    `attach_park_factor` above which deliberately exclude the current
    row) -- mirrors features.current_form_snapshot()'s exact convention
    and reasoning: no leakage concern, since every game in `games` is
    legitimately in the past relative to a brand-new one being scored.
    Indexed by `home_franchise` -- the specific team whose CURRENT home
    park this reflects (that franchise's most recent recorded home game's
    park_id), so live scoring can look this up the same simple way it
    looks up pf_l10/win_pct_l10 via `pipeline["current_form"].loc[...]`.
    """
    sorted_games = games.sort_values("date", kind="stable")

    league_avg_now = sorted_games["actual_total"].tail(ROLL_N_LEAGUE_GAMES).mean()
    if pd.isna(league_avg_now):
        league_avg_now = LEAGUE_AVG_TOTAL_RUNS

    park_avg_now = sorted_games.groupby("park_id")["actual_total"].apply(
        lambda s: s.tail(ROLL_N_PARK_GAMES).mean()
    )
    park_factor_now = park_avg_now / league_avg_now

    latest_park_by_franchise = sorted_games.groupby("home_franchise")["park_id"].last()
    factor_by_franchise = latest_park_by_franchise.map(park_factor_now).fillna(1.0)
    return factor_by_franchise.rename("park_scoring_factor").to_frame()


_current_snapshot_cache = None


def get_current_park_factor(home_franchise: str) -> float:
    """
    Live-scoring counterpart to `attach_park_factor` above, same real gap/
    convention sports/nfl/nflfastr_features.get_current_trailing_epa() and
    sports/mlb/features.py's other extra_matchup_features() hooks close:
    core/matchup.py's build_matchup_feature_row() has no loader row to
    read `park_scoring_factor` from for a brand-new, not-yet-played game.
    Cached lazily (module-level, reset via `reset_cache()`) so repeated
    live scoring calls in the same process don't re-load/re-aggregate the
    full historical dataset per call -- same pattern as nflfastr_features'
    `_team_game_epa_cache`.
    """
    global _current_snapshot_cache
    if _current_snapshot_cache is None:
        from .loader import load_games
        _current_snapshot_cache = current_park_factor_snapshot(load_games())
    if home_franchise in _current_snapshot_cache.index:
        return float(_current_snapshot_cache.loc[home_franchise, "park_scoring_factor"])
    return 1.0


def reset_cache() -> None:
    """See nflfastr_features.reset_caches()'s docstring for the real
    staleness bug class this exists to prevent in a long-lived process."""
    global _current_snapshot_cache
    _current_snapshot_cache = None
