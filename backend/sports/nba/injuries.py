"""
Starting-five availability check across the league, via ESPN's roster +
depth-chart endpoints (`core.espn_client`) -- NBA's counterpart to
sports/nfl/injuries.py's QB-availability flag and sports/mlb/probables.py's
probable-pitcher flag. Deliberately NOT disk-cached across days, same
reasoning as both: injury status is time-sensitive (changes day to day,
sometimes hour to hour close to tip-off), so every call fetches fresh.

WHY THIS EXISTS, AND WHY IT ISN'T A MODEL INPUT -- read before changing scope
-------------------------------------------------------------------------
This module was written as part of a real investigation (real HTTP requests
made, not assumed) into whether NBA roster-availability data could clear
this project's bar for a genuine, backtested ML_FEATURE_COLS input, the way
sports/nba/research_starters_out.py's box-score-derived starters-out
feature was tested. Two things were confirmed directly:

1. ESPN's LIVE NBA data is real and good -- arguably better-shaped than
   NFL's. A dedicated league-wide endpoint
   (`site.api.espn.com/.../basketball/nba/injuries`) exists with real
   per-player status/date/comment, and NBA teams have a genuine 5-position
   depth chart (pg/sg/sf/pf/c) -- an actual ESPN-designated starting five,
   not NFL's "assume roster-listing order, prefer depth chart if present"
   fallback heuristic.

2. ESPN's HISTORICAL depth is confirmed to be ZERO. Directly tested: a
   `summary?event=` call for a real 2015-03-01 game (Clippers @ Bulls,
   event 400579169) and a real 2005-12-01 game both returned an `injuries`
   block reflecting TODAY's live injury report, not that date's -- the
   2005-12-01 game's response listed "Santi Aldama," an athlete who did not
   enter the NBA until 2021. Passing `?dates=`/`?season=` to the dedicated
   `/injuries` endpoint had no effect either -- it always returns the
   current live report. ESPN cannot be used to backtest this, at all.

Given that, `sports/nba/research_starters_out.py` tested a DIFFERENT real
historical source instead -- `Datasets/NBA/nba-box-scores.csv`'s player-
level DNP/inactive rows, which give genuine multi-season (2001-2026) ground
truth for "did this normally-starting player actually play tonight." Run
through this project's full walk-forward ML + evaluate_hypothesis()
discipline, that feature came back an honest, harmless NULL --
`adopt_cautiously` (margin_corr +0.00082, total_corr +0.00008, both well
inside the 0.005 noise floor; ROI +0.23pp, inside its own stderr) -- see
decision_log.jsonl's "Hypothesis test: nba_starters_out_availability" entry
for the exact numbers. Per this project's own standing precedent
(research_schedule_density.py's docstring: "an adopt_cautiously label is a
floor, not a mandate"), a feature that measurably moves nothing on the fit
axis, while also being the single most expensive per-build computation in
sports/nba/ (a 118MB player-level file + a per-team sequential scan, versus
everything else in ML_FEATURE_COLS being cheap vectorized rolling stats),
is not adopted into ML_FEATURE_COLS. sports/nba/features.py is unchanged.

There is also a real, honest reason this would have been a bad live-scoring
wire even if the historical test HAD cleared the bar: the historical
feature is built from ex-post ground truth ("did they actually play"),
while the only live-available signal is ex-ante ESPN status ("Out" /
"Doubtful" / "Day-to-Day" announced before tip-off) -- a forecast, not a
certainty, and a structurally different measurement from what the model
would have been trained against. This module narrows that gap the same
way `core.espn_client.starter_out` already does for NFL: only "Out"/
"Doubtful" count as `likely_out` (a real near-certain non-participation
signal), never "Questionable"/"Day-To-Day" alone. That gap is moot here
only because the underlying hypothesis was never adopted -- flagged
anyway, honestly, per the task's explicit instruction to surface this
class of live/historical mismatch rather than paper over it (see MLB's
`market_fair_home_prob` precedent in sports/mlb/features.py for the same
kind of honest disclosure).

Scope, stated the same way as the QB flag and the probable-pitcher flag:
this is a CONTEXT FLAG meant to be shown alongside a pick, never a trained
model input. sports/nba/features.py's ML_FEATURE_COLS does not reference
anything in this module, and core/matchup.py's live-scoring path never
imports it -- there is nothing here for `_warned_missing_features` to ever
warn about, by design.

NBA-SHAPE NOTE: `core.espn_client.parse_key_position_status`/`starter_out`
assume NFL's roster shape (`athletes` grouped by position, each group
holding an `items` list) -- verified directly that NBA's roster endpoint
returns a FLAT list of athlete dicts under `athletes` instead (no `items`
nesting at all), so those two NFL-shaped helpers silently find nothing
against real NBA data and are not reused here. `_flatten_nba_roster` below
is the NBA-shaped equivalent; `core.espn_client.depth_chart_starter_id` and
the OUT_INJURY_STATUSES/QUESTIONABLE_INJURY_STATUSES/
CONCERNING_ROSTER_STATUSES constants ARE sport-agnostic and are reused
as-is.
"""

import time

from core import espn_client
from . import espn_teams

DEPTH_CHART_POSITIONS = ["pg", "sg", "sf", "pf", "c"]  # a real starting five


def _flatten_nba_roster(roster_json: dict) -> dict:
    """NBA's roster endpoint returns athlete dicts directly under `athletes`
    (verified: no NFL-style position-group/`items` nesting). Returns
    {athlete_id: {id, name, position, status, status_abbr, injuries}}."""
    out = {}
    for player in roster_json.get("athletes", []) or []:
        pid = player.get("id")
        if not pid:
            continue
        status = player.get("status") or {}
        out[pid] = {
            "id": pid,
            "name": player.get("displayName"),
            "position": (player.get("position") or {}).get("abbreviation"),
            "status": status.get("name"),
            "status_abbr": status.get("abbreviation"),
            "injuries": [
                {"status": inj.get("status"), "date": inj.get("date"),
                 "short_comment": inj.get("shortComment")}
                for inj in (player.get("injuries") or [])
            ],
        }
    return out


def _starter_status(athlete: dict) -> dict:
    """Same likely_out/questionable convention as
    `core.espn_client.starter_out` -- errs toward NOT flagging a merely-
    questionable player, to avoid crying wolf on the common case of a
    probable/questionable tag that rarely means a starter actually sits."""
    injury_statuses = {inj["status"] for inj in athlete["injuries"] if inj.get("status")}
    likely_out = (
        athlete["status"] in espn_client.CONCERNING_ROSTER_STATUSES
        or bool(injury_statuses & espn_client.OUT_INJURY_STATUSES)
    )
    questionable = bool(injury_statuses & espn_client.QUESTIONABLE_INJURY_STATUSES)
    return {**athlete, "likely_out": likely_out, "questionable": questionable}


def fetch_starting_five_statuses(team_id: str) -> dict:
    """Returns {position_abbr: status_dict | None} for one team's real,
    ESPN-depth-chart-designated starting five (pg/sg/sf/pf/c). None for a
    position ESPN's depth chart or roster doesn't resolve, rather than a
    fabricated value. Two calls (roster + depth chart) -- the depth chart
    alone doesn't carry injury status; the roster alone doesn't say who's
    actually starting."""
    roster = espn_client.fetch_team_roster("NBA", team_id)
    if not roster:
        return {}
    depth_chart = espn_client.fetch_team_depth_chart("NBA", team_id)
    if not depth_chart:
        return {}
    athletes_by_id = _flatten_nba_roster(roster)

    out = {}
    for pos in DEPTH_CHART_POSITIONS:
        starter_id = espn_client.depth_chart_starter_id(depth_chart, pos)
        athlete = athletes_by_id.get(starter_id) if starter_id else None
        out[pos] = _starter_status(athlete) if athlete else None
    return out


def fetch_all_starting_five_statuses(delay_s: float = 0.5) -> dict:
    """Returns {franchise: {position: status_dict | None}} for every NBA
    team. `delay_s` spaces requests out defensively, same reasoning as
    sports/nfl/injuries.py's fetch_all_qb_statuses -- ESPN hasn't shown
    rate-limiting on these endpoints, but there's no reason to hammer it."""
    out = {}
    team_ids = sorted(espn_teams.ESPN_TEAM_ID_TO_FRANCHISE.items(), key=lambda kv: kv[0])
    for i, (team_id, franchise) in enumerate(team_ids):
        if i > 0:
            time.sleep(delay_s)
        statuses = fetch_starting_five_statuses(team_id)
        if statuses:
            out[franchise] = statuses
    return out


def starters_out_count(statuses: dict) -> int:
    """Convenience rollup for display: how many of a team's real starting
    five (a `fetch_starting_five_statuses()` result) are flagged
    `likely_out` right now."""
    return sum(1 for s in statuses.values() if s and s.get("likely_out"))
