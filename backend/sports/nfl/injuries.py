"""
Starting-QB availability check across the league, via ESPN's roster
endpoint (`core.espn_client.starter_out`). Unlike weather, this is
deliberately NOT disk-cached across days — injury status is time-sensitive
(changes week to week, sometimes day to day), so every call fetches fresh.

Scope, stated plainly: this is a CONTEXT FLAG shown alongside a pick, not a
trained model input. The historical dataset this model was built on has no
injury data at all, so there's no way to have learned "how many points a
backup QB is worth" from real outcomes — baking a guessed adjustment into
the model's probability would be exactly the kind of unvalidated assumption
this project has tried to avoid elsewhere. Surfacing "the starting QB looks
questionable/out" as a visible warning next to a pick is honest; silently
shifting the model's probability by some made-up amount would not be.
"""

import time

from core import espn_client
from . import espn_teams


def fetch_all_qb_statuses(delay_s: float = 0.5) -> dict:
    """Returns {franchise: starter_out_dict} for every NFL team. `delay_s`
    spaces requests out defensively — ESPN hasn't shown rate-limiting on
    these endpoints so far this session, unlike Open-Meteo, but there's no
    reason to hammer it either. Two calls per team (roster + depth chart) —
    the depth chart alone doesn't carry injury status, and roster-listing
    order alone isn't reliable for identifying who the real starter is
    (verified directly: a real case had a backup listed first in roster
    order while the depth chart had it right), so both are needed."""
    out = {}
    team_ids = sorted(espn_teams.ESPN_TEAM_ID_TO_FRANCHISE.items(), key=lambda kv: kv[0])
    for i, (team_id, franchise) in enumerate(team_ids):
        if i > 0:
            time.sleep(delay_s)
        roster = espn_client.fetch_team_roster("NFL", team_id)
        if not roster:
            continue
        time.sleep(delay_s)
        depth_chart = espn_client.fetch_team_depth_chart("NFL", team_id)
        qb = espn_client.starter_out(roster, "QB", depth_chart_json=depth_chart)
        if qb:
            out[franchise] = qb
    return out
