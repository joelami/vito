"""
Turns core/watchlist.py's stale-item NUDGE into something a human can
actually act on in one step, without changing what gets to happen
silently.

Real gap this closes (2026-09-06): stale_items() already runs daily (see
scheduler.py's _run_full()) and correctly, deliberately does NOT
auto-rerun the underlying research or auto-promote anything -- every real
adoption in this project has required a human looking at freshly-derived
numbers, and that's staying true here too. But the nudge it produces is
just a print() into harness stdout, which nobody watches proactively (see
this project's own "no alerting" gap) -- and even someone who DOES see it
has to go hunt down which research script under sports/*/research_*.py
corresponds to which watchlist entry name. This module is that mapping,
plus a function that actually RUNS the re-derivation for whichever items
are stale -- but the RESULT is exactly the same as a human manually typing
`python -m sports.nba.research_...` themselves: a fresh, honestly-logged
watchlist entry (or an upgrade to a real Hypothesis "adopt"/"reject" if
the effect finally clears the bar), never anything wired into live scoring
automatically. Triggering this (see main.py's /api/admin/recheck-watchlist)
is still a deliberate action a person takes, same as run-harness already is.
"""

import importlib

from core import watchlist

# watchlist entry `name` -> the module whose main() re-derives it from
# scratch and calls evaluate_subgroup_hypothesis()/evaluate_hypothesis()
# again (see each module for its own reasoning). Add an entry here
# whenever a new research script produces a "watch" verdict -- nothing
# else discovers this mapping automatically, by design (see this module's
# docstring: no silent auto-promotion means no silent auto-registration
# either, this is a real, reviewed line someone adds).
RECHECK_REGISTRY = {
    "nba_spread_favorite_underdog_subgroup": "sports.nba.research_spread_favorite_underdog_shrinkage",
    "nhl_starting_goalie_individual_save_pct_total_market": "sports.nhl.research_starting_goalie_save_pct",
}


def run_stale_rechecks(max_age_days: int = 30) -> list:
    """
    For every watchlist item stale by `max_age_days` AND present in
    RECHECK_REGISTRY, imports and re-runs its module's main() -- the exact
    same "re-derive from scratch, don't trust cached numbers" discipline
    every research script in this project already follows (see e.g.
    sports/nba/research_spread_favorite_underdog_shrinkage.py's own
    docstring). Returns a list of {"name", "status", "error"} so a caller
    (the admin endpoint) can report what happened without needing to
    parse decision_log.jsonl itself.

    A stale item NOT in RECHECK_REGISTRY is reported as "no_registered_
    recheck" rather than silently skipped -- the registry needing a
    manual entry is a real gap worth surfacing, not something to paper
    over.
    """
    results = []
    for e in watchlist.stale_items(max_age_days=max_age_days):
        name = e["name"]
        module_path = RECHECK_REGISTRY.get(name)
        if not module_path:
            results.append({"name": name, "status": "no_registered_recheck", "error": None})
            continue
        try:
            module = importlib.import_module(module_path)
            module.main()
            results.append({"name": name, "status": "rechecked", "error": None})
        except Exception as e2:
            results.append({"name": name, "status": "failed", "error": str(e2)})
    return results
