"""
A real organizing structure for this project's growing feature set, adopted
2026-09-14 at the app owner's request after reviewing a factor-taxonomy
pattern from another product's docs (see decision_log.jsonl for the full
context -- that product's own "tune Low/Medium/High and eyeball it"
workflow is NOT being adopted, it's closer to the "keep tuning until it
looks good" anti-pattern this project's whole methodology exists to
prevent; the taxonomy ITSELF -- categorizing factors by what kind of
signal they carry -- is the genuinely useful, portable idea).

Four categories, matching real, meaningfully different sources of signal:

  TEAM_PERFORMANCE -- how good is this team on its own merits (rating,
    recent scoring, ATS record, Pythagorean win%, hot/cold form). The
    largest category in every sport here today.
  MATCHUP -- signals that only exist by comparing the two specific teams
    together (head-to-head history, pace/style clashes, naive combined
    scoring rate) -- not reducible to either team's own standalone stats.
  SITUATIONAL -- external, schedule/environment-driven factors neither
    team's own quality (rest, travel, weather, divisional/playoff/neutral-
    site context) -- the same team plays differently under different
    circumstances.
  PLAYER -- individual player availability/performance impact (starting
    QB quality, injuries to stars). See the real gap this surfaces below.

Real, honest use for this: `coverage_report()` below isn't decorative --
it's what showed, directly and for the first time, that NFL's live
ML_FEATURE_COLS has ZERO Player-category features despite QB status
already being tracked and displayed elsewhere in this app (sports/nfl/
injuries.py's fetch_all_qb_statuses(), shown on the Suggestions page) --
tracked for humans to read, never wired into what the model actually
predicts with. That's a real, concrete lead for the next hypothesis, not
a hypothetical one this taxonomy was built to go looking for.
"""

from dataclasses import dataclass

TEAM_PERFORMANCE = "team_performance"
MATCHUP = "matchup"
SITUATIONAL = "situational"
PLAYER = "player"
CATEGORIES = [TEAM_PERFORMANCE, MATCHUP, SITUATIONAL, PLAYER]


@dataclass
class Factor:
    name: str             # exact ML_FEATURE_COLS entry (or "home_X"/"away_X" pair's shared root)
    category: str
    note: str = ""         # one-line why-this-bucket, only when it's not obvious from the name


# Real registry, sport by sport -- hand-built by reading each sport's own
# features.py ML_FEATURE_COLS, not inferred/guessed. Add to this whenever
# a feature is adopted (see core/research.py's Hypothesis discipline --
# same rule: real reasoning, not a placeholder entry).
REGISTRY = {
    "NFL": [
        Factor("rating_diff_pre", TEAM_PERFORMANCE, "Elo rating differential -- the core team-strength signal"),
        Factor("home_ats_pct_l10", TEAM_PERFORMANCE), Factor("away_ats_pct_l10", TEAM_PERFORMANCE),
        Factor("home_win_pct_l10", TEAM_PERFORMANCE), Factor("away_win_pct_l10", TEAM_PERFORMANCE),
        Factor("home_pf_l10", TEAM_PERFORMANCE), Factor("home_pa_l10", TEAM_PERFORMANCE),
        Factor("away_pf_l10", TEAM_PERFORMANCE), Factor("away_pa_l10", TEAM_PERFORMANCE),
        Factor("home_pyth_pct", TEAM_PERFORMANCE), Factor("away_pyth_pct", TEAM_PERFORMANCE),
        Factor("pyth_pct_diff", TEAM_PERFORMANCE),
        Factor("home_streak", TEAM_PERFORMANCE, "recent form, not an external circumstance"),
        Factor("away_streak", TEAM_PERFORMANCE),
        Factor("naive_total", MATCHUP, "combines BOTH teams' scoring rates -- not either team's own stat alone"),
        Factor("naive_margin", MATCHUP),
        Factor("rest_diff", SITUATIONAL), Factor("home_rest_days", SITUATIONAL), Factor("away_rest_days", SITUATIONAL),
        Factor("is_divisional", SITUATIONAL), Factor("is_playoff", SITUATIONAL), Factor("is_neutral_venue", SITUATIONAL),
        Factor("game_temp_f", SITUATIONAL), Factor("game_wind_mph", SITUATIONAL),
        Factor("game_precip_mm", SITUATIONAL), Factor("is_dome", SITUATIONAL),
        # PLAYER: deliberately empty -- see this module's docstring. Real
        # gap, not an oversight: sports/nfl/injuries.py already tracks
        # real starting-QB status live, it's just never reached the model.
    ],
}


def coverage_report(sport: str) -> dict:
    """Real counts per category for a sport's registered factors, plus
    which categories have ZERO entries -- the honest, at-a-glance answer
    to 'where are we structurally thin' that motivated adopting this at
    all, not just a categorized list for its own sake."""
    factors = REGISTRY.get(sport, [])
    counts = {cat: 0 for cat in CATEGORIES}
    for f in factors:
        counts[f.category] += 1
    return {
        "sport": sport,
        "total_factors": len(factors),
        "by_category": counts,
        "empty_categories": [cat for cat, n in counts.items() if n == 0],
    }


def print_coverage(sport: str) -> None:
    r = coverage_report(sport)
    print(f"{sport}: {r['total_factors']} registered factors")
    for cat in CATEGORIES:
        print(f"  {cat}: {r['by_category'][cat]}")
    if r["empty_categories"]:
        print(f"  ZERO coverage: {', '.join(r['empty_categories'])} -- a real gap, not a design choice")
