"""
Real, PLAYER-level trailing features for NFL, built from nflverse's free
play-by-play (see scripts/fetch_nflfastr_qb_starts.py -- STARTING QB is a
proxy there, not a real box-score flag; see that script's own docstring).
This is the first PLAYER-category signal this project has ever had (see
core/factor_taxonomy.py's REGISTRY, which found zero before this).

STATUS (checked 2026-09-15, see sports/nfl/research_qb_features.py and
decision_log.jsonl): tested via this project's standard Hypothesis
framework -- REJECTED, not wired into ML_FEATURE_COLS or live scoring.
The full-sample result looked like an "adopt" (total_corr +0.0072), but
a split-half stability check found margin_corr degrades in BOTH season
halves and total_corr FLIPS SIGN between them -- not a stable, real
effect, just aggregate noise. Kept as real, working, tested
infrastructure (this module, its fetch script, and its own unit tests)
because the underlying data and walk-forward-safety are sound and a
different formulation (longer/shorter trailing window, a real
depth-chart/injury-based signal instead of a start-continuity proxy) is
a legitimate future re-test, not a dead end -- just not this one, as
built.

DESIGN CONSTRAINT, worth stating up front because it shapes everything
below: for a genuinely not-yet-played game, nobody knows who will start
at QB (no live depth-chart/injury-report source exists in this codebase
-- see sports/cfb/features.py's extra_matchup_features() docstring for
the same class of honest gap). Every feature here is therefore phrased
ENTIRELY in terms of a team's own PAST, already-completed games -- never
"this game's actual starter," which would be unusable live even though
it's technically knowable for historical training rows. Two real,
live-computable signals result:

  qb_epa_trail: trailing mean of the team's own game-by-game proxy-
    starter's EPA/dropback over their last N games (walk-forward-safe,
    shift-then-roll like every other trailing feature here) -- a genuine
    player-level efficiency signal, distinct from nflfastr_team_game_epa's
    off_epa_per_play (which blends the QB with the whole offensive unit,
    including run plays the QB never touches). If the team changed QBs
    partway through that window, this becomes a blend across different
    players' games -- still a real, honest "recent production at the
    position" signal, not a broken one.

  qb_continuity_flag: 1 if the team's own most recent TWO PRIOR games
    (both already completed before the game being featured, in both
    training and live contexts) had the SAME proxy-starter, else 0. This
    measures roster stability heading into a game without needing to know
    or predict who starts THAT game -- a real, externally-motivated
    proxy for "is this team dealing with QB uncertainty" (injury/
    benching churn), which the sports-betting literature treats as a
    material factor QB-level continuity captures directly and team-level
    EPA does not.
"""

from pathlib import Path

import pandas as pd

from .nflfastr_features import TEAM_CODE_TO_FRANCHISE

DATA_PATH = Path(__file__).parent.parent.parent.parent / "Datasets" / "NFL" / "nflfastr_qb_starts.csv"

_qb_starts_cache = None


def load_qb_starts() -> pd.DataFrame:
    """Same "raise, don't silently no-op" contract as
    nflfastr_features.load_team_game_epa() -- see that function's
    docstring."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found -- run `python3 scripts/fetch_nflfastr_qb_starts.py` "
            f"from backend/ first (downloads real nflverse play-by-play, ~20 seasons)."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["gameday"])
    df["franchise"] = df["team"].map(TEAM_CODE_TO_FRANCHISE)
    unmapped = df[df["franchise"].isna()]["team"].unique()
    if len(unmapped):
        raise ValueError(f"nflverse team code(s) with no franchise mapping: {list(unmapped)} -- "
                          f"TEAM_CODE_TO_FRANCHISE (sports/nfl/nflfastr_features.py) needs an entry.")
    return df.sort_values(["franchise", "gameday"], kind="stable")


def build_qb_continuity_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Adds home_/away_ prefixed qb_epa_trail and qb_continuity_flag columns.
    Same contract as nflfastr_features.build_trailing_epa_features(): one
    row in, one row out, never drops/reorders `games`.
    """
    franchise_names = set(games["home_franchise"].unique()) | set(games["away_franchise"].unique())
    qb = load_qb_starts()
    qb = qb[qb["franchise"].isin(franchise_names)].copy()

    league_avg_epa = float(qb["qb_epa_per_dropback"].mean())

    qb["qb_epa_trail"] = qb.groupby("franchise")["qb_epa_per_dropback"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    # continuity: was the PRIOR game's starter the same as the game before
    # THAT one -- both strictly before the row being featured, so this is
    # knowable the same way live as it is historically (see this module's
    # own docstring for why "this game's actual starter" is never used).
    prior_starter = qb.groupby("franchise")["starting_qb_id"].shift(1)
    prior_prior_starter = qb.groupby("franchise")["starting_qb_id"].shift(2)
    qb["qb_continuity_flag"] = (prior_starter == prior_prior_starter).astype(float)
    # A team's first two games ever (or first two after this data's own
    # cold start) have no real prior-pair to compare -- neutral fallback,
    # not a fabricated "continuous" or "changed" claim either way.
    qb.loc[prior_starter.isna() | prior_prior_starter.isna(), "qb_continuity_flag"] = 0.5

    # Joined on (franchise, date) not game_id -- this data's own game_id is
    # nflverse's format ("2006_01_ATL_CAR"), not this project's own
    # sports/nfl/loader.py game_id, same real reason
    # nflfastr_features.build_trailing_epa_features() joins this way too.
    out = games.copy()
    for side, franchise_col in (("home", "home_franchise"), ("away", "away_franchise")):
        merged = out[[franchise_col, "date"]].merge(
            qb[["franchise", "gameday", "qb_epa_trail", "qb_continuity_flag"]],
            left_on=[franchise_col, "date"], right_on=["franchise", "gameday"], how="left")
        out[f"{side}_qb_epa_trail"] = merged["qb_epa_trail"].fillna(league_avg_epa).values
        out[f"{side}_qb_continuity_flag"] = merged["qb_continuity_flag"].fillna(0.5).values
    return out


def get_current_qb_form(franchise: str) -> dict:
    """
    Live-scoring counterpart, same real gap/convention as
    nflfastr_features.get_current_trailing_epa() -- see that function's
    docstring. Reads the team's real last-5-games proxy-starter EPA and
    the real continuity flag from their two most recent COMPLETED games
    -- both knowable for a genuinely upcoming game, per this module's own
    design-constraint discussion above.
    """
    global _qb_starts_cache
    if _qb_starts_cache is None:
        _qb_starts_cache = load_qb_starts()
    qb = _qb_starts_cache
    league_avg_epa = float(qb["qb_epa_per_dropback"].mean())

    team_rows = qb[qb["franchise"] == franchise].sort_values("gameday", kind="stable")
    if team_rows.empty:
        return {"qb_epa_trail": league_avg_epa, "qb_continuity_flag": 0.5}

    recent = team_rows.tail(5)
    epa_trail = float(recent["qb_epa_per_dropback"].mean())

    last_two = team_rows.tail(2)["starting_qb_id"].tolist()
    if len(last_two) < 2:
        continuity = 0.5
    else:
        continuity = 1.0 if last_two[0] == last_two[1] else 0.0
    return {"qb_epa_trail": epa_trail, "qb_continuity_flag": continuity}


def reset_cache() -> None:
    """Same real staleness bug as sports/nfl/nflfastr_features.py's
    reset_caches() -- see that function's docstring. _qb_starts_cache
    freezes after its first population and never picks up new seasons'
    data on its own; called every scheduled full run (scheduler.py's
    _run_full()) for consistency with the rest of this project's kept-
    but-not-yet-adopted infrastructure, even though this module isn't
    currently wired into ML_FEATURE_COLS."""
    global _qb_starts_cache
    _qb_starts_cache = None
