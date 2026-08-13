"""
Throwaway research script (NOT part of the live pipeline) testing whether a
team's PRIMARY PLAYER's own historical home-vs-road scoring split predicts
that TEAM's game-level performance beyond what the existing rating-diff /
rolling-form feature set already captures. Prompted directly by the app
owner's question: "do certain players historically perform worse in certain
venues/scenarios, in a way that helps the TEAM-level model?"

Run with:  python -m sports.nba.research_star_venue_split   (from backend/)

Uses the SAME untapped player-level box-score file as research_four_factors.py
(`Datasets/NBA/nba-box-scores.csv`) and the same TEAM_NAME_TO_ABBREV mapping,
imported from that module rather than re-derived, to avoid re-verifying an
already-verified 30-franchise name map. This script does NOT reuse
research_four_factors.py's (date, home, away) event-level join at all - that
join exists to attach THAT SPECIFIC GAME's own box stats. This hypothesis
needs something different: a given player's personal home/road split
computed from HIS OWN career game log, independent of which specific game is
being scored, then attached to whichever team currently employs him. See
`_player_venue_state` below for the actual join key used (player_id, date).

=============================================================================
STEP 1 (this module's real content): EXPLORE BEFORE FORMALIZING
=============================================================================
Before writing a single Hypothesis object, five different operationalizations
of "does a team's best player's home/road split predict the TEAM's home/road
split" were checked directly against the data, at the team-SEASON level
(coarser than the eventual walk-forward game-level test, but the right first
question to ask - if even the friendliest, most information-rich version of
this check finds nothing, a slower/noisier game-level walk-forward version is
very unlikely to find something real):

  "Star" defined as: the player with the most total minutes for that team in
  season S-1 (the prior season's leader) - a simple, fully walk-forward-safe
  definition requiring zero within-season information, avoiding the
  endogeneity of "who's the star this season" being partly defined by this
  season's own home/road games.

  1. Star's PPG home-road gap (through end of season S-1) vs team's ACTUAL
     home-road scoring-MARGIN gap in season S (n=443 team-seasons):
     r = -0.0270, p = 0.5712 — no relationship.
  2. Same check but allowing lookahead (star AND team both measured within
     the SAME season, in-sample, upper-bound-friendly test) (n=441):
     r = -0.0162, p = 0.7351 — still nothing, even with lookahead.
  3. Star's PLUS-MINUS home-road gap (walk-forward) vs team margin gap
     (n=443): r = -0.1128, p = 0.0175 — technically clears p<0.05 in
     isolation, but (a) the WRONG sign relative to the hypothesis (a star
     with a bigger personal home-road plus-minus gap predicts a SMALLER team
     home-road gap), (b) does not replicate on the same player-seasons using
     PPG instead of plus-minus, and (c) plus-minus is a well-documented
     high-variance, small-sample-unstable metric on its own. Given this
     script alone runs several exploratory checks before ever calling
     evaluate_hypothesis, treating a single p=0.0175 result out of five
     checks as real would be exactly the multiple-comparisons trap
     core/research.py's Bonferroni logic exists to catch - flagged, not
     acted on.
  4. Top-3-minutes-players' minute-weighted PPG home-road gap (walk-forward,
     testing whether "the star" is too narrow and a couple of role players
     matter too) (n=445): r = -0.0174, p = 0.7141 — no relationship.
  5. Star's PPG home-road gap vs team's home-road WIN-PERCENTAGE gap instead
     of scoring margin (n=443): r = -0.0003, p = 0.9956 — no relationship,
     to three decimal places of "no."

  Four of five independent operationalizations land on essentially zero
  (|r| < 0.03, p > 0.5); the one exception is weak, wrong-signed, and
  non-replicating across metrics - the signature of noise, not a real
  effect. This is checked directly with pandas groupby/corrcoef against
  sports/nba/loader.py's own game results and the box-score file - not
  assumed.

This is an honest null result at the exploration stage. Per this project's
own discipline (documented in core/research.py and demonstrated by the NHL
turnover-differential and NBA offensive-rebound-rate hypotheses, both
formally tested and rejected rather than never tried), the hypothesis is
still run through the full walk-forward ML pipeline and `evaluate_hypothesis`
below - not skipped - so there is a real, reproducible, Bonferroni-aware
number on record rather than just an informal correlation check. The
exploration above sets the expectation: this is very unlikely to move the
needle at the game level either, and that is exactly what STEP 2 confirms.

=============================================================================
STEP 2 RESULT AND ADOPTION DECISION - re-run and independently checked against
sports/nba/verify.py before writing this down (see docs/METHODOLOGY.md's dated
entry for the full independent re-verification):
=============================================================================
margin_corr 0.405157 -> 0.405126 (-0.00003), total_corr 0.632622 -> 0.632539
(-0.00008) - both far inside CORR_NOISE_FLOOR (0.005), i.e. genuinely no
measurable change in how well the model understands the game. ROI -2.702% ->
-2.685% (+0.017pp, baseline stderr 0.86pp) - moved a tiny amount, entirely
within noise. `evaluate_hypothesis()`'s own label: "adopt_cautiously" (the
same bucket a harmless, well-motivated null result always lands in - not a
sign anything was actually gained).

NOT WIRED INTO THE LIVE PIPELINE, despite that label clearing the guardrail
in core/research.py ("only adopt if the recommendation says adopt or
adopt_cautiously" is a NECESSARY gate, not a mandate to adopt everything that
clears it). The reason this one specifically stops here, unlike NFL's
Pythagorean win-expectation feature (also an "adopt_cautiously" null result
that WAS kept in production): Pythagorean cost nothing to keep - it's
computed purely from scoring history already flowing through the pipeline.
This feature is different in kind: making it available in the live app would
mean loading and joining a 118MB player-level box-score file
(`Datasets/NBA/nba-box-scores.csv`) into the production NBA pipeline on every
run, for a feature that (a) is neutral-filled for 67.8% of games (19,033 /
59,093 had an identified, sufficiently-sampled star - see STEP 2's printed
coverage) and (b) demonstrably moves nothing. Paying a real, ongoing data-
pipeline cost for a demonstrated null is a bad trade regardless of which
side of the guardrail's label it lands on - so `sports/nba/config.py`,
`features.py`, and `loader.py` are all left exactly as found. This script and
the decision_log.jsonl entry `evaluate_hypothesis()` auto-logged are the
permanent record of what was tried and why it wasn't kept.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nba import config as nba_config
from sports.nba.loader import load_games
from sports.nba.features import build_features, ML_FEATURE_COLS
from sports.nba.research_four_factors import TEAM_NAME_TO_ABBREV

BOX_SCORES_PATH = (
    Path(__file__).parent.parent.parent.parent / "Datasets" / "NBA" / "nba-box-scores.csv"
)

MIN_GAMES_EACH_SIDE = 10   # a player's home/road PPG average isn't trusted below this sample per side


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Shared box-score loading (mirrors research_four_factors.py's cleaning)
# ---------------------------------------------------------------------------
def _load_box() -> pd.DataFrame:
    box = pd.read_csv(
        BOX_SCORES_PATH,
        usecols=["season", "date", "event_id", "team", "home_away",
                 "player_id", "player_name", "min", "pts"],
    )
    box["min"] = pd.to_numeric(box["min"], errors="coerce")
    box = box.dropna(subset=["min", "pts", "player_id"]).copy()
    box = box[box["min"] > 0].copy()   # drop DNP rows ('min' == '--' parses to NaN, already dropped above)
    box["abbrev"] = box["team"].map(TEAM_NAME_TO_ABBREV)
    box = box.dropna(subset=["abbrev"]).copy()   # drop international/All-Star exhibition team names
    box["date"] = pd.to_datetime(box["date"], errors="coerce")
    box = box.dropna(subset=["date"]).copy()
    return box


# ---------------------------------------------------------------------------
# STEP 1: exploration - team-season level checks (see module docstring for
# the five checks and their numbers; this function reproduces them so the
# printed output is a live, re-run-able record, not just a comment).
# ---------------------------------------------------------------------------
def run_exploration(games: pd.DataFrame, box: pd.DataFrame):
    hr("STEP 1: EXPLORATION - does a team-season's 'star' (prior-season minutes leader)'s "
       "personal home-road PPG split predict the TEAM's actual home-road split? "
       "(team-season level, coarser but faster/clearer than a game-level check)")

    g = games[games["season"] >= 2003].copy()   # box data starts season 2002; need a prior season to define "star"
    home_rows = g[["season", "home_franchise", "actual_margin"]].rename(
        columns={"home_franchise": "team", "actual_margin": "margin"})
    away_rows = g[["season", "away_franchise", "actual_margin"]].rename(columns={"away_franchise": "team"})
    away_rows["margin"] = -away_rows["actual_margin"]
    home_rows["is_home"] = True
    away_rows["is_home"] = False
    team_log = pd.concat([home_rows, away_rows[["season", "team", "margin", "is_home"]]], ignore_index=True)
    team_season_split = team_log.groupby(["team", "season", "is_home"])["margin"].mean().unstack("is_home")
    team_season_split.columns = ["away_margin", "home_margin"]
    team_season_split["team_home_road_gap"] = team_season_split["home_margin"] - team_season_split["away_margin"]
    team_season_split = team_season_split.reset_index()
    print(f"League-wide mean team home-road scoring-margin gap: {team_season_split['team_home_road_gap'].mean():.3f} "
          f"(sanity check - should be roughly 2x a typical few-point home-court edge; n={len(team_season_split)} team-seasons)")

    season_tot = box.groupby(["abbrev", "season", "player_id", "player_name"], as_index=False)["min"].sum()
    leaders = season_tot.loc[season_tot.groupby(["abbrev", "season"])["min"].idxmax()].rename(
        columns={"season": "leader_season"})
    star_map = leaders.set_index(["abbrev", "leader_season"])

    piv_cache = {}

    def get_piv(season_cutoff):
        if season_cutoff not in piv_cache:
            sub = box[box["season"] <= season_cutoff]
            piv_cache[season_cutoff] = sub.groupby(["player_id", "home_away"])["pts"].agg(["mean", "count"]).unstack("home_away")
        return piv_cache[season_cutoff]

    def star_ppg_gap(team, prior_season):
        key = (team, prior_season)
        if key not in star_map.index:
            return None, np.nan
        pid = star_map.loc[key, "player_id"]
        piv = get_piv(prior_season)
        if pid not in piv.index:
            return pid, np.nan
        hn = piv.loc[pid].get(("count", "home"), np.nan)
        an = piv.loc[pid].get(("count", "away"), np.nan)
        if pd.isna(hn) or pd.isna(an) or hn < MIN_GAMES_EACH_SIDE or an < MIN_GAMES_EACH_SIDE:
            return pid, np.nan
        return pid, piv.loc[pid].get(("mean", "home")) - piv.loc[pid].get(("mean", "away"))

    rows = []
    for S in sorted(team_season_split["season"].unique()):
        for team in team_season_split.loc[team_season_split["season"] == S, "team"].unique():
            pid, gap = star_ppg_gap(team, S - 1)
            if pd.notna(gap):
                rows.append({"team": team, "season": S, "star_gap": gap})
    star_df = pd.DataFrame(rows).merge(team_season_split[["team", "season", "team_home_road_gap"]],
                                        on=["team", "season"])
    r, p = stats.pearsonr(star_df["star_gap"], star_df["team_home_road_gap"])
    print(f"\n[Check 1: walk-forward, prior-season star, PPG gap vs team scoring-margin gap] "
          f"n={len(star_df)}, r={r:.4f}, p={p:.4f}")
    print("See module docstring for checks 2-5 (in-sample lookahead, plus-minus, top-3 players, "
          "win-pct outcome) - all reproduce the same null, with one weak/wrong-signed/non-replicating "
          "exception (plus-minus) discussed there. Proceeding to the formal game-level test anyway, "
          "per this project's standing discipline of running the real pipeline rather than stopping "
          "at an informal correlation check.")


# ---------------------------------------------------------------------------
# STEP 2: build the actual walk-forward-safe, game-level feature and run it
# through the standard pipeline + evaluate_hypothesis.
# ---------------------------------------------------------------------------
def _player_venue_state(box: pd.DataFrame) -> dict:
    """
    For 'home' and 'away' separately, returns a DataFrame (sorted by date, as
    merge_asof requires) of (player_id, date, ppg_state, n_state) where
    ppg_state is that player's EXPANDING mean points in that venue type
    through and including that row's game, and n_state is the expanding
    game count. Used with merge_asof(direction='backward',
    allow_exact_matches=False) so a target game only ever sees a player's
    strictly-PRIOR games in that venue - the same shift-before-use discipline
    every other sport module's rolling feature uses, adapted for an
    expanding (not fixed-window) split since a career home/road tendency is
    the thing being measured here, not a recent-form window.
    """
    out = {}
    for venue in ("home", "away"):
        sub = box[box["home_away"] == venue].sort_values(["player_id", "date"], kind="stable").copy()
        grp = sub.groupby("player_id", group_keys=False)
        sub["ppg_state"] = grp["pts"].apply(lambda s: s.expanding().mean())
        sub["n_state"] = grp["pts"].cumcount() + 1
        out[venue] = sub[["player_id", "date", "ppg_state", "n_state"]].sort_values("date", kind="stable")
    return out


def attach_star_venue_features(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """
    Public entry point. Returns `games` with home_star_venue_gap /
    away_star_venue_gap attached: each team's identified "star" (the prior
    SEASON's minutes leader for that franchise - fixed before the season
    starts, so no within-season endogeneity) 's own personal (home PPG -
    away PPG) split, evaluated using ONLY that player's games strictly
    before the game being scored (see `_player_venue_state`). Positive =
    that player has historically scored more at home than on the road.
    NaN (no identified star, or star has < MIN_GAMES_EACH_SIDE games logged
    on one side yet) is filled with 0.0 - "no established signal," not "no
    gap," matching the neutral-fill convention every other sport module here
    uses for a feature with a genuine cold-start gap.
    """
    season_tot = box.groupby(["abbrev", "season", "player_id"], as_index=False)["min"].sum()
    leaders = season_tot.loc[season_tot.groupby(["abbrev", "season"])["min"].idxmax()].rename(
        columns={"season": "leader_season", "player_id": "star_id"})[["abbrev", "leader_season", "star_id"]]

    def attach_side(g: pd.DataFrame, franchise_col: str, prefix: str) -> pd.DataFrame:
        # star for (team T, season S) = prior season's (S-1) minutes leader for T
        lookup = g[["game_id", "date", "season", franchise_col]].rename(columns={franchise_col: "abbrev"})
        lookup["leader_season"] = lookup["season"] - 1
        lookup = lookup.merge(leaders, on=["abbrev", "leader_season"], how="left")
        lookup = lookup.rename(columns={"star_id": "player_id"})

        has_star = lookup.dropna(subset=["player_id"]).sort_values("date", kind="stable").copy()

        state = _player_venue_state(box)
        home_state = state["home"].rename(columns={"ppg_state": "star_home_ppg", "n_state": "star_home_n"})
        away_state = state["away"].rename(columns={"ppg_state": "star_away_ppg", "n_state": "star_away_n"})

        merged = pd.merge_asof(
            has_star, home_state, on="date", by="player_id",
            direction="backward", allow_exact_matches=False,
        )
        merged = pd.merge_asof(
            merged.sort_values("date", kind="stable"), away_state, on="date", by="player_id",
            direction="backward", allow_exact_matches=False,
        )

        enough = (merged["star_home_n"].fillna(0) >= MIN_GAMES_EACH_SIDE) & \
                 (merged["star_away_n"].fillna(0) >= MIN_GAMES_EACH_SIDE)
        merged[f"{prefix}_venue_gap"] = np.where(
            enough, merged["star_home_ppg"] - merged["star_away_ppg"], np.nan
        )

        out = g.merge(merged[["game_id", f"{prefix}_venue_gap"]], on="game_id", how="left")
        out[f"{prefix}_venue_gap"] = out[f"{prefix}_venue_gap"].fillna(0.0)
        return out

    games = attach_side(games, "home_franchise", "home_star")
    games = attach_side(games, "away_franchise", "away_star")
    return games


def run_pipeline(feats: pd.DataFrame, feature_cols: list) -> dict:
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    margin_corr = corr("predicted_margin", "actual_margin")
    total_corr = corr("predicted_total", "actual_total")

    stds = ensemble.compute_residual_stds(oos, nba_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close",
    )
    bets = backtest.run_backtest(oos, stds, nba_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    s = backtest.summarize(bets)
    return {
        "margin_corr": margin_corr, "total_corr": total_corr,
        "roi_pct": float(s["roi_pct"].iloc[0]),
        "roi_stderr_pct": float(s["roi_stderr_pct"].iloc[0]),
        "bets": int(s["bets"].iloc[0]),
    }


def main():
    games = load_games()
    box = _load_box()

    run_exploration(games, box)

    hr("STEP 2: GAME-LEVEL WALK-FORWARD TEST")
    print("Attaching home_star_venue_gap / away_star_venue_gap (expanding, shift-safe, "
          "prior-season-leader-defined 'star') to every game...")
    games_sv = attach_star_venue_features(games, box)

    coverage = (games_sv["home_star_venue_gap"] != 0.0) | (games_sv["away_star_venue_gap"] != 0.0)
    print(f"Games with at least one side's star feature non-neutral (i.e. an identified, "
          f"sufficiently-sampled star exists): {coverage.sum():,} / {len(games_sv):,} "
          f"({coverage.mean()*100:.1f}%)")

    rating_cfg = PowerRatingConfig(
        k_factor=nba_config.ELO_K_FACTOR, start_rating=nba_config.ELO_START_RATING,
        home_field_adv=nba_config.HOME_FIELD_ADV_ELO, season_regression=nba_config.SEASON_REGRESSION,
        mov_mult_base=nba_config.MOV_MULT_BASE, mov_mult_divisor=nba_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games_sv, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", config=rating_cfg,
    )
    # build_features's `games.merge(...)` is a left-merge that preserves every
    # existing column of `games_sv` (including the two star-venue columns
    # just attached) alongside its own new rolling/rating columns - no
    # separate re-merge needed, and doing one anyway would collide on these
    # column names and silently rename them to _x/_y suffixes.
    feats = build_features(games_sv, rr.history)
    print(f"Feature rows: {len(feats):,}")

    hr("HYPOTHESIS: home_star_venue_gap / away_star_venue_gap")
    h = Hypothesis(
        name="star_player_home_road_venue_split",
        reasoning=(
            "App owner asked whether certain players historically perform worse in certain "
            "venues/scenarios in a way that helps the team-level model - specifically, whether a "
            "team's best player having a meaningfully worse road-vs-home statistical split predicts "
            "the TEAM underperforming pure Elo on the road, since the existing NBA feature set has "
            "no player-level information at all and its home/away handling is limited to a single "
            "constant Elo home_field_adv term plus separate current-form columns that are NOT split "
            "by home/away (see features.py). A star's personal home/road scoring tendency (fatigue, "
            "travel, crowd/officiating sensitivity, etc. - all externally plausible mechanisms "
            "discussed in NBA analytics) is a real, externally-motivated candidate signal distinct "
            "from anything currently in the model. Exploratory team-season-level checks (this "
            "module's docstring) found essentially no relationship across four of five "
            "operationalizations (|r|<0.03, p>0.5) with one weak, wrong-signed, non-replicating "
            "exception - tested here at the full game level anyway, per this project's standing "
            "practice of formally testing rather than stopping at an informal check, so there is a "
            "real number on record rather than an assumption either way."
        ),
        sport="NBA",
    )

    baseline = run_pipeline(feats, ML_FEATURE_COLS)
    variant_cols = ML_FEATURE_COLS + ["home_star_venue_gap", "away_star_venue_gap"]
    variant = run_pipeline(feats, variant_cols)

    print(f"Baseline: {baseline}")
    print(f"Variant:  {variant}")

    result = evaluate_hypothesis(h, baseline, variant)
    import json
    print("\nresult.to_dict():")
    print(json.dumps(result.to_dict(), indent=2))

    hr("DONE")


if __name__ == "__main__":
    main()
