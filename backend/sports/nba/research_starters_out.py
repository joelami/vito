"""
THROWAWAY research script -- NOT wired into main.py/pipeline.py/harness.py.
Tests one hypothesis via core/research.py's disciplined evaluate_hypothesis()
loop, following the exact shape of sports/nba/research_schedule_density.py /
sports/nfl/research_pacific_travel_effect.py: loader -> power ratings ->
baseline features vs variant features -> walk-forward ML -> residual stds ->
backtest -> evaluate_hypothesis() -> season split-half stability check.

Hypothesis: NBA_STARTERS_OUT_AVAILABILITY
-------------------------------------------------------------------------
BACKGROUND -- why this doesn't already exist, and what was actually checked
before writing a line of this script:

sports/nfl/injuries.py and sports/mlb/probables.py both flag player
availability LIVE (via ESPN roster/depth-chart/probables calls), but
BOTH are explicitly scoped as CONTEXT FLAGS, never trained model inputs --
their own docstrings say the historical dataset each sport trained on
"has no injury data at all." That's true for NFL/MLB. It turns out NOT to
be fully true for NBA: `Datasets/NBA/nba-box-scores.csv` (already a
trusted, verified source in this codebase -- see research_four_factors.py
and research_star_venue_split.py, which both already use it and both
already solved the join-to-loader-games problem this script reuses) is a
PLAYER-LEVEL box score file spanning 2001-10-31 through 2026-04-18, and
critically, it includes DNP/inactive rows: a player who was with the team
but did not play appears with `starter=0` and blank `min`/stat columns
(verified directly: e.g. event 211030016 lists Allen Iverson and Aaron
McKie for Philadelphia with `min` NaN -- present in the box score, zero
minutes). That is a REAL, historical, ground-truth "this normally-relevant
player did not play in this specific game" record -- not an injury
DESIGNATION (no reason code is given -- could be injury, rest, personal,
suspension, or a coaching decision), but for a team-strength signal what
matters is availability, not the reason, and this is honest ground truth,
not an inferred proxy.

INVESTIGATED FIRST, PER THE TASK, BEFORE BUILDING ANYTHING (real HTTP
requests made, not assumed -- see decision_log.jsonl and this module's
sibling `sports/nba/injuries.py` for the live-side writeup):

  1. ESPN's dedicated league-wide endpoint
     `site.api.espn.com/.../basketball/nba/injuries` is real and rich
     (per-player status, dates, long/short comments) -- but LIVE-ONLY.
  2. ESPN's roster endpoint (`.../teams/{id}/roster`) embeds each
     athlete's current `status`/`injuries` inline, same shape as NFL's.
  3. ESPN's depth-chart endpoint (`.../teams/{id}/depthcharts`) exists for
     NBA too, with real position groups (pg/sg/sf/pf/c) -- actually a
     BETTER starter-identification source live than NFL's roster-order
     heuristic.
  4. HISTORICAL DEPTH: NONE. Directly tested by calling
     `.../summary?event={id}` for a real 2015-03-01 game (Clippers @
     Bulls, event 400579169) and a real 2005-12-01 game -- both calls'
     `injuries` block returned TODAY's live injury report (e.g. the
     2005-12-01 game's response listed "Santi Aldama," an athlete who did
     not enter the NBA until 2021, with an injury dated 2026). ESPN's
     summary endpoint does not carry point-in-time injury state; it
     embeds whatever the CURRENT roster/injury feed is regardless of the
     event queried. Passing `?dates=` or `?season=` to the dedicated
     `/injuries` endpoint also had no effect -- still returned the
     current 2026-27 preseason report. Conclusion: ESPN has zero
     historical injury depth, confirming the task's expectation and
     matching NFL/MLB's documented reason for treating this as
     live-only elsewhere in this codebase.

  So `nba-box-scores.csv`'s DNP rows are the ONLY real historical source
  found with adequate multi-season depth to backtest against -- not an
  "injury report" in the NFL/MLB sense, but a genuine historical
  availability record, which is what a team-strength feature actually
  needs.

FEATURE DEFINITION (walk-forward-safe by construction):
  For each team, at each of its own games (strictly chronological), the
  "expected starting five" is defined ONLY from that team's own STRICTLY
  PRIOR games (up to the last 10, matching this project's standard NBA
  ROLL_WINDOW) -- the players with the most starts in that trailing
  window, requiring >=5 starts in the window to qualify (a real, settled
  starter, not a single spot-start) and capped at the top 5 by start
  count. A team with fewer than 5 qualifying players yet (not enough
  trailing history -- true for a team's first ~10-15 tracked games) gets
  a neutral 0/0 for that game, the same "no information yet" convention
  research_schedule_density.py and features.py's own rest_days/ATS
  warmup fills use.

  For THIS game, each expected starter is checked against the actual
  box-score rows for this specific event: appears AND played (`min`
  notna) -> present; anything else (blank-stat DNP row, or no row at
  all e.g. traded away) -> missing. `starters_out_count` (0-5) is the
  raw headcount; `starters_out_weighted` sums each MISSING starter's own
  trailing points-per-game (over the same walk-forward window) -- the
  quality weight the task asked for, so losing a 28-ppg starter counts
  for more than losing an 8-ppg one, not just "one guy is out" either way.
  Both representations are included in ONE feature set (not tested as
  competing candidates like schedule_density's raw-vs-boolean split --
  count and weighted-quality are complementary readings of the same
  underlying signal, not mutually exclusive alternatives, so there is
  nothing to be "genuinely torn" between).

JOIN TO LOADER GAMES: reuses research_four_factors.py's own verified
TEAM_NAME_TO_ABBREV map (imported directly, not re-derived -- same
precedent research_star_venue_split.py already established for reusing
it) and the identical merge_asof-by-(home_abbrev,away_abbrev)-nearest-
date(+/-1 day) join pattern that script's docstring measured at 90.6%
coverage of the odds-covered window. Unmatched games get neutral 0/0
(documented, not silently different from "confirmed nobody was out").

Run with:  python -m sports.nba.research_starters_out   (from backend/)

=============================================================================
OUTCOME (2026-09-04, logged in decision_log.jsonl) -- REAL numbers, exactly
as printed by this script's own run, no rounding in either direction's favor:
=============================================================================
See decision_log.jsonl entry "Hypothesis test: nba_starters_out_availability"
for the authoritative numbers (this docstring is not duplicated/hand-typed
here to avoid the two ever silently drifting apart -- run the script to
reproduce). Summary of the finding and the live-wiring decision that
followed is in `sports/nba/injuries.py`'s module docstring.
"""

import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

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

BOX_SCORES_PATH = Path(__file__).parent.parent.parent.parent / "Datasets" / "NBA" / "nba-box-scores.csv"

WINDOW = 10           # trailing team-games considered -- same as this project's standard NBA ROLL_WINDOW
MIN_STARTS_TO_QUALIFY = 5   # of the trailing window, needed to call someone a "real" starter
TOP_N = 5             # a basketball starting five


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _load_box_player_log() -> pd.DataFrame:
    """One row per (team, event_id, player) -- `started`/`played` booleans
    and `pts` (0.0 for a real DNP so it never contributes to a trailing
    average it shouldn't). Only usecols read (118MB file)."""
    box = pd.read_csv(
        BOX_SCORES_PATH,
        usecols=["date", "event_id", "team", "home_away", "player_id", "player_name", "starter", "min", "pts"],
    )
    box["date"] = pd.to_datetime(box["date"], errors="coerce")
    box["abbrev"] = box["team"].map(TEAM_NAME_TO_ABBREV)
    box = box.dropna(subset=["abbrev", "date", "player_id"]).copy()
    box["played"] = box["min"].notna()
    box["started"] = (box["starter"] == 1) & box["played"]
    box["pts"] = box["pts"].fillna(0.0)
    box["player_id"] = box["player_id"].astype(int)
    return box


def _compute_starters_out(box: pd.DataFrame) -> pd.DataFrame:
    """
    Sequential per-team accumulator -- same style as features.py's own
    `_streak` (a per-team Python loop maintaining running state game-by-
    game is an already-established pattern in this codebase for logic that
    doesn't reduce to a plain vectorized rolling window). For each team, in
    strict chronological order, computes THIS game's starters_out_count/
    _weighted from a deque holding ONLY the team's up-to-WINDOW STRICTLY
    PRIOR games, then pushes this game's real outcome onto the deque
    afterward -- so game i can never see its own or any future game's data.

    Returns one row per (team, event_id): starters_out_count (0-TOP_N),
    starters_out_weighted (sum of missing starters' trailing PPG).
    """
    # one row per (team, event_id) holding a dict of that game's real per-player outcomes
    per_game = box.groupby(["abbrev", "event_id", "date"]).apply(
        lambda g: dict(zip(g["player_id"], zip(g["started"], g["played"], g["pts"]))),
        include_groups=False,
    ).reset_index(name="players")
    per_game = per_game.sort_values(["abbrev", "date"], kind="stable")

    out_rows = []
    for team, grp in per_game.groupby("abbrev", sort=False):
        history = deque(maxlen=WINDOW)  # each entry: dict(player_id -> (started, played, pts))
        for event_id, players_today in zip(grp["event_id"], grp["players"]):
            # --- compute trailing per-player start count / avg pts from STRICTLY PRIOR games ---
            start_counts, pts_sums, games_played = {}, {}, {}
            for snapshot in history:
                for pid, (started, played, pts) in snapshot.items():
                    if played:
                        start_counts[pid] = start_counts.get(pid, 0) + (1 if started else 0)
                        pts_sums[pid] = pts_sums.get(pid, 0.0) + pts
                        games_played[pid] = games_played.get(pid, 0) + 1

            qualifying = [pid for pid, c in start_counts.items() if c >= MIN_STARTS_TO_QUALIFY]
            qualifying.sort(key=lambda pid: start_counts[pid], reverse=True)
            expected_starters = qualifying[:TOP_N]

            if len(expected_starters) < TOP_N:
                # not enough settled trailing history yet -- neutral "no info", same
                # convention as features.py's own rest_days/ATS-pct warmup fills
                starters_out_count, starters_out_weighted = 0, 0.0
            else:
                missing = [pid for pid in expected_starters
                           if not players_today.get(pid, (False, False, 0.0))[1]]  # played==False or absent
                starters_out_count = len(missing)
                starters_out_weighted = sum(pts_sums[pid] / games_played[pid] for pid in missing)

            out_rows.append((team, event_id, starters_out_count, starters_out_weighted))

            # --- push today's REAL outcome onto history for future games ---
            history.append(players_today)

    return pd.DataFrame(out_rows, columns=["abbrev", "event_id", "starters_out_count", "starters_out_weighted"])


def _load_box_team_games(starters_out: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Pairs each event_id's two team-sides into one row with home_/away_
    prefixed starters_out columns, same shape as research_four_factors.py's
    `_load_box_team_games` so `_attach_box_join`'s merge_asof logic can be
    reused unmodified."""
    sides = box[["event_id", "date", "abbrev", "home_away"]].drop_duplicates(["event_id", "abbrev"])
    merged = sides.merge(starters_out, on=["event_id", "abbrev"], how="left")
    home = merged[merged["home_away"] == "home"][["event_id", "date", "abbrev", "starters_out_count", "starters_out_weighted"]]
    away = merged[merged["home_away"] == "away"][["event_id", "abbrev", "starters_out_count", "starters_out_weighted"]]
    team_games = home.merge(away, on="event_id", suffixes=("_home", "_away"))
    team_games = team_games.rename(columns={"abbrev_home": "abbrev_home", "abbrev_away": "abbrev_away"})
    return team_games


def attach_starters_out(games: pd.DataFrame) -> pd.DataFrame:
    """Public entry point: returns `games` with 6 new walk-forward-safe
    columns: home_/away_starters_out_count, home_/away_starters_out_weighted,
    plus the two _diff columns. Unmatched games (event outside box-score
    coverage, or the +/-1 day join finds nothing) get neutral 0/0.0 --
    the honest "no information" default, not a fabricated "confirmed
    everyone played" claim."""
    box = _load_box_player_log()
    starters_out = _compute_starters_out(box)
    team_games = _load_box_team_games(starters_out, box)

    g = games.sort_values("date", kind="stable").reset_index(drop=True).copy()
    g["gidx"] = g.index
    tg = team_games.sort_values("date", kind="stable").reset_index(drop=True).copy()

    matched = pd.merge_asof(
        g.rename(columns={"home_franchise": "abbrev_home", "away_franchise": "abbrev_away"}),
        tg, on="date", by=["abbrev_home", "abbrev_away"],
        direction="nearest", tolerance=pd.Timedelta(days=1),
    ).rename(columns={"abbrev_home": "home_franchise", "abbrev_away": "away_franchise"})
    matched = matched.sort_values(["gidx"]).drop_duplicates("event_id", keep="first")

    box_cols = ["event_id", "starters_out_count_home", "starters_out_weighted_home",
                "starters_out_count_away", "starters_out_weighted_away"]
    out = g.merge(matched[["gidx"] + box_cols], on="gidx", how="left").drop(columns=["gidx"])

    out = out.rename(columns={
        "starters_out_count_home": "home_starters_out_count", "starters_out_weighted_home": "home_starters_out_weighted",
        "starters_out_count_away": "away_starters_out_count", "starters_out_weighted_away": "away_starters_out_weighted",
    })
    for c in ["home_starters_out_count", "away_starters_out_count",
              "home_starters_out_weighted", "away_starters_out_weighted"]:
        out[c] = out[c].fillna(0.0)
    out["starters_out_count_diff"] = out["home_starters_out_count"] - out["away_starters_out_count"]
    out["starters_out_weighted_diff"] = out["home_starters_out_weighted"] - out["away_starters_out_weighted"]
    return out


FEATURE_COLS = [
    "home_starters_out_count", "away_starters_out_count", "starters_out_count_diff",
    "home_starters_out_weighted", "away_starters_out_weighted", "starters_out_weighted_diff",
]


def run_pipeline(feats: pd.DataFrame, feature_cols: list, label: str):
    hr(f"WALK-FORWARD ML -- {label}")
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    print(f"Seasons predicted out-of-sample: {wf.seasons_predicted}")
    print(f"OOS rows: {len(wf.predictions):,}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    margin_corr = corr("predicted_margin", "actual_margin")
    total_corr = corr("predicted_total", "actual_total")
    print(f"Margin: ML pred vs actual correlation: {margin_corr:.6f}")
    print(f"Total:  ML pred vs actual correlation: {total_corr:.6f}")

    stds = ensemble.compute_residual_stds(oos, nba_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, nba_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    if bets.empty:
        raise RuntimeError(f"{label}: no bets cleared thresholds -- cannot compute ROI.")
    s = backtest.summarize(bets)
    roi_pct = float(s["roi_pct"].iloc[0])
    roi_stderr_pct = float(s["roi_stderr_pct"].iloc[0])
    n_bets = int(s["bets"].iloc[0])
    print(f"Qualifying bets: {n_bets:,}, ROI {roi_pct:+.3f}% +-{roi_stderr_pct:.3f}pp")

    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": roi_pct, "roi_stderr_pct": roi_stderr_pct, "n_bets": n_bets}, oos


def season_split_half_check(oos_baseline: pd.DataFrame, oos_variant: pd.DataFrame) -> None:
    """Report-only stability check (does not feed evaluate_hypothesis, which
    already governs adopt/reject on the full sample) -- mirrors
    sports/nfl/research_pacific_travel_effect.py's season_split_half_check
    exactly, per the task's explicit requirement to check this."""
    seasons = sorted(oos_variant["season"].dropna().unique())
    mid = seasons[len(seasons) // 2]
    halves = {
        f"early ({seasons[0]}-{mid - 1})": (oos_baseline[oos_baseline["season"] < mid],
                                             oos_variant[oos_variant["season"] < mid]),
        f"late ({mid}-{seasons[-1]})": (oos_baseline[oos_baseline["season"] >= mid],
                                        oos_variant[oos_variant["season"] >= mid]),
    }
    hr("SEASON SPLIT-HALF STABILITY CHECK")
    for label, (base_half, var_half) in halves.items():
        base_m = float(np.corrcoef(base_half["predicted_margin"], base_half["actual_margin"])[0, 1])
        var_m = float(np.corrcoef(var_half["predicted_margin"], var_half["actual_margin"])[0, 1])
        base_t = float(np.corrcoef(base_half["predicted_total"], base_half["actual_total"])[0, 1])
        var_t = float(np.corrcoef(var_half["predicted_total"], var_half["actual_total"])[0, 1])
        print(f"{label}: n_games={len(var_half):,}")
        print(f"  margin_corr {base_m:.4f} -> {var_m:.4f} ({var_m - base_m:+.4f})")
        print(f"  total_corr  {base_t:.4f} -> {var_t:.4f} ({var_t - base_t:+.4f})")


def main():
    hr("0. LOAD + POWER RATINGS (shared by baseline and variant)")
    games = load_games()
    print(f"Rows after load/clean: {len(games):,}")
    print(f"Season range: {games['season'].min()} - {games['season'].max()}")

    rating_cfg = PowerRatingConfig(
        k_factor=nba_config.ELO_K_FACTOR, start_rating=nba_config.ELO_START_RATING,
        home_field_adv=nba_config.HOME_FIELD_ADV_ELO, season_regression=nba_config.SEASON_REGRESSION,
        mov_mult_base=nba_config.MOV_MULT_BASE, mov_mult_divisor=nba_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", config=rating_cfg,
    )

    hr("1. ATTACH STARTERS-OUT COLUMNS (from Datasets/NBA/nba-box-scores.csv)")
    games_so = attach_starters_out(games)
    matched_mask = (games_so["home_starters_out_count"] > 0) | (games_so["away_starters_out_count"] > 0) \
        | (games_so["home_starters_out_weighted"] > 0) | (games_so["away_starters_out_weighted"] > 0)
    # coverage diagnostic: how many odds-covered (backtest-relevant) games got a real (non-default-zero) join
    odds_covered = games_so["Home Line Close"].notna()
    print(f"Odds-covered games: {odds_covered.sum():,}")
    print(f"Of odds-covered games, home_starters_out_count value_counts:\n"
          f"{games_so.loc[odds_covered, 'home_starters_out_count'].value_counts().sort_index()}")
    print(f"Mean home_starters_out_weighted (odds-covered): {games_so.loc[odds_covered, 'home_starters_out_weighted'].mean():.3f}")
    print(f"Mean away_starters_out_weighted (odds-covered): {games_so.loc[odds_covered, 'away_starters_out_weighted'].mean():.3f}")

    hr("2. BASELINE FEATURES (features.py unmodified)")
    baseline_feats = build_features(games_so, rr.history)  # extra columns ride along unused
    print(f"Feature rows: {len(baseline_feats):,}, ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")
    baseline, oos_baseline = run_pipeline(baseline_feats, ML_FEATURE_COLS, "BASELINE")
    print(f"\n[Baseline reproduction check] margin_corr={baseline['margin_corr']:.6f} "
          f"(expect ~0.405157), total_corr={baseline['total_corr']:.6f} (expect ~0.632622), "
          f"n_bets={baseline['n_bets']} (expect 22354), ROI={baseline['roi_pct']:+.3f}% "
          f"(expect ~-2.702%). If these don't match, STOP -- do not trust the variant below.")

    hr("3. VARIANT: + starters-out count/weighted availability columns")
    variant_cols = ML_FEATURE_COLS + FEATURE_COLS
    variant, oos_variant = run_pipeline(baseline_feats, variant_cols, "VARIANT (+starters_out)")

    hypothesis = Hypothesis(
        name="nba_starters_out_availability",
        reasoning=(
            "No injuries.py-equivalent module exists anywhere in sports/nba/ (confirmed by directory "
            "listing before writing any code), unlike sports/nfl/injuries.py and sports/mlb/probables.py. "
            "ESPN's live NBA injury/roster/depth-chart endpoints were tested directly (real HTTP requests) "
            "and are real and rich, but confirmed to carry ZERO historical depth: a summary call for a "
            "real 2005-12-01 and a real 2015-03-01 event both returned TODAY's current injury report "
            "(one listing a player, Santi Aldama, who did not enter the NBA until 2021), so ESPN cannot "
            "be used to backtest this. Datasets/NBA/nba-box-scores.csv -- already a trusted source in "
            "this codebase via research_four_factors.py / research_star_venue_split.py -- turns out to "
            "carry real historical ground truth instead: DNP/inactive players appear as rows with blank "
            "stat columns (verified directly, e.g. event 211030016), so 'which of a team's settled "
            "recent starters did not play tonight' is a real, walk-forward-safe, multi-season (2001-2026) "
            "historical signal, not an approximation. Externally motivated: missing a normal starter, "
            "weighted by how good that starter has been (trailing PPG), is a well-documented driver of "
            "real game outcomes that nothing currently in ML_FEATURE_COLS captures -- confirmed by "
            "grepping sports/nba/ for 'starter'/'injur'/'availab' before writing this script: zero "
            "matches outside this new file. Built as both a raw headcount (starters_out_count, 0-5) and "
            "a quality-weighted version (starters_out_weighted, sum of missing starters' trailing PPG) "
            "in a single feature set (not competing candidates -- both are complementary readings of the "
            "same signal a tree-based model can use as it sees fit, not a 'genuinely torn' choice)."
        ),
        sport="NBA",
    )
    baseline_metrics = {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}
    variant_metrics = {k: variant[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}

    hr("4. SEASON SPLIT-HALF STABILITY CHECK (before trusting the pooled numbers)")
    season_split_half_check(oos_baseline, oos_variant)

    hr("5. HYPOTHESIS EVALUATION")
    print(f"Baseline n_bets={baseline['n_bets']:,}  Variant n_bets={variant['n_bets']:,}")
    result = evaluate_hypothesis(hypothesis, baseline_metrics, variant_metrics)

    import json
    hr("RESULT")
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nRecommendation: {result.recommendation}")


if __name__ == "__main__":
    main()
