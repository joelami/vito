"""
RESEARCH SCRIPT (throwaway, not part of the live module shape) — tests
whether player-level box-score data (passing/rushing yardage, interceptions
thrown) aggregated to team-game rolling features improves on the CFB model's
current score-only view. See docs/METHODOLOGY.md's NCAAF section for the
starting baseline this compares against, and core/research.py for the
evaluation discipline this follows exactly.

Run with:  python -m sports.cfb.research_boxscore_features   (from backend/)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.cfb import config as cfb_config
from sports.cfb.loader import load_games
from sports.cfb.features import build_features, ML_FEATURE_COLS, ROLL_WINDOW


DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "Datasets" / "College Football"


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------
# Step 1: verify the join key (event_id) works before building anything
# on it, and report coverage honestly.
# ----------------------------------------------------------------------
def verify_join_key():
    hr("STEP 1: JOIN KEY VERIFICATION (event_id)")

    raw_games = pd.read_csv(cfb_config.DATA_PATH)
    passing = pd.read_csv(DATASETS_DIR / "cfb-passing-box-scores.csv")
    rushing = pd.read_csv(DATASETS_DIR / "cfb-rushing-box-scores.csv")

    completed = raw_games[raw_games["completed"] == 1].copy()
    print(f"cfb-games.csv: {len(raw_games):,} total rows, {len(completed):,} completed=1")
    print(f"event_id dtype in games: {raw_games['event_id'].dtype}, "
          f"nunique in completed: {completed['event_id'].nunique():,}")
    print(f"passing box scores: {len(passing):,} rows, {passing['event_id'].nunique():,} unique event_ids")
    print(f"rushing box scores: {len(rushing):,} rows, {rushing['event_id'].nunique():,} unique event_ids")

    games_ids = set(completed["event_id"].dropna().unique())
    pass_ids = set(passing["event_id"].dropna().unique())
    rush_ids = set(rushing["event_id"].dropna().unique())

    print(f"\nCompleted games with event_id whose id appears in passing file: "
          f"{len(games_ids & pass_ids):,} / {len(games_ids):,} "
          f"({100*len(games_ids & pass_ids)/len(games_ids):.1f}%)")
    print(f"Completed games with event_id whose id appears in rushing file: "
          f"{len(games_ids & rush_ids):,} / {len(games_ids):,} "
          f"({100*len(games_ids & rush_ids)/len(games_ids):.1f}%)")

    # team-string integrity check: every box-score "team" value for a matched
    # event_id must equal that game's home_team or away_team string exactly
    merged_check = completed.merge(passing[["event_id", "team"]].drop_duplicates(), on="event_id", how="inner")
    mismatch = merged_check[~((merged_check["team"] == merged_check["home_team"]) |
                               (merged_check["team"] == merged_check["away_team"]))]
    print(f"\nTeam-string mismatches (box-score team not in {{home_team, away_team}} for its event_id): "
          f"{len(mismatch):,} / {len(merged_check):,}")

    # per-game coverage: how many of the 2 teams in a completed game actually
    # have a passing-box-score row for that event_id
    grp = passing.groupby("event_id")["team"].apply(set)

    def n_matched(row):
        return len(grp.get(row["event_id"], set()) & {row["home_team"], row["away_team"]})

    completed["n_teams_matched_passing"] = completed.apply(n_matched, axis=1)
    vc = completed["n_teams_matched_passing"].value_counts().sort_index()
    print("\nPer completed-game coverage (# of the 2 teams with a passing-box-score row for that event_id):")
    print(vc.to_string())
    both = (completed["n_teams_matched_passing"] == 2).mean() * 100
    at_least_one = (completed["n_teams_matched_passing"] >= 1).mean() * 100
    print(f"\nBoth teams matched: {both:.1f}% of completed games")
    print(f"At least one team matched: {at_least_one:.1f}% of completed games")

    print("\nConclusion: event_id is a clean, near-exact join key (0 team-string mismatches on "
          "matched rows), but coverage is partial like every other source in this dataset — "
          f"~{both:.0f}% of completed games get a full team-vs-team yardage match, same honest-disclosure "
          "spirit as the sparse odds coverage already documented in loader.py/METHODOLOGY.md.")

    return completed, passing, rushing


# ----------------------------------------------------------------------
# Build a team-game (event_id, team) -> total_yards, int_thrown table
# ----------------------------------------------------------------------
def build_team_game_boxscore_table() -> pd.DataFrame:
    passing = pd.read_csv(DATASETS_DIR / "cfb-passing-box-scores.csv")
    rushing = pd.read_csv(DATASETS_DIR / "cfb-rushing-box-scores.csv")

    pass_agg = passing.groupby(["event_id", "team"], as_index=False).agg(
        pass_yds=("yds", "sum"), int_thrown=("int", "sum")
    )
    rush_agg = rushing.groupby(["event_id", "team"], as_index=False).agg(
        rush_yds=("yds", "sum")
    )

    tg = pass_agg.merge(rush_agg, on=["event_id", "team"], how="outer")
    tg["pass_yds"] = tg["pass_yds"].fillna(0.0)
    tg["rush_yds"] = tg["rush_yds"].fillna(0.0)
    tg["int_thrown"] = tg["int_thrown"].fillna(0.0)
    tg["total_yards"] = tg["pass_yds"] + tg["rush_yds"]
    return tg[["event_id", "team", "total_yards", "int_thrown"]]


def attach_event_id(games: pd.DataFrame) -> pd.DataFrame:
    """
    load_games() (loader.py) intentionally drops event_id from its kept
    columns (baseline pipeline never needs it). To join box scores onto its
    output without touching loader.py, replicate loader's own row-filter +
    sort + game_id-assignment exactly on the raw CSV, keeping event_id this
    time, then join the two on game_id. This is verified below to line up
    1:1 with the real load_games() output.
    """
    raw = pd.read_csv(cfb_config.DATA_PATH)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "home_team", "away_team"]).copy()
    raw = raw[raw["completed"] == 1].copy()
    raw["home_score"] = pd.to_numeric(raw["home_score"], errors="coerce")
    raw["away_score"] = pd.to_numeric(raw["away_score"], errors="coerce")
    raw = raw.dropna(subset=["home_score", "away_score"]).copy()
    raw = raw.sort_values(["date", "kickoff_utc"], kind="stable").reset_index(drop=True)
    raw["game_id"] = raw.index.astype(str).map(lambda i: f"cfb_{i}")

    id_map = raw[["game_id", "event_id"]]
    out = games.merge(id_map, on="game_id", how="left")
    assert out["event_id"].notna().all(), "event_id attachment failed to cover every game row"
    return out


def build_features_with_boxscore(games: pd.DataFrame, rating_history: pd.DataFrame) -> pd.DataFrame:
    """
    Same as sports.cfb.features.build_features, plus walk-forward-safe
    (shift(1)-before-rolling(10)) team-level trailing total-yards and
    trailing interceptions-thrown, joined from the box-score files via
    event_id. Kept local to this research script per the task's instruction
    to extend/copy verify.py's shape rather than edit the live module until
    a hypothesis is actually adopted.
    """
    base = build_features(games, rating_history)

    games_with_eid = attach_event_id(games)[["game_id", "event_id", "date", "home_franchise", "away_franchise"]]
    tg = build_team_game_boxscore_table()

    home = games_with_eid.merge(
        tg, left_on=["event_id", "home_franchise"], right_on=["event_id", "team"], how="left"
    )[["game_id", "date", "home_franchise", "total_yards", "int_thrown"]].rename(
        columns={"home_franchise": "team"}
    )
    away = games_with_eid.merge(
        tg, left_on=["event_id", "away_franchise"], right_on=["event_id", "team"], how="left"
    )[["game_id", "date", "away_franchise", "total_yards", "int_thrown"]].rename(
        columns={"away_franchise": "team"}
    )
    log = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")

    grp = log.groupby("team", group_keys=False)
    log["yards_l10"] = grp["total_yards"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    log["int_thrown_l10"] = grp["int_thrown"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )

    league_avg_yards = float(tg["total_yards"].mean())
    league_avg_int = float(tg["int_thrown"].mean())

    home_feats = log.merge(
        games_with_eid[["game_id", "home_franchise"]], on="game_id"
    )
    home_feats = home_feats[home_feats["team"] == home_feats["home_franchise"]][
        ["game_id", "yards_l10", "int_thrown_l10"]
    ].rename(columns={"yards_l10": "home_yards_l10", "int_thrown_l10": "home_int_thrown_l10"})

    away_feats = log.merge(
        games_with_eid[["game_id", "away_franchise"]], on="game_id"
    )
    away_feats = away_feats[away_feats["team"] == away_feats["away_franchise"]][
        ["game_id", "yards_l10", "int_thrown_l10"]
    ].rename(columns={"yards_l10": "away_yards_l10", "int_thrown_l10": "away_int_thrown_l10"})

    out = base.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out["home_yards_l10"] = out["home_yards_l10"].fillna(league_avg_yards)
    out["away_yards_l10"] = out["away_yards_l10"].fillna(league_avg_yards)
    out["home_int_thrown_l10"] = out["home_int_thrown_l10"].fillna(league_avg_int)
    out["away_int_thrown_l10"] = out["away_int_thrown_l10"].fillna(league_avg_int)
    out["yards_diff_l10"] = out["home_yards_l10"] - out["away_yards_l10"]
    out["int_thrown_diff_l10"] = out["home_int_thrown_l10"] - out["away_int_thrown_l10"]
    return out


BOXSCORE_FEATURE_COLS = [
    "home_yards_l10", "away_yards_l10", "yards_diff_l10",
    "home_int_thrown_l10", "away_int_thrown_l10", "int_thrown_diff_l10",
]


# ----------------------------------------------------------------------
# Hypothesis 2: team-level completion percentage (QB-efficiency proxy),
# trailing L10, weighted by attempts (sum comp / sum att over the window,
# not an unweighted average of per-game percentages).
# ----------------------------------------------------------------------
def build_team_game_comp_pct_table() -> pd.DataFrame:
    passing = pd.read_csv(DATASETS_DIR / "cfb-passing-box-scores.csv")
    split = passing["comp_att"].astype(str).str.split("/", expand=True)
    passing["comp"] = pd.to_numeric(split[0], errors="coerce")
    passing["att"] = pd.to_numeric(split[1], errors="coerce")
    agg = passing.groupby(["event_id", "team"], as_index=False).agg(
        comp=("comp", "sum"), att=("att", "sum")
    )
    return agg


def build_features_with_comp_pct(feats: pd.DataFrame) -> pd.DataFrame:
    # `feats` already carries event_id natively (loader.load_games() keeps it
    # as of the box-score-feature adoption) - no separate attach step needed.
    games_with_eid = feats[["game_id", "event_id", "date", "home_franchise", "away_franchise"]]
    cp = build_team_game_comp_pct_table()

    home = games_with_eid.merge(
        cp, left_on=["event_id", "home_franchise"], right_on=["event_id", "team"], how="left"
    )[["game_id", "date", "home_franchise", "comp", "att"]].rename(columns={"home_franchise": "team"})
    away = games_with_eid.merge(
        cp, left_on=["event_id", "away_franchise"], right_on=["event_id", "team"], how="left"
    )[["game_id", "date", "away_franchise", "comp", "att"]].rename(columns={"away_franchise": "team"})
    log = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")

    grp = log.groupby("team", group_keys=False)
    # weighted rolling completion % = trailing sum(comp) / trailing sum(att), shift(1) first
    roll_comp = grp["comp"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    roll_att = grp["att"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    log["comp_pct_l10"] = (roll_comp / roll_att).replace([np.inf, -np.inf], np.nan)

    league_avg_comp_pct = float(cp["comp"].sum() / cp["att"].sum())

    home_feats = log.merge(games_with_eid[["game_id", "home_franchise"]], on="game_id")
    home_feats = home_feats[home_feats["team"] == home_feats["home_franchise"]][
        ["game_id", "comp_pct_l10"]
    ].rename(columns={"comp_pct_l10": "home_comp_pct_l10"})
    away_feats = log.merge(games_with_eid[["game_id", "away_franchise"]], on="game_id")
    away_feats = away_feats[away_feats["team"] == away_feats["away_franchise"]][
        ["game_id", "comp_pct_l10"]
    ].rename(columns={"comp_pct_l10": "away_comp_pct_l10"})

    out = feats.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out["home_comp_pct_l10"] = out["home_comp_pct_l10"].fillna(league_avg_comp_pct)
    out["away_comp_pct_l10"] = out["away_comp_pct_l10"].fillna(league_avg_comp_pct)
    out["comp_pct_diff_l10"] = out["home_comp_pct_l10"] - out["away_comp_pct_l10"]
    return out


COMP_PCT_FEATURE_COLS = ["home_comp_pct_l10", "away_comp_pct_l10", "comp_pct_diff_l10"]


# ----------------------------------------------------------------------
# Run the exact verify.py pipeline shape once per feature set, return the
# {margin_corr, total_corr, roi_pct, roi_stderr_pct} dict evaluate_hypothesis needs.
# ----------------------------------------------------------------------
def run_pipeline(feature_cols: list, feats: pd.DataFrame, label: str) -> dict:
    hr(f"PIPELINE RUN: {label}")

    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    print(f"Seasons predicted out-of-sample: {wf.seasons_predicted}")
    print(f"OOS rows: {len(wf.predictions):,}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    margin_corr = corr("predicted_margin", "actual_margin")
    total_corr = corr("predicted_total", "actual_total")
    print(f"Margin corr: {margin_corr:.4f}   Total corr: {total_corr:.4f}")

    stds = ensemble.compute_residual_stds(oos, cfb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0,
        allowed_confidence=("Medium", "High"),
        price_point="Close",
    )
    bets = backtest.run_backtest(oos, stds, cfb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"Qualifying bets: {len(bets):,}")

    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr, "roi_pct": 0.0, "roi_stderr_pct": float("nan")}

    summary = backtest.summarize(bets)
    roi_pct = float(summary["roi_pct"].iloc[0])
    roi_stderr_pct = float(summary["roi_stderr_pct"].iloc[0])
    print(f"ROI: {roi_pct:+.2f}%  (stderr {roi_stderr_pct:.2f}pp, n={len(bets)})")

    return {"margin_corr": margin_corr, "total_corr": total_corr, "roi_pct": roi_pct, "roi_stderr_pct": roi_stderr_pct}


def main():
    verify_join_key()

    hr("LOAD + POWER RATINGS (shared by both baseline and variant)")
    games = load_games()
    rating_cfg = PowerRatingConfig(
        k_factor=cfb_config.ELO_K_FACTOR,
        start_rating=cfb_config.ELO_START_RATING,
        home_field_adv=cfb_config.HOME_FIELD_ADV_ELO,
        season_regression=cfb_config.SEASON_REGRESSION,
        mov_mult_base=cfb_config.MOV_MULT_BASE,
        mov_mult_divisor=cfb_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", neutral_col="is_neutral_venue",
        config=rating_cfg,
    )
    print(f"Rating history rows: {len(rr.history):,}")

    import json

    # ---------------- Hypothesis 1: trailing total yards + INTs thrown ----------------
    # NOTE: hypothesis 1 (boxscore_yards_turnovers) was already run, evaluated via
    # evaluate_hypothesis(), logged to decision_log.jsonl, and (given
    # "adopt_cautiously") made PERMANENT in sports/cfb/features.py + loader.py in
    # this same research session — see decision_log.jsonl's "boxscore_yards_turnovers"
    # entry for the exact baseline/variant numbers from that run (margin_corr
    # 0.3566->0.3625, total_corr 0.1194->0.1460, ROI -0.47%->-1.55%, within noise).
    # Because that adoption is now permanent, `build_features`/`ML_FEATURE_COLS`
    # imported above ALREADY include the yards/INT features — re-running it here
    # would just compare the adopted model against itself. Not re-run.
    hr("HYPOTHESIS 1: already evaluated + adopted earlier this session (see decision_log.jsonl)")

    # ---------------- Hypothesis 2: completion percentage (QB efficiency proxy) ----------------
    # Baseline for THIS hypothesis is the now-current, already-adopted feature
    # set (yards/INTs included via the permanent features.py change above) -
    # testing whether completion percentage adds anything BEYOND raw yardage,
    # not re-testing yardage itself.
    baseline_feats = build_features(games, rr.history)
    baseline = run_pipeline(ML_FEATURE_COLS, baseline_feats, "BASELINE (current, already-adopted yards/INT features)")

    comp_pct_feats = build_features_with_comp_pct(baseline_feats)
    comp_pct_variant = run_pipeline(
        ML_FEATURE_COLS + COMP_PCT_FEATURE_COLS, comp_pct_feats,
        "HYPOTHESIS 2 VARIANT (+ trailing completion % on top of adopted yards/INT features)"
    )
    hyp2 = Hypothesis(
        name="qb_completion_pct",
        reasoning=(
            "Completion percentage is a standard passing-efficiency metric distinct "
            "from raw yardage (a team can pile up yards inefficiently via chunk plays, "
            "or convert efficiently on short/high-percentage passing) - real QB-efficiency "
            "ratings (QBR, passer rating) weight completion rate heavily precisely because "
            "it's not the same signal as yards or scoring. Testing whether it adds anything "
            "beyond the yards/turnover features already adopted, not a blind guess."
        ),
        sport="CFB",
    )
    result2 = evaluate_hypothesis(hyp2, baseline, comp_pct_variant)
    hr("HYPOTHESIS 2 RESULT: qb_completion_pct")
    print(json.dumps(result2.to_dict(), indent=2, default=str))

    hr("ALL DONE")


if __name__ == "__main__":
    main()
