"""
THROWAWAY research script — NOT part of the production pipeline, not
imported by anything else. Tests two hypotheses about currently-unused
columns in `Datasets/NHL/archive/nhl_data_plus.csv` (`shots`,
`power_play_goals`, `power_play_opportunities`, `giveaways`, `takeaways`),
following the discipline in `core/research.py`.

LEAKAGE CHECK (done first, printed below): `shots`, `power_play_goals`, etc.
are this-game's own final box-score stats, known only after the game ends —
confirmed directly: shots correlates 0.16 and power_play_goals correlates
0.44 with that SAME game's goals_for. Using a team's own current-game shot
count as a same-game feature would be lookahead leakage. The only safe use
is the same convention `pf_l10`/`pa_l10` already use: shift(1) each team's
own game log BEFORE rolling, so a game's feature row only reflects that
team's games strictly before it.

Run with:  python -m sports.nhl.research_shot_metrics   (from backend/)
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
from sports.nhl import config as nhl_config
from sports.nhl.loader import load_games
from sports.nhl.features import build_features, ML_FEATURE_COLS

ROLL_WINDOW = 10


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Step 1: leakage check
# ---------------------------------------------------------------------------
def leakage_check():
    hr("STEP 1: LEAKAGE CHECK ON shots / power_play_goals / power_play_opportunities")
    raw = pd.read_csv(nhl_config.DATA_PATH)
    for col in ["shots", "power_play_goals", "power_play_opportunities", "goals_for"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    sub = raw.dropna(subset=["shots", "goals_for"])
    corr_shots = float(np.corrcoef(sub["shots"], sub["goals_for"])[0, 1])
    sub2 = raw.dropna(subset=["power_play_goals", "goals_for"])
    corr_ppg = float(np.corrcoef(sub2["power_play_goals"], sub2["goals_for"])[0, 1])
    print(f"corr(shots, goals_for) WITHIN THE SAME GAME ROW: {corr_shots:.4f}")
    print(f"corr(power_play_goals, goals_for) WITHIN THE SAME GAME ROW: {corr_ppg:.4f}")
    print("Non-trivial same-game correlation confirms these are box-score outcomes of the game")
    print("being scored, not pre-game information (same as home_score/away_score). CONFIRMED:")
    print("must not be used as same-game features. Rolled, shift(1)-first trailing history of a")
    print("team's OWN prior games is the only safe use - identical convention to pf_l10/pa_l10.")
    print("Proceeding to build ONLY shift(1)-then-rolling trailing features below.")


# ---------------------------------------------------------------------------
# Attach raw box-score columns (shots, PP, giveaways, takeaways) onto the
# already-cleaned, pivoted `games` frame via `source_game_id`.
# ---------------------------------------------------------------------------
def _load_raw_box_scores() -> pd.DataFrame:
    raw = pd.read_csv(nhl_config.DATA_PATH)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).copy()
    for col in ["shots", "power_play_goals", "power_play_opportunities", "giveaways", "takeaways",
                "faceoff_win_pct"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    # CONTAMINATION-AWARE TREATMENT: literal 0.0 in faceoff_win_pct is a missing-
    # data sentinel, not a real "won 0% of faceoffs" result - verified directly:
    # when both sides of a game have non-zero faceoff_win_pct, they sum to ~100.0
    # (mean 99.9996, std 0.17, n=19,408); only 4/19,412 games have exactly one
    # side zero (negligible data glitches). Concentrated in 2010-2017 but present
    # in negligible amounts every year, so treating exact 0.0 as NaN uniformly
    # (not just in the contaminated years) is the correct, consistent rule.
    raw["faceoff_win_pct"] = raw["faceoff_win_pct"].replace(0.0, np.nan)
    raw = raw[raw["team_id"].isin(nhl_config.REAL_TEAM_IDS)].copy()
    raw["franchise_id"] = raw["team_id"].apply(nhl_config.canonical_team_id)
    return raw


def _attach_box_scores(games: pd.DataFrame) -> pd.DataFrame:
    raw = _load_raw_box_scores()
    cols = ["game_id", "franchise_id", "shots", "power_play_goals", "power_play_opportunities",
            "giveaways", "takeaways", "faceoff_win_pct"]
    home = raw[raw["is_home"] == 1][cols].rename(columns={
        "franchise_id": "home_team_id_chk", "shots": "home_shots",
        "power_play_goals": "home_pp_goals", "power_play_opportunities": "home_pp_opp",
        "giveaways": "home_giveaways", "takeaways": "home_takeaways",
        "faceoff_win_pct": "home_faceoff_pct",
    })
    away = raw[raw["is_home"] == 0][cols].rename(columns={
        "franchise_id": "away_team_id_chk", "shots": "away_shots",
        "power_play_goals": "away_pp_goals", "power_play_opportunities": "away_pp_opp",
        "giveaways": "away_giveaways", "takeaways": "away_takeaways",
        "faceoff_win_pct": "away_faceoff_pct",
    })
    wide = home.merge(away, on="game_id", how="inner")
    out = games.merge(wide, left_on="source_game_id", right_on="game_id", how="left",
                       suffixes=("", "_rawjoin"))

    n_missing = out["home_shots"].isna().sum()
    n_side_mismatch = ((out["home_team_id"] != out["home_team_id_chk"]) & out["home_team_id_chk"].notna()).sum()
    print(f"Box-score join: {len(out) - n_missing:,}/{len(out):,} games matched "
          f"({(1 - n_missing/len(out))*100:.1f}%); home-side identity mismatches: {n_side_mismatch}")
    return out


def _rolled_team_stat_log(games_raw: pd.DataFrame) -> pd.DataFrame:
    """Long team-game log with shot/PP/giveaway/takeaway columns, shift(1)-then-rolled per team."""
    cols = ["game_id", "date"]
    home = games_raw[cols + ["home_team_id", "away_team_id", "home_shots", "away_shots",
                              "home_pp_goals", "home_pp_opp", "home_giveaways", "home_takeaways",
                              "home_faceoff_pct"]].rename(columns={
        "home_team_id": "team", "away_team_id": "opponent",
        "home_shots": "shots_for", "away_shots": "shots_against",
        "home_pp_goals": "pp_goals", "home_pp_opp": "pp_opp",
        "home_giveaways": "giveaways", "home_takeaways": "takeaways",
        "home_faceoff_pct": "faceoff_win_pct",
    })
    home["is_home"] = True
    away = games_raw[cols + ["away_team_id", "home_team_id", "away_shots", "home_shots",
                              "away_pp_goals", "away_pp_opp", "away_giveaways", "away_takeaways",
                              "away_faceoff_pct"]].rename(columns={
        "away_team_id": "team", "home_team_id": "opponent",
        "away_shots": "shots_for", "home_shots": "shots_against",
        "away_pp_goals": "pp_goals", "away_pp_opp": "pp_opp",
        "away_giveaways": "giveaways", "away_takeaways": "takeaways",
        "away_faceoff_pct": "faceoff_win_pct",
    })
    away["is_home"] = False
    log = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")

    grp = log.groupby("team", group_keys=False)
    log["shot_diff"] = log["shots_for"] - log["shots_against"]
    log["shot_diff_l10"] = grp["shot_diff"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())

    # PP conversion rate: rolling SUM of goals over rolling SUM of opportunities
    # (shift(1) first), not a mean-of-per-game-ratios - avoids 0/0 blowups on
    # the ~1.7% of games with zero power-play opportunities and matches how
    # PP% is conventionally reported over a multi-game window.
    pp_goals_sum = grp["pp_goals"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    pp_opp_sum = grp["pp_opp"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    log["pp_pct_l10"] = pp_goals_sum / pp_opp_sum

    log["turnover_diff"] = log["takeaways"] - log["giveaways"]
    log["turnover_diff_l10"] = grp["turnover_diff"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )

    # Trailing faceoff win% - contamination already stripped to NaN in
    # _load_raw_box_scores. Rolling mean with skipna semantics (pandas
    # default) naturally excludes NaN games from the trailing average rather
    # than treating them as 0%.
    log["faceoff_pct_l10"] = grp["faceoff_win_pct"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    return log


def add_shot_pp_features(feats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    games_raw = _attach_box_scores(games)
    log = _rolled_team_stat_log(games_raw)

    league_avg_pp_pct = float(log["pp_goals"].sum() / log["pp_opp"].sum())
    print(f"League-average PP% (fallback fill value): {league_avg_pp_pct:.4f}")

    home_feats = log[log["is_home"]][["game_id", "shot_diff_l10", "pp_pct_l10"]].rename(columns={
        "shot_diff_l10": "home_shot_diff_l10", "pp_pct_l10": "home_pp_pct_l10",
    })
    away_feats = log[~log["is_home"]][["game_id", "shot_diff_l10", "pp_pct_l10"]].rename(columns={
        "shot_diff_l10": "away_shot_diff_l10", "pp_pct_l10": "away_pp_pct_l10",
    })

    out = feats.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out["home_shot_diff_l10"] = out["home_shot_diff_l10"].fillna(0.0)
    out["away_shot_diff_l10"] = out["away_shot_diff_l10"].fillna(0.0)
    out["home_pp_pct_l10"] = out["home_pp_pct_l10"].fillna(league_avg_pp_pct)
    out["away_pp_pct_l10"] = out["away_pp_pct_l10"].fillna(league_avg_pp_pct)
    return out


def add_faceoff_feature(feats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    games_raw = _attach_box_scores(games)
    log = _rolled_team_stat_log(games_raw)

    league_avg_faceoff_pct = float(log["faceoff_win_pct"].mean())  # NaN-safe, contamination already stripped
    print(f"League-average faceoff win% (fallback fill value, contamination-stripped): {league_avg_faceoff_pct:.4f}")

    home_feats = log[log["is_home"]][["game_id", "faceoff_pct_l10"]].rename(columns={
        "faceoff_pct_l10": "home_faceoff_pct_l10",
    })
    away_feats = log[~log["is_home"]][["game_id", "faceoff_pct_l10"]].rename(columns={
        "faceoff_pct_l10": "away_faceoff_pct_l10",
    })

    out = feats.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out["home_faceoff_pct_l10"] = out["home_faceoff_pct_l10"].fillna(league_avg_faceoff_pct)
    out["away_faceoff_pct_l10"] = out["away_faceoff_pct_l10"].fillna(league_avg_faceoff_pct)
    return out


def add_turnover_feature(feats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    games_raw = _attach_box_scores(games)
    log = _rolled_team_stat_log(games_raw)

    home_feats = log[log["is_home"]][["game_id", "turnover_diff_l10"]].rename(columns={
        "turnover_diff_l10": "home_turnover_diff_l10",
    })
    away_feats = log[~log["is_home"]][["game_id", "turnover_diff_l10"]].rename(columns={
        "turnover_diff_l10": "away_turnover_diff_l10",
    })

    out = feats.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out["home_turnover_diff_l10"] = out["home_turnover_diff_l10"].fillna(0.0)
    out["away_turnover_diff_l10"] = out["away_turnover_diff_l10"].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Shared pipeline runner - identical shape to verify.py
# ---------------------------------------------------------------------------
def run_pipeline(games: pd.DataFrame, rr_history: pd.DataFrame, feats: pd.DataFrame, feature_cols: list) -> dict:
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0,
        allowed_confidence=("Medium", "High"),
        price_point="Close",
    )
    bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "n_bets": 0}

    summary = backtest.summarize(bets)
    return {
        "margin_corr": margin_corr,
        "total_corr": total_corr,
        "roi_pct": float(summary["roi_pct"].iloc[0]),
        "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
        "n_bets": int(summary["bets"].iloc[0]),
    }


def main():
    leakage_check()

    hr("BUILDING BASE PIPELINE (loader -> power ratings -> features)")
    games = load_games()
    rating_cfg = PowerRatingConfig(
        k_factor=nhl_config.ELO_K_FACTOR,
        start_rating=nhl_config.ELO_START_RATING,
        home_field_adv=nhl_config.HOME_FIELD_ADV_ELO,
        season_regression=nhl_config.SEASON_REGRESSION,
        mov_mult_base=nhl_config.MOV_MULT_BASE,
        mov_mult_divisor=nhl_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_team_id", away_col="away_team_id",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        config=rating_cfg,
    )
    feats_base = build_features(games, rr.history)
    print(f"Base feature rows: {len(feats_base):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")

    # NOTE: `feats_base`/`ML_FEATURE_COLS` here already include HYPOTHESIS 1
    # (trailing shot differential + PP conversion rate) - it was tested,
    # found genuinely non-suspicious (margin_corr +0.0060, beyond
    # CORR_NOISE_FLOOR, ROI moved DOWN not up), and adopted permanently into
    # sports/nhl/features.py. HYPOTHESIS 2 (trailing turnover differential)
    # was tested and REJECTED - both margin_corr and total_corr moved beyond
    # the noise floor in the NEGATIVE direction (a genuine fit deterioration)
    # despite evaluate_hypothesis()'s raw label being "adopt_cautiously" (a
    # blind spot in that function's logic - it doesn't check fit_moved's
    # sign when roi_delta<0-and-within-noise). Both are preserved in
    # decision_log.jsonl. This baseline below is therefore the CURRENT,
    # already-adopted production state - the correct baseline for any next
    # hypothesis per the same discipline.
    hr("BASELINE RUN (current production ML_FEATURE_COLS, already includes adopted shot-diff/PP feature)")
    baseline = run_pipeline(games, rr.history, feats_base, ML_FEATURE_COLS)
    print(baseline)

    import json

    # -----------------------------------------------------------------
    # HYPOTHESIS 3: trailing faceoff win% (contamination-aware)
    # -----------------------------------------------------------------
    hr("HYPOTHESIS 3: adding trailing faceoff win% (literal 0.0 treated as missing, not real 0%)")
    feats_faceoff = add_faceoff_feature(feats_base, games)
    feature_cols_faceoff = ML_FEATURE_COLS + ["home_faceoff_pct_l10", "away_faceoff_pct_l10"]
    variant3 = run_pipeline(games, rr.history, feats_faceoff, feature_cols_faceoff)
    print(variant3)

    h3 = Hypothesis(
        name="nhl_trailing_faceoff_pct_contamination_aware",
        reasoning=(
            "Faceoff win% is a specifically-debated metric in published hockey analytics - Eric "
            "Tulsky/War-on-Ice-era research (c.2013-2015) directly tested whether faceoff win% "
            "predicts team success and found the effect much smaller than conventional hockey "
            "wisdom assumed, but not zero (immediate/near-immediate possession off a won draw is "
            "a real, if modest, process-level signal distinct from goals/shots/PP already in the "
            "model). Worth testing empirically on this dataset rather than assuming either the "
            "old conventional wisdom or the revisionist finding. Data-quality checked directly: "
            "literal 0.0 is a missing-data sentinel, not a real result (when both sides of a game "
            "are non-zero they sum to ~100.0, mean 99.9996/std 0.17 across 19,408 games; only 4 "
            "games have exactly one side zero) - treated as NaN before rolling, which correctly "
            "excludes contaminated games from the trailing average instead of averaging in false "
            "0% seasons."
        ),
        sport="NHL",
    )
    result3 = evaluate_hypothesis(h3, baseline, variant3)
    hr("HYPOTHESIS 3 RESULT (evaluate_hypothesis().to_dict())")
    print(json.dumps(result3.to_dict(), indent=2))

    hr("DONE")


if __name__ == "__main__":
    main()
