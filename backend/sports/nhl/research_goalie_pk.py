"""
THROWAWAY research script — NOT part of the production pipeline, not
imported by anything else. Tests two NEW hypotheses about goaltending form
and penalty-kill percentage, following the discipline in `core/research.py`.
Deliberately does NOT repeat any of the three tests already logged for NHL
in decision_log.jsonl (nhl_trailing_shot_diff_and_pp_rate [adopted],
nhl_trailing_turnover_diff [rejected], nhl_trailing_faceoff_pct
[fit didn't move, sub-noise-floor]).

Both new features are built entirely from columns already present in the
pivoted `games` frame that `sports/nhl/loader.py` produces (home_shots,
away_shots, home_pp_goals/opportunities, away_pp_goals/opportunities,
home_score, away_score) — no new raw-file join needed, unlike
research_shot_metrics.py's giveaways/takeaways/faceoff features which
required re-reading the raw team-game CSV.

LEAKAGE CHECK: goals_against/shots_against and the opponent's PP
goals/opportunities are this-game's own final box-score outcomes (same
leakage family as shots/power_play_goals already documented in
research_shot_metrics.py) — never usable as same-game features. Both new
features below are shift(1)-then-rolled per team, identical convention to
every other trailing feature in features.py.

HYPOTHESIS 1: trailing team save percentage (goaltending form).
  A team's own goaltending performance over its last 10 games — real,
  extensively documented driver of hockey outcomes (goals allowed are
  heavily influenced by goaltending variance independent of the skater
  possession game already captured by shot_diff_l10 — a team can out-shoot
  its opponent and still lose to a hot goalie, or a cold one can blow a
  shot-differential edge). Distinct signal family from shot_diff_l10 (shot
  VOLUME/possession) and pp_pct_l10 (special-teams offense) — this is
  shot-stopping EFFICIENCY, nothing already in ML_FEATURE_COLS measures it.
  save_pct = 1 - goals_against/shots_against, rolled as
  sum(goals_against)/sum(shots_against) over the trailing 10 games
  (shift(1) first), matching the sum-of-ratio convention pp_pct_l10 already
  uses in features.py (avoids degenerate low-shots-against games skewing a
  mean-of-per-game-ratios).

HYPOTHESIS 2: trailing penalty-kill percentage (mirror of the already-
  adopted power-play conversion rate). If PP offense meaningfully improved
  margin fit (see features.py's ADOPTED docstring), PK defense is the other
  half of special teams and a standard, well-documented pairing in hockey
  analytics (PP% and PK% are reported together everywhere from NHL.com to
  broadcast graphics) — there is no a priori reason offense-side special
  teams would matter to the model while defense-side didn't, and nothing
  currently in ML_FEATURE_COLS captures a team's own penalty-killing form.
  Computed from the OPPONENT's PP goals/opportunities in each of a team's
  games (the opponent's power play = this team's kill situation), rolled
  the same sum-of-ratio way.

Run with:  python -m sports.nhl.research_goalie_pk   (from backend/)
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


def leakage_check(games: pd.DataFrame):
    hr("STEP 1: LEAKAGE CHECK ON goals_against/shots_against and opponent PP goals/opportunities")
    home_goals_against = games["away_score"]
    home_shots_against = games["away_shots"]
    valid = home_shots_against.notna() & home_goals_against.notna() & (home_shots_against > 0)
    corr_ga_shots_against = float(np.corrcoef(
        home_goals_against[valid], home_shots_against[valid])[0, 1])
    print(f"corr(home goals_against, home shots_against) WITHIN THE SAME GAME: {corr_ga_shots_against:.4f}")
    save_pct_this_game = 1 - home_goals_against[valid] / home_shots_against[valid]
    corr_sv_ga = float(np.corrcoef(save_pct_this_game, home_goals_against[valid])[0, 1])
    print(f"corr(this-game save%, this-game goals_against): {corr_sv_ga:.4f} (strongly negative, as expected -")
    print("  confirms these are box-score OUTCOMES of the game being scored, same leakage family as")
    print("  shots/power_play_goals already documented in research_shot_metrics.py. Must not be used")
    print("  as same-game features - shift(1)-then-rolled trailing history only.")

    away_pp_valid = games["away_pp_opportunities"].notna() & (games["away_pp_opportunities"] > 0)
    home_pk_this_game = 1 - games.loc[away_pp_valid, "away_pp_goals"] / games.loc[away_pp_valid, "away_pp_opportunities"]
    print(f"\nSanity: mean this-game home PK% (from opponent's PP): {home_pk_this_game.mean():.4f} "
          f"(n={away_pp_valid.sum():,}) — realistic NHL PK range (~78-82% league-wide).")
    print("Proceeding to build ONLY shift(1)-then-rolling trailing features below.")


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Long format, one row per (team, game): goals_against/shots_against (own net's workload)
    and the OPPONENT's PP goals/opportunities (this team's kill situations), for rolling."""
    cols = ["game_id", "date"]
    home = games[cols + ["home_team_id", "away_team_id", "away_score", "away_shots",
                          "away_pp_goals", "away_pp_opportunities"]].rename(columns={
        "home_team_id": "team", "away_team_id": "opponent",
        "away_score": "goals_against", "away_shots": "shots_against",
        "away_pp_goals": "opp_pp_goals_against_us", "away_pp_opportunities": "times_shorthanded",
    })
    home["is_home"] = True
    away = games[cols + ["away_team_id", "home_team_id", "home_score", "home_shots",
                          "home_pp_goals", "home_pp_opportunities"]].rename(columns={
        "away_team_id": "team", "home_team_id": "opponent",
        "home_score": "goals_against", "home_shots": "shots_against",
        "home_pp_goals": "opp_pp_goals_against_us", "home_pp_opportunities": "times_shorthanded",
    })
    away["is_home"] = False
    log = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")

    grp = log.groupby("team", group_keys=False)

    # HYPOTHESIS 1: trailing save% = 1 - sum(goals_against)/sum(shots_against), shift(1) first
    ga_sum = grp["goals_against"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    sa_sum = grp["shots_against"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    log["save_pct_l10"] = 1.0 - ga_sum / sa_sum

    # HYPOTHESIS 2: trailing PK% = 1 - sum(opponent PP goals against us)/sum(times shorthanded), shift(1) first
    pk_goals_sum = grp["opp_pp_goals_against_us"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    pk_opp_sum = grp["times_shorthanded"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    log["pk_pct_l10"] = 1.0 - pk_goals_sum / pk_opp_sum

    return log


def add_save_pct_feature(feats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    log = _team_game_log(games)
    league_avg_save_pct = 1.0 - float(log["goals_against"].sum() / log["shots_against"].sum())
    print(f"League-average save% (fallback fill value): {league_avg_save_pct:.4f}")

    home_feats = log[log["is_home"]][["game_id", "save_pct_l10"]].rename(
        columns={"save_pct_l10": "home_save_pct_l10"})
    away_feats = log[~log["is_home"]][["game_id", "save_pct_l10"]].rename(
        columns={"save_pct_l10": "away_save_pct_l10"})

    out = feats.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out["home_save_pct_l10"] = out["home_save_pct_l10"].fillna(league_avg_save_pct)
    out["away_save_pct_l10"] = out["away_save_pct_l10"].fillna(league_avg_save_pct)
    return out


def add_pk_pct_feature(feats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    log = _team_game_log(games)
    league_avg_pk_pct = 1.0 - float(log["opp_pp_goals_against_us"].sum() / log["times_shorthanded"].sum())
    print(f"League-average PK% (fallback fill value): {league_avg_pk_pct:.4f}")

    home_feats = log[log["is_home"]][["game_id", "pk_pct_l10"]].rename(
        columns={"pk_pct_l10": "home_pk_pct_l10"})
    away_feats = log[~log["is_home"]][["game_id", "pk_pct_l10"]].rename(
        columns={"pk_pct_l10": "away_pk_pct_l10"})

    out = feats.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out["home_pk_pct_l10"] = out["home_pk_pct_l10"].fillna(league_avg_pk_pct)
    out["away_pk_pct_l10"] = out["away_pk_pct_l10"].fillna(league_avg_pk_pct)
    return out


def run_pipeline(feats: pd.DataFrame, feature_cols: list) -> dict:
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
    hr("BUILDING BASE PIPELINE (loader -> power ratings -> features, current production state)")
    games = load_games()
    leakage_check(games)

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
    print(f"Base feature rows: {len(feats_base):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)} "
          f"(current production state, already includes adopted shot_diff_l10/pp_pct_l10)")

    hr("BASELINE RUN (current production ML_FEATURE_COLS)")
    baseline = run_pipeline(feats_base, ML_FEATURE_COLS)
    print(baseline)

    import json

    # -----------------------------------------------------------------
    # HYPOTHESIS 1: trailing team save percentage (goaltending form)
    # -----------------------------------------------------------------
    hr("HYPOTHESIS 1: adding trailing save percentage (goaltending form)")
    feats_sv = add_save_pct_feature(feats_base, games)
    feature_cols_sv = ML_FEATURE_COLS + ["home_save_pct_l10", "away_save_pct_l10"]
    variant1 = run_pipeline(feats_sv, feature_cols_sv)
    print(variant1)

    h1 = Hypothesis(
        name="nhl_trailing_save_pct",
        reasoning=(
            "A team's recent goaltending form (save percentage) is a real, extensively documented "
            "driver of hockey outcomes, distinct from the skater possession/shot-volume signal "
            "already adopted (shot_diff_l10) and from special-teams offense (pp_pct_l10) - none of "
            "the currently-adopted features measure shot-stopping EFFICIENCY. A team can win the "
            "shot-differential battle and still lose to a hot opposing goalie, or blow a favorable "
            "shot differential with a cold one - hockey analytics treats 'hot/cold goaltending' "
            "(driven by team-level save% over a recent window, in the absence of goalie-specific "
            "start data) as one of the sport's largest sources of short-term variance separate from "
            "the possession game. save_pct_l10 = 1 - sum(goals_against)/sum(shots_against) over the "
            "trailing 10 games, shift(1)'d first per team, same sum-of-ratio convention pp_pct_l10 "
            "already uses in features.py."
        ),
        sport="NHL",
    )
    result1 = evaluate_hypothesis(h1, baseline, variant1)
    hr("HYPOTHESIS 1 RESULT (evaluate_hypothesis().to_dict())")
    print(json.dumps(result1.to_dict(), indent=2))

    # -----------------------------------------------------------------
    # HYPOTHESIS 2: trailing penalty-kill percentage
    # -----------------------------------------------------------------
    hr("HYPOTHESIS 2: adding trailing penalty-kill percentage (defensive mirror of adopted PP%)")
    feats_pk = add_pk_pct_feature(feats_base, games)
    feature_cols_pk = ML_FEATURE_COLS + ["home_pk_pct_l10", "away_pk_pct_l10"]
    variant2 = run_pipeline(feats_pk, feature_cols_pk)
    print(variant2)

    h2 = Hypothesis(
        name="nhl_trailing_pk_pct",
        reasoning=(
            "Penalty-kill percentage is the defensive mirror of power-play conversion rate, which "
            "was already adopted into this model on a genuine margin-fit improvement (see features.py's "
            "ADOPTED docstring, nhl_trailing_shot_diff_and_pp_rate). PP% and PK% are reported together "
            "everywhere in hockey (NHL.com team stats, broadcast graphics) as the two halves of special "
            "teams performance, and there is no a priori reason offense-side special teams would carry "
            "signal for this model while defense-side special teams carried none - nothing currently in "
            "ML_FEATURE_COLS measures a team's own penalty-killing form. Computed from the OPPONENT's PP "
            "goals/opportunities in each of a team's games (the opponent's power play is this team's kill "
            "situation): pk_pct_l10 = 1 - sum(opponent PP goals against us)/sum(times shorthanded) over "
            "the trailing 10 games, shift(1)'d first per team, same sum-of-ratio convention as pp_pct_l10."
        ),
        sport="NHL",
    )
    result2 = evaluate_hypothesis(h2, baseline, variant2)
    hr("HYPOTHESIS 2 RESULT (evaluate_hypothesis().to_dict())")
    print(json.dumps(result2.to_dict(), indent=2))

    hr("DONE")


if __name__ == "__main__":
    main()
