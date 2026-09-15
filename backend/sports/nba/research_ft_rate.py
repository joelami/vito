"""
THROWAWAY research script -- NOT wired into main.py/pipeline.py/harness.py
unless/until this hypothesis clears the adopt bar (see bottom of this
docstring). Tests one hypothesis via core/research.py's disciplined
evaluate_hypothesis() loop, following the exact shape of
sports/nba/research_four_factors.py / sports/nba/research_starters_out.py:
loader -> power ratings -> baseline features vs variant features -> walk-
forward ML -> residual stds -> backtest -> evaluate_hypothesis() -> REQUIRED
season split-half stability check.

Hypothesis: NBA_FOUR_FACTORS_FT_RATE
-------------------------------------------------------------------------
Dean Oliver's "Four Factors" (eFG%, TOV%, OREB%, FT Rate) is the explicit,
externally-cited basketball-analytics framework this project has already
adopted 3 of 4 factors from (decision_log.jsonl: `four_factors_efg_tov` and
`four_factors_oreb_pct`, both `adopt_cautiously`). FT Rate -- the fourth
published factor, defined here exactly as Dean Oliver's own original
formula: FTA / FGA, "how good a team is at drawing fouls and getting to the
line relative to how many shots it takes" -- has never been tested,
confirmed directly by grepping decision_log.jsonl for `ft_rate`/`free_throw`
before writing this script: zero matches anywhere. Deliberately NOT
free-throw shooting PERCENTAGE (FTM/FTA) -- that's a separate skill (finishing
the chances once earned) outside Dean Oliver's own published factor, which is
specifically about the RATE of getting to the line, not what happens once
there.

SYMMETRIC DEFENSIVE COMPLEMENT (task's own explicit prompt: "decide based on
what's externally standard and what the existing Four Factors features
already do for symmetry -- likely also a defensive/opponent-FT-rate-allowed
side"): Dean Oliver's own published framework is explicitly two-sided --
real Four Factors reporting (e.g. Basketball-Reference's own team pages)
always shows a team's offensive Four Factors AND its opponents' Four Factors
(what the team ALLOWS) side by side, since "gets to the line a lot" and
"fouls a lot / lets opponents get to the line" are two different, both real,
mechanisms. This project's own already-adopted OREB% feature actually already
leans on the opposing side's box line internally (oreb / (oreb + opponent's
dreb)) even though it's framed as one offensive number -- so a genuinely
separate opponent-facing column for FT rate specifically (where the two
directions are NOT arithmetically linked the way OREB%/DREB% are) is a
natural, motivated completion, not scope creep. Tested here as ONE hypothesis
with both readings in a single feature set (own ft_rate_l10 AND
opp_ft_rate_allowed_l10) -- the same "complementary readings of one
mechanism, not competing candidates" reasoning already used for
nba_starters_out_availability's count+weighted pair, not a "genuinely torn"
split needing two separate tests.

DATA / JOIN: reuses research_four_factors.py's own verified box-score join
machinery directly (imported, not reimplemented) -- `_load_box_team_games`
already aggregates `fga`/`fta` (needed for FT rate) alongside the
eFG%/TOV%/OREB% inputs it was built for, and `_attach_box_join` already
solves the event_id-to-loader-game_id join (merge_asof, nearest date +/-1
day, one-to-one, ~90.6% coverage of the odds-covered window -- see that
module's own docstring for the full verification). No new join logic
needed or written.

FEATURE: walk-forward-safe (shift(1) before any rolling window, identical
discipline to every other trailing feature in this project) trailing-10-game
mean of each team's own FTA/FGA (offense) and of the OPPONENT's FTA/FGA in
each of that team's own games (defense/allowed) -- home_/away_ft_rate_l10,
home_/away_opp_ft_rate_allowed_l10, plus diffs (home minus away, matching
efg_diff/tov_diff/oreb_pct_diff's existing sign convention). League-average
fallback used for warmup/unmatched rows is COMPUTED from this project's own
real join output below (printed at runtime), not guessed.

BASELINE/VARIANT: per the task's explicit instruction, baseline is the
CURRENT PRODUCTION `ML_FEATURE_COLS` (features.py) -- eFG%/TOV%/OREB% were
never wired into production (adopt_cautiously, this project's established
precedent per sports/nba/injuries.py's own decision-log entry: "consistent
with this project's established precedent of not wiring in cautious nulls"),
so the honest baseline to test FT rate against is the actual live feature
set, not a hypothetical one including the other three factors.

REQUIRED SPLIT-HALF STABILITY CHECK: per this project's cautionary tale
(decision_log.jsonl's `nfl_qb_trailing_epa_and_continuity`, REJECTED -- a
full-sample evaluate_hypothesis() result that looked like "adopt" did NOT
hold up independently in both season halves and was correctly rejected
despite the full-sample number), this script's `season_split_half_check()`
is run and its verdict is treated as authoritative alongside (not instead
of) evaluate_hypothesis()'s own recommendation -- see main() for how the two
are combined into the final call. Mirrors
sports/nba/research_starters_out.py's own `season_split_half_check` shape,
extended with the explicit STABLE/NOT-STABLE verdict logic
sports/nfl/research_nflfastr_epa_sos_adjustment.py's equivalent check uses
(a degradation past CORR_NOISE_FLOOR in either half disqualifies the
result), since research_starters_out.py's version was report-only and this
task requires an explicit pass/fail gate.

Run with:  python -m sports.nba.research_ft_rate   (from backend/)
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
from core.research import Hypothesis, evaluate_hypothesis, CORR_NOISE_FLOOR
from sports.nba import config as nba_config
from sports.nba.loader import load_games
from sports.nba.features import build_features, ML_FEATURE_COLS
from sports.nba.research_four_factors import _load_box_team_games, _attach_box_join

ROLL_WINDOW = 10


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _ft_rate_team_log(games_with_box: pd.DataFrame) -> pd.DataFrame:
    """Long format, one row per (team, game) carrying that team's own FTA/FGA
    (offense) and its OPPONENT's FTA/FGA in that same game (defense/allowed)
    -- not yet rolled. Built the same shape as research_four_factors.py's
    `_box_team_log` so the rolling step below is a direct copy of the proven
    shift(1)-before-rolling pattern every trailing feature in this project
    uses."""
    cols = ["game_id", "date", "season"]

    home = games_with_box[cols + ["home_franchise", "fta_home", "fga_home", "fta_away", "fga_away"]].rename(
        columns={
            "home_franchise": "team", "fta_home": "own_fta", "fga_home": "own_fga",
            "fta_away": "opp_fta", "fga_away": "opp_fga",
        })
    home["is_home"] = True
    away = games_with_box[cols + ["away_franchise", "fta_away", "fga_away", "fta_home", "fga_home"]].rename(
        columns={
            "away_franchise": "team", "fta_away": "own_fta", "fga_away": "own_fga",
            "fta_home": "opp_fta", "fga_home": "opp_fga",
        })
    away["is_home"] = False
    log = pd.concat([home, away], ignore_index=True)

    # Dean Oliver's own FT Rate formula: FTA / FGA (attempts, not makes --
    # this is specifically about getting to the line, not finishing once
    # there, which is a separate skill outside the published Four Factors).
    log["ft_rate"] = log["own_fta"] / log["own_fga"].replace(0, np.nan)
    log["opp_ft_rate_allowed"] = log["opp_fta"] / log["opp_fga"].replace(0, np.nan)

    return log.sort_values(["team", "date"], kind="stable")


def attach_ft_rate_features(games: pd.DataFrame) -> pd.DataFrame:
    """Public entry point: returns `games` with 6 new walk-forward-safe
    (shift(1)-before-rolling) columns: home_/away_ft_rate_l10 (offense,
    drawing fouls), home_/away_opp_ft_rate_allowed_l10 (defense, fouls
    allowed), and their diffs. League-average fallback for warmup/unmatched
    rows is computed directly from this project's own real join output
    (printed at runtime), not guessed."""
    team_games = _load_box_team_games()

    # League-average FT rate, computed directly from the box-score team-games
    # table itself (both home and away sides, every game) -- NOT from the
    # post-join log below, which only reflects whatever slice of games the
    # caller passed in and could be empty/unmatched in a small/edge-case call
    # (e.g. a single-game test fixture). Real, not guessed.
    all_ft_rates = pd.concat([
        team_games["fta_home"] / team_games["fga_home"].replace(0, np.nan),
        team_games["fta_away"] / team_games["fga_away"].replace(0, np.nan),
    ])
    league_avg_ft_rate = float(all_ft_rates.mean())
    print(f"[research_ft_rate] League-average FT rate (FTA/FGA), computed directly from the real "
          f"box-score data: {league_avg_ft_rate:.4f} (used as the warmup/unmatched-row fallback)")

    games_with_box = _attach_box_join(games, team_games)
    log = _ft_rate_team_log(games_with_box)

    grp = log.groupby("team", group_keys=False)
    log["ft_rate_l10"] = grp["ft_rate"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    log["opp_ft_rate_allowed_l10"] = grp["opp_ft_rate_allowed"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())

    home_feats = log[log["is_home"]][["game_id", "ft_rate_l10", "opp_ft_rate_allowed_l10"]].rename(columns={
        "ft_rate_l10": "home_ft_rate_l10", "opp_ft_rate_allowed_l10": "home_opp_ft_rate_allowed_l10"})
    away_feats = log[~log["is_home"]][["game_id", "ft_rate_l10", "opp_ft_rate_allowed_l10"]].rename(columns={
        "ft_rate_l10": "away_ft_rate_l10", "opp_ft_rate_allowed_l10": "away_opp_ft_rate_allowed_l10"})

    out = games.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")

    for col in ["home_ft_rate_l10", "away_ft_rate_l10",
                "home_opp_ft_rate_allowed_l10", "away_opp_ft_rate_allowed_l10"]:
        out[col] = out[col].fillna(league_avg_ft_rate)

    out["ft_rate_diff"] = out["home_ft_rate_l10"] - out["away_ft_rate_l10"]
    out["opp_ft_rate_allowed_diff"] = out["home_opp_ft_rate_allowed_l10"] - out["away_opp_ft_rate_allowed_l10"]
    return out


FEATURE_COLS = [
    "home_ft_rate_l10", "away_ft_rate_l10", "ft_rate_diff",
    "home_opp_ft_rate_allowed_l10", "away_opp_ft_rate_allowed_l10", "opp_ft_rate_allowed_diff",
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


def season_split_half_check(oos_baseline: pd.DataFrame, oos_variant: pd.DataFrame) -> bool:
    """REQUIRED stability check -- see module docstring. Returns True only if
    the variant's fit does NOT degrade past CORR_NOISE_FLOOR in EITHER
    season half, relative to the baseline computed on that same half
    (mirrors sports/nfl/research_nflfastr_epa_sos_adjustment.py's equivalent
    check, extended from sports/nba/research_starters_out.py's report-only
    version with an explicit pass/fail verdict)."""
    seasons = sorted(oos_variant["season"].dropna().unique())
    mid = seasons[len(seasons) // 2]
    halves = {
        f"early ({seasons[0]}-{mid - 1})": (oos_baseline[oos_baseline["season"] < mid],
                                             oos_variant[oos_variant["season"] < mid]),
        f"late ({mid}-{seasons[-1]})": (oos_baseline[oos_baseline["season"] >= mid],
                                        oos_variant[oos_variant["season"] >= mid]),
    }
    hr("SEASON SPLIT-HALF STABILITY CHECK (required before trusting the full-sample result)")
    stable = True
    for label, (base_half, var_half) in halves.items():
        base_m = float(np.corrcoef(base_half["predicted_margin"], base_half["actual_margin"])[0, 1])
        var_m = float(np.corrcoef(var_half["predicted_margin"], var_half["actual_margin"])[0, 1])
        base_t = float(np.corrcoef(base_half["predicted_total"], base_half["actual_total"])[0, 1])
        var_t = float(np.corrcoef(var_half["predicted_total"], var_half["actual_total"])[0, 1])
        md, td = var_m - base_m, var_t - base_t
        print(f"{label}: n_games={len(var_half):,}")
        print(f"  margin_corr {base_m:.4f} -> {var_m:.4f} ({md:+.4f})")
        print(f"  total_corr  {base_t:.4f} -> {var_t:.4f} ({td:+.4f})")
        if md < -CORR_NOISE_FLOOR or td < -CORR_NOISE_FLOOR:
            stable = False
    print(f"\n{'STABLE' if stable else 'NOT STABLE'} across both halves (neither corr degraded past the "
          f"noise floor in either half) -- {'trust' if stable else 'do NOT trust'} the full-sample "
          f"recommendation below.")
    return stable


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

    hr("1. ATTACH FT-RATE COLUMNS (from Datasets/NBA/nba-box-scores.csv, reusing "
       "research_four_factors.py's verified join)")
    games_ft = attach_ft_rate_features(games)
    odds_covered = games_ft["Home Line Close"].notna()
    matched_mask = games_ft.loc[odds_covered, "home_ft_rate_l10"].notna()
    print(f"Odds-covered games: {odds_covered.sum():,}")
    print(f"Mean home_ft_rate_l10 (odds-covered): {games_ft.loc[odds_covered, 'home_ft_rate_l10'].mean():.4f}")
    print(f"Mean home_opp_ft_rate_allowed_l10 (odds-covered): "
          f"{games_ft.loc[odds_covered, 'home_opp_ft_rate_allowed_l10'].mean():.4f}")

    hr("2. BASELINE FEATURES (features.py unmodified -- current production ML_FEATURE_COLS)")
    baseline_feats = build_features(games_ft, rr.history)  # extra columns ride along unused
    print(f"Feature rows: {len(baseline_feats):,}, ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")
    baseline, oos_baseline = run_pipeline(baseline_feats, ML_FEATURE_COLS, "BASELINE")

    hr("3. VARIANT: + FT rate (offense) / opponent FT rate allowed (defense) columns")
    variant_cols = ML_FEATURE_COLS + FEATURE_COLS
    variant, oos_variant = run_pipeline(baseline_feats, variant_cols, "VARIANT (+ft_rate)")

    hypothesis = Hypothesis(
        name="nba_four_factors_ft_rate",
        reasoning=(
            "Dean Oliver's 'Four Factors' (eFG%, TOV%, OREB%, FT rate) is the same long-established, "
            "externally published basketball-analytics framework this project already adopted 3 of 4 "
            "factors from (four_factors_efg_tov, four_factors_oreb_pct, both adopt_cautiously). FT rate "
            "(FTA/FGA -- a team's ability to draw fouls and get to the line relative to how many shots it "
            "takes, deliberately distinct from free-throw shooting percentage, a separate skill outside "
            "the published factor) is the fourth and final factor, never tested -- confirmed by grepping "
            "decision_log.jsonl for ft_rate/free_throw before writing this script: zero matches anywhere. "
            "Tested here alongside its natural defensive complement (opp_ft_rate_allowed_l10, the "
            "opponent's own FTA/FGA in each of a team's games -- fouling discipline/foul-drawing allowed), "
            "since real Four Factors reporting is always two-sided (a team's own factors AND what it "
            "allows opponents) and this project's own already-adopted OREB% feature already leans on the "
            "opposing side's box line internally -- a natural, motivated completion of the same framework, "
            "not scope creep. Feasibility is very high: same box-score dataset "
            "(Datasets/NBA/nba-box-scores.csv) and the same verified join machinery already used for the "
            "other three factors (research_four_factors.py) and for research_starters_out.py, no new data "
            "source needed."
        ),
        sport="NBA",
    )
    baseline_metrics = {k: baseline[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}
    variant_metrics = {k: variant[k] for k in ("margin_corr", "total_corr", "roi_pct", "roi_stderr_pct")}

    stable = season_split_half_check(oos_baseline, oos_variant)

    hr("4. HYPOTHESIS EVALUATION (full sample)")
    print(f"Baseline n_bets={baseline['n_bets']:,}  Variant n_bets={variant['n_bets']:,}")
    result = evaluate_hypothesis(hypothesis, baseline_metrics, variant_metrics)

    import json
    hr("RESULT")
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nFull-sample recommendation: {result.recommendation}")
    print(f"Split-half stable: {stable}")

    # Final call combines BOTH checks -- a full-sample "adopt" that does not
    # survive the split-half check is downgraded to "reject", the exact
    # discipline nfl_qb_trailing_epa_and_continuity's rejection established
    # (see this module's docstring). A full-sample "adopt_cautiously"/
    # "inconclusive" is left as-is either way (nothing to downgrade FROM).
    if result.recommendation == "adopt" and not stable:
        final = "reject"
        print("\nFINAL CALL: 'adopt' on the full sample did NOT survive the split-half stability check "
              "(same failure mode as nfl_qb_trailing_epa_and_continuity) -- downgraded to REJECT.")
    else:
        final = result.recommendation
        print(f"\nFINAL CALL: {final}")


if __name__ == "__main__":
    main()
