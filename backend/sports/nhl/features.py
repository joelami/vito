"""
Walk-forward-safe feature engineering for NHL games. Structurally identical
to sports/cfb/features.py and sports/mlb/features.py - every rolling/
trailing stat here is computed with the current game excluded (via shift(1)
before any rolling window), so a game's feature row only ever reflects
information that existed strictly before that game's puck drop.

Deliberately does NOT touch Datasets/NHL/archive/nhl_data_extensive.csv's
pre-computed roll_*/opp_*/season_* columns at all - see config.py's module
docstring point 1 for why (that file isn't even the one this module loads;
nhl_data_plus.csv, which this module's loader.py reads, has no pre-computed
rolling columns to begin with). Every rolling feature below is computed from
scratch here, the same shift-then-rolling pattern used by every other sport
in this project.

Differences from CFB, driven by the data rather than stylistic: `is_bowl`
becomes `is_playoff` (a real, derived flag - see loader.py's `_flag_playoff`
- rather than a season_type column); there is no neutral-venue flag (this
data carries no venue-neutral indicator, and NHL's rare neutral-site games -
Winter Classic, Stadium Series, Global Series - are still nominally the
listed team's "home" game for standings purposes, so no proxy is fabricated
here, unlike CFB's is_bowl-as-neutral-site approximation); `LEAGUE_AVG_TEAM_
SCORE` is NHL's real scale (~3 goals/team) instead of football's ~22-27.

ADOPTED via core/research.py's disciplined hypothesis test (see
decision_log.jsonl, "nhl_trailing_shot_diff_and_pp_rate"): trailing shot
differential (`home_shots`/`away_shots`, rolled) and power-play conversion
rate (`home_pp_goals`/`home_pp_opportunities`, rolled) - process-level,
possession-family signals published in hockey analytics (Corsi/Fenwick) as
predicting FUTURE performance more reliably than goals alone, since goals
are heavily influenced by shooting/goaltending variance over small samples.
Both are box-score numbers for the game being scored (verified: same-game
correlation with that game's own goals_for is 0.16 for shots, 0.44 for
power_play_goals) - NEVER usable as same-game features - so, like every
other rolling feature here, they are shift(1)'d per team before rolling, so
a game's feature row only reflects that team's games strictly before it.
Measured result: margin_corr improved 0.1399->0.1459 (+0.0060, beyond
CORR_NOISE_FLOOR), total_corr moved -0.0043 (within noise), backtest ROI
moved -0.17pp (within noise, i.e. NOT an ROI-chasing artifact - if anything
ROI moved the wrong way while fit genuinely improved, the opposite of the
`suspicious` overfitting signature). Adopted on the strength of the margin
fit improvement, not the (flat-vig-contaminated, see config.py) ROI number.

ADOPTED 2026-08-12 via core/research.py (see decision_log.jsonl,
"nhl_trailing_pk_pct", and docs/METHODOLOGY.md's dated entry): trailing
penalty-kill percentage - the defensive mirror of the already-adopted PP
conversion rate above. Computed from the OPPONENT's power-play goals/
opportunities in each of a team's games (the opponent's power play is this
team's kill situation), shift(1)'d then rolled the same sum-of-ratio way as
pp_pct_l10. Measured result: total_corr improved 0.0750->0.0849 (+0.0099,
well beyond CORR_NOISE_FLOOR and a meaningful ~13% relative gain on NHL's
historically weakest leg), margin_corr moved -0.0009 (within noise), ROI
moved +0.09pp (within noise, same direction as fit - not the `suspicious`
ROI-with-no-fit-basis pattern). `sports/nhl/research_goalie_pk.py`'s
companion hypothesis (trailing team save percentage / goaltending form) was
tested the same session and found a clean null - fit moved less than the
noise floor in both directions - matching the same "well-motivated but
didn't move anything" outcome trailing faceoff win% produced, and was
therefore NOT wired in here, per that same precedent.
"""

import pandas as pd

ROLL_WINDOW = 10  # trailing games considered for form stats - same window as CFB/MLB for structural consistency

# League-average goals scored per team in this dataset (post-cleaning
# completed games, 2004-2026): actual_total mean ~5.85 => ~2.92/team, rounded.
# Used only to fill a team's very first game in the data, where no trailing
# history exists yet.
LEAGUE_AVG_TEAM_SCORE = 2.9

# NHL teams play every 1-3 days typically (back-to-backs are common) - used
# only to fill a team's very first game, where no prior game date exists.
LEAGUE_AVG_REST_DAYS = 2.0

# League-average power-play conversion rate (goals / opportunities) across
# this dataset's real, cleaned games - used only to fill a team's very first
# game(s), where no trailing PP history exists yet (same role as
# LEAGUE_AVG_TEAM_SCORE above).
LEAGUE_AVG_PP_PCT = 0.189

# League-average penalty-kill percentage (1 - opponent PP goals / opponent PP
# opportunities) across this dataset's real, cleaned games - used only to
# fill a team's very first game(s), where no trailing PK history exists yet.
# Necessarily close to 1 - LEAGUE_AVG_PP_PCT (PK is the defensive mirror of
# PP - one side's conversion is the other side's failure to kill) but kept as
# its own measured constant rather than derived, same spirit as every other
# league-average fallback here.
LEAGUE_AVG_PK_PCT = 0.811


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (team, game) so rolling stats can be computed per-team."""
    cols = ["game_id", "date", "season", "home_covers_close", "home_win",
            "home_score", "away_score"]
    box_cols = ["home_shots", "away_shots", "home_pp_goals", "home_pp_opportunities",
                "away_pp_goals", "away_pp_opportunities"]

    home = games[cols + box_cols + ["home_team_id", "away_team_id"]].rename(columns={
        "home_team_id": "team", "away_team_id": "opponent", "home_covers_close": "covered",
        "home_win": "won", "home_score": "points_for", "away_score": "points_against",
        "home_shots": "shots_for", "away_shots": "shots_against",
        "home_pp_goals": "pp_goals", "home_pp_opportunities": "pp_opportunities",
        # opponent's PP is THIS team's kill situation - see PK% docstring below
        "away_pp_goals": "pk_goals_against", "away_pp_opportunities": "pk_times_shorthanded",
    })
    home["is_home"] = True

    away = games[cols + box_cols + ["away_team_id", "home_team_id"]].rename(columns={
        "away_team_id": "team", "home_team_id": "opponent", "home_covers_close": "covered",
        "home_win": "won", "home_score": "points_against", "away_score": "points_for",
        "away_shots": "shots_for", "home_shots": "shots_against",
        "away_pp_goals": "pp_goals", "away_pp_opportunities": "pp_opportunities",
        # opponent's PP is THIS team's kill situation - see PK% docstring below
        "home_pp_goals": "pk_goals_against", "home_pp_opportunities": "pk_times_shorthanded",
    })
    away["covered"] = away["covered"].apply(lambda v: -v if pd.notna(v) else v)
    away["won"] = 1 - away["won"]
    away["is_home"] = False

    log = pd.concat([home, away], ignore_index=True)
    # rescale covered from {-1, 0, 1} (loss/push/win) to a genuine 0-1 rate so a rolling
    # mean is an actual "cover percentage" rather than ranging -1..1
    log["covered"] = log["covered"].map({1: 1.0, 0: 0.5, -1: 0.0})
    return log.sort_values(["team", "date"], kind="stable")


def _rolling_form(log: pd.DataFrame) -> pd.DataFrame:
    grp = log.groupby("team", group_keys=False)

    # shift(1) first so the current game is never included in its own rolling window
    log["ats_pct_l10"] = grp["covered"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    log["win_pct_l10"] = grp["won"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    log["pf_l10"] = grp["points_for"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    log["pa_l10"] = grp["points_against"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )

    # Trailing shot differential (own shots for minus against, rolled) - a
    # process-level possession signal (the Corsi/Fenwick family), computed
    # with the exact same shift(1)-before-rolling discipline as every other
    # feature here. shots_for/shots_against are this-game's own box-score
    # numbers (see loader.py) so they must never be used un-shifted.
    log["shot_diff"] = log["shots_for"] - log["shots_against"]
    log["shot_diff_l10"] = grp["shot_diff"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )

    # Trailing power-play conversion rate: rolling SUM of goals over rolling
    # SUM of opportunities (both shift(1)'d first), not a mean of per-game
    # ratios - avoids 0/0 blowups on the ~1.7% of games with zero power-play
    # opportunities and matches how PP% is conventionally reported over a
    # multi-game window.
    pp_goals_l10sum = grp["pp_goals"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum()
    )
    pp_opp_l10sum = grp["pp_opportunities"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum()
    )
    log["pp_pct_l10"] = pp_goals_l10sum / pp_opp_l10sum

    # Trailing penalty-kill percentage: the defensive mirror of pp_pct_l10
    # above, same rolling-SUM-of-ratio convention (not mean-of-per-game-
    # ratios, for the same 0/0-avoidance reason). pk_goals_against/
    # pk_times_shorthanded come from the OPPONENT's PP goals/opportunities in
    # each game (see _team_game_log - the opponent's power play is this
    # team's kill situation), so this measures the team's OWN penalty-
    # killing form, not anything about whichever opponents it happened to
    # face. ADOPTED via core/research.py hypothesis test - see this module's
    # docstring and decision_log.jsonl ("nhl_trailing_pk_pct").
    pk_goals_l10sum = grp["pk_goals_against"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum()
    )
    pk_opp_l10sum = grp["pk_times_shorthanded"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum()
    )
    log["pk_pct_l10"] = 1.0 - pk_goals_l10sum / pk_opp_l10sum

    log["rest_days"] = grp["date"].apply(lambda s: s.diff().dt.days)

    # signed current streak (positive = winning streak, negative = losing streak), pre-game
    def _streak(sub: pd.DataFrame) -> pd.Series:
        prior = sub["won"].shift(1)
        streaks = []
        cur = 0
        for w in prior:
            if pd.isna(w):
                streaks.append(0)
                cur = 0
                continue
            if w == 1:
                cur = cur + 1 if cur >= 0 else 1
            else:
                cur = cur - 1 if cur <= 0 else -1
            streaks.append(cur)
        return pd.Series(streaks, index=sub.index)

    log["streak"] = grp.apply(_streak).reset_index(level=0, drop=True)
    return log


def build_features(games: pd.DataFrame, rating_history: pd.DataFrame) -> pd.DataFrame:
    """
    `games` is the loader's cleaned DataFrame; `rating_history` is the
    `RatingResult.history` from power_ratings (indexed by game_id, holding
    pre-game ratings). Returns `games` joined with engineered, walk-forward
    -safe feature columns for both teams.
    """
    log = _team_game_log(games)
    log = _rolling_form(log)

    home_feats = log[log["is_home"]][
        ["game_id", "ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak",
         "shot_diff_l10", "pp_pct_l10", "pk_pct_l10"]
    ].rename(columns={
        "ats_pct_l10": "home_ats_pct_l10", "win_pct_l10": "home_win_pct_l10",
        "pf_l10": "home_pf_l10", "pa_l10": "home_pa_l10",
        "rest_days": "home_rest_days", "streak": "home_streak",
        "shot_diff_l10": "home_shot_diff_l10", "pp_pct_l10": "home_pp_pct_l10",
        "pk_pct_l10": "home_pk_pct_l10",
    })
    away_feats = log[~log["is_home"]][
        ["game_id", "ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak",
         "shot_diff_l10", "pp_pct_l10", "pk_pct_l10"]
    ].rename(columns={
        "ats_pct_l10": "away_ats_pct_l10", "win_pct_l10": "away_win_pct_l10",
        "pf_l10": "away_pf_l10", "pa_l10": "away_pa_l10",
        "rest_days": "away_rest_days", "streak": "away_streak",
        "shot_diff_l10": "away_shot_diff_l10", "pp_pct_l10": "away_pp_pct_l10",
        "pk_pct_l10": "away_pk_pct_l10",
    })

    out = games.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out = out.merge(rating_history[["home_rating_pre", "away_rating_pre", "rating_diff_pre",
                                     "elo_home_win_prob"]], left_on="game_id", right_index=True, how="left")

    out["home_rest_days"] = out["home_rest_days"].fillna(LEAGUE_AVG_REST_DAYS)
    out["away_rest_days"] = out["away_rest_days"].fillna(LEAGUE_AVG_REST_DAYS)
    out["home_ats_pct_l10"] = out["home_ats_pct_l10"].fillna(0.5)
    out["away_ats_pct_l10"] = out["away_ats_pct_l10"].fillna(0.5)
    out["home_win_pct_l10"] = out["home_win_pct_l10"].fillna(0.5)
    out["away_win_pct_l10"] = out["away_win_pct_l10"].fillna(0.5)
    out["home_streak"] = out["home_streak"].fillna(0)
    out["away_streak"] = out["away_streak"].fillna(0)
    out["home_pf_l10"] = out["home_pf_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["home_pa_l10"] = out["home_pa_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["away_pf_l10"] = out["away_pf_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["away_pa_l10"] = out["away_pa_l10"].fillna(LEAGUE_AVG_TEAM_SCORE)
    out["home_shot_diff_l10"] = out["home_shot_diff_l10"].fillna(0.0)
    out["away_shot_diff_l10"] = out["away_shot_diff_l10"].fillna(0.0)
    out["home_pp_pct_l10"] = out["home_pp_pct_l10"].fillna(LEAGUE_AVG_PP_PCT)
    out["away_pp_pct_l10"] = out["away_pp_pct_l10"].fillna(LEAGUE_AVG_PP_PCT)
    out["home_pk_pct_l10"] = out["home_pk_pct_l10"].fillna(LEAGUE_AVG_PK_PCT)
    out["away_pk_pct_l10"] = out["away_pk_pct_l10"].fillna(LEAGUE_AVG_PK_PCT)

    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]

    out["naive_total"], out["naive_margin"] = naive_score_features(
        out["home_pf_l10"], out["home_pa_l10"], out["away_pf_l10"], out["away_pa_l10"]
    )
    return out


def naive_score_features(home_pf, home_pa, away_pf, away_pa):
    """
    Naive "implied score" baseline: each team's expected goals = average of
    (its own scoring rate, its opponent's rate of allowing goals) - a
    standard, simple total/margin prior that gives the ML model a strong
    starting signal for scoring pace, which team strength alone (rating
    differential) doesn't capture. Identical formula to every other sport
    here - sport-agnostic, just fed NHL-shaped (goals, not points/runs)
    inputs. Shared by `build_features` (historical, vectorized over a
    DataFrame) and any future manually-entered-game caller so the formula
    only lives in one place.
    """
    implied_home_score = (home_pf + away_pa) / 2.0
    implied_away_score = (away_pf + home_pa) / 2.0
    return implied_home_score + implied_away_score, implied_home_score - implied_away_score


def current_form_snapshot(games: pd.DataFrame) -> pd.DataFrame:
    """
    Each team's rolling form as of RIGHT NOW (i.e. including their most
    recent played game, unlike the walk-forward columns in `build_features`
    which deliberately exclude the current row). Used only to score brand
    new manually-entered upcoming games - there's no leakage concern since
    every game in `games` is legitimately in the past relative to a new one.
    Indexed by canonical team_id (see config.canonical_team_id). Not
    exercised by verify.py (out of scope for this backend-only pass) but
    kept for interface parity with the other sports' features.py modules.
    """
    log = _team_game_log(games)
    grp = log.groupby("team", group_keys=False)
    log["ats_pct_l10"] = grp["covered"].apply(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    log["win_pct_l10"] = grp["won"].apply(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    log["pf_l10"] = grp["points_for"].apply(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    log["pa_l10"] = grp["points_against"].apply(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())

    def _streak_now(sub: pd.DataFrame) -> pd.Series:
        cur = 0
        out = []
        for w in sub["won"]:
            cur = (cur + 1 if cur >= 0 else 1) if w == 1 else (cur - 1 if cur <= 0 else -1)
            out.append(cur)
        return pd.Series(out, index=sub.index)

    log["streak"] = grp.apply(_streak_now).reset_index(level=0, drop=True)

    latest = log.sort_values(["team", "date"]).groupby("team").tail(1).set_index("team")
    return latest[["ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "streak", "date"]].rename(
        columns={"date": "last_game_date"}
    )


ML_FEATURE_COLS = [
    "rating_diff_pre", "rest_diff", "home_rest_days", "away_rest_days",
    "home_ats_pct_l10", "away_ats_pct_l10", "home_win_pct_l10", "away_win_pct_l10",
    "home_pf_l10", "home_pa_l10", "away_pf_l10", "away_pa_l10",
    "naive_total", "naive_margin",
    "home_streak", "away_streak", "is_playoff",
    # Adopted via core/research.py hypothesis test (see decision_log.jsonl,
    # "nhl_trailing_shot_diff_and_pp_rate") - trailing shot differential and
    # power-play conversion rate, genuinely improved margin_corr beyond
    # CORR_NOISE_FLOOR without the ROI number moving up (see features.py's
    # module docstring for the full measured result).
    "home_shot_diff_l10", "away_shot_diff_l10", "home_pp_pct_l10", "away_pp_pct_l10",
    # Adopted via core/research.py hypothesis test (see decision_log.jsonl,
    # "nhl_trailing_pk_pct") - trailing penalty-kill percentage, the
    # defensive mirror of pp_pct_l10 above. Genuinely improved total_corr
    # beyond CORR_NOISE_FLOOR (see features.py's module docstring for the
    # full measured result).
    "home_pk_pct_l10", "away_pk_pct_l10",
]
