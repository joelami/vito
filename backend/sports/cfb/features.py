"""
Walk-forward-safe feature engineering for CFB games. Structurally identical
to sports/nfl/features.py - every rolling/trailing stat here is computed
with the current game excluded (via shift(1) before any rolling window), so
a game's feature row only ever reflects information that existed strictly
before that game's kickoff.

Differences from the NFL version, both driven by the data rather than
stylistic: `is_divisional` doesn't exist (CFB has 130+ FBS teams across many
conferences and this CSV carries no conference column, so there's nothing
to compute it from - dropped rather than faked); `is_bowl` stands in for
NFL's `is_playoff`; and the neutral-score fill for a team's first-ever game
uses CFB's actual league-average score (~27/team) instead of NFL's ~22.

A real consequence of the sparse odds coverage (see loader.py docstring):
`home_covers_close` is NaN for the ~58% of games with no spread, so
`ats_pct_l10` is computed over far fewer real observations per team than
in the NFL version - for long stretches a team's rolling ATS window may
contain zero non-NaN games, in which case it falls back to the same 0.5
neutral fill NFL uses. This is walk-forward-safe (no leakage), just noisier
as a signal than NFL's, and is reported honestly in verify.py's output
rather than hidden.

Box-score features (yards_l10, int_thrown_l10, adopted via
core/research.py's hypothesis test "boxscore_yards_turnovers", see
decision_log.jsonl): trailing team-level total offensive yards (passing +
rushing) and interceptions thrown, from loader.load_team_game_boxscores(),
rolled the same shift(1)-then-rolling(10) walk-forward-safe way as every
other feature here. Motivation: final score is a noisy downstream outcome
of underlying offensive production - two teams can reach the same score off
very different yardage/turnover performances, and the model had zero
visibility into that distinction before this. Measured result: total_corr
+0.027 and margin_corr +0.006 (both above the noise floor), ROI -1.08pp but
within its own standard error - recommendation "adopt_cautiously", the same
standard already applied to NFL's Pythagorean and MAE-loss changes. Join
coverage is partial (~92% of games get a full team-vs-team match, ~95% get
at least one side) - the unmatched team-games roll through as NaN and get
skipped by the rolling window the same way sparse ats_pct_l10 data does,
then a league-average fallback covers windows with zero real observations.

Completion percentage (comp_pct_l10, adopted via hypothesis test
"qb_completion_pct", see decision_log.jsonl): trailing team-level completion
percentage (attempts-weighted: trailing sum(comp) / trailing sum(att) over
the L10 window, not an unweighted average of per-game percentages), from
loader.load_team_game_comp_pct(). Motivation: completion rate is a passing-
efficiency signal distinct from raw yardage (a team can pile up yards
inefficiently via a few chunk plays, or convert efficiently on short,
high-percentage throws) - real QB-efficiency ratings (QBR, passer rating)
weight it heavily for exactly that reason. Measured result (tested as an
addition ON TOP of the already-adopted yards/INT features, not standalone):
total_corr +0.010 (above noise floor), margin_corr -0.004 (within noise,
essentially flat), ROI -1.51pp within its own standard error -
recommendation "adopt_cautiously", a weaker/more marginal case than the
yards/INT features above (mixed fit signal, larger within-noise ROI dip)
but still meeting the same mechanical bar this project uses throughout.

Season-phase signal (season_week_adj, adopted via hypothesis test
"cfb_season_week_adj", see decision_log.jsonl and
sports/cfb/research_week_trends.py): a diagnostic split of the model's own
out-of-sample fit by phase of season found real dispersion the model had no
direct way to see - margin_corr ran ~0.38 in weeks 1-3, dropped to ~0.27 for
weeks 4-16, and fell further to ~0.13 in bowl/postseason games - despite
is_bowl and rest_days already being available features. The raw `week`
column can't be used as-is for this: every postseason/bowl game in this
dataset is labeled week=1 (a season_type-driven placeholder, not a real
in-season week number - verified 599/599 is_bowl rows have week==1).
`season_week_adj` is `week` for regular-season games, with bowl games
reassigned to (that season's max regular-season week + 1) so they sort
chronologically after the regular season the way they actually occur,
instead of looking like week-1 openers. Measured result: margin_corr +0.019
(well above the noise floor), total_corr +0.001 (essentially flat, below
the noise floor), ROI -3.05% -> -2.38% (+0.67pp, within its own standard
error) - recommendation "adopt" (real fit improvement, ROI did not get
worse). See docs/METHODOLOGY.md's dated CFB subsection for the full
week-over-week diagnostic table this was built from.
"""

import pandas as pd

from . import loader

ROLL_WINDOW = 10  # trailing games considered for form stats

# League-average points scored per team in this dataset (completed games,
# 2004-2025): actual_total mean ~54.2 => ~27.1/team. Used only to fill a
# team's very first game in the data, where no trailing history exists yet.
LEAGUE_AVG_TEAM_SCORE = 27.0

# League-average team-game total offensive yards / interceptions thrown,
# from the box-score aggregate (loader.load_team_game_boxscores()). Used
# only to fill a team-game's rolling window when it has zero real box-score
# observations behind it yet (first-ever tracked game, or a stretch with no
# matched box-score rows for that event_id) - same role LEAGUE_AVG_TEAM_SCORE
# plays for pf_l10/pa_l10. Computed once at import time from the same source
# data the rolling features are built from, not hand-picked.
_BOXSCORE_TABLE = loader.load_team_game_boxscores()
LEAGUE_AVG_TEAM_YARDS = float(_BOXSCORE_TABLE["total_yards"].mean())
LEAGUE_AVG_TEAM_INT_THROWN = float(_BOXSCORE_TABLE["int_thrown"].mean())

# League-wide attempts-weighted completion percentage (sum of all completions
# / sum of all attempts across the whole dataset), from
# loader.load_team_game_comp_pct(). Used only to fill a team-game's rolling
# completion-pct window when it has zero real observations behind it yet.
_COMP_PCT_TABLE = loader.load_team_game_comp_pct()
LEAGUE_AVG_COMP_PCT = float(_COMP_PCT_TABLE["comp"].sum() / _COMP_PCT_TABLE["att"].sum())


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (team, game) so rolling stats can be computed per-team."""
    cols = ["game_id", "date", "season", "home_covers_close", "home_win",
            "home_score", "away_score"]

    home = games[cols + ["home_franchise", "away_franchise"]].rename(columns={
        "home_franchise": "team", "away_franchise": "opponent", "home_covers_close": "covered",
        "home_win": "won", "home_score": "points_for", "away_score": "points_against",
    })
    home["is_home"] = True

    away = games[cols + ["away_franchise", "home_franchise"]].rename(columns={
        "away_franchise": "team", "home_franchise": "opponent", "home_covers_close": "covered",
        "home_win": "won", "home_score": "points_against", "away_score": "points_for",
    })
    # flip perspective for the away team: covered is {1, 0 (push), -1, NaN}; negate, push/NaN unchanged
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


def _boxscore_rolling(games: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward-safe (shift(1)-then-rolling(10)) trailing total offensive
    yards and interceptions thrown, per team per game_id, joined from
    loader.load_team_game_boxscores() via event_id. Returns one row per
    game_id with home_yards_l10/away_yards_l10/home_int_thrown_l10/
    away_int_thrown_l10 - same shape contract as `_rolling_form`'s output,
    just a separate table since its source data (event_id-keyed box scores)
    is different from the games CSV's own columns.
    """
    id_cols = games[["game_id", "event_id", "date", "home_franchise", "away_franchise"]]

    home = id_cols.merge(
        _BOXSCORE_TABLE, left_on=["event_id", "home_franchise"], right_on=["event_id", "team"], how="left"
    )[["game_id", "date", "home_franchise", "total_yards", "int_thrown"]].rename(
        columns={"home_franchise": "team"}
    )
    away = id_cols.merge(
        _BOXSCORE_TABLE, left_on=["event_id", "away_franchise"], right_on=["event_id", "team"], how="left"
    )[["game_id", "date", "away_franchise", "total_yards", "int_thrown"]].rename(
        columns={"away_franchise": "team"}
    )
    log = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")

    grp = log.groupby("team", group_keys=False)
    # NaN team-games (no box-score match for that event_id/team) are simply
    # excluded from the rolling mean by pandas, the same way sparse
    # home_covers_close values are handled for ats_pct_l10 above - not
    # filled here, only after the window has zero real observations at all.
    log["yards_l10"] = grp["total_yards"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    log["int_thrown_l10"] = grp["int_thrown"].apply(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )

    home_out = log.merge(id_cols[["game_id", "home_franchise"]], on="game_id")
    home_out = home_out[home_out["team"] == home_out["home_franchise"]][
        ["game_id", "yards_l10", "int_thrown_l10"]
    ].rename(columns={"yards_l10": "home_yards_l10", "int_thrown_l10": "home_int_thrown_l10"})

    away_out = log.merge(id_cols[["game_id", "away_franchise"]], on="game_id")
    away_out = away_out[away_out["team"] == away_out["away_franchise"]][
        ["game_id", "yards_l10", "int_thrown_l10"]
    ].rename(columns={"yards_l10": "away_yards_l10", "int_thrown_l10": "away_int_thrown_l10"})

    return home_out.merge(away_out, on="game_id", how="outer")


def _comp_pct_rolling(games: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward-safe trailing team-level completion percentage, attempts-
    weighted: trailing sum(comp) / trailing sum(att) over the L10 window
    (shift(1) first), not an unweighted average of per-game percentages -
    a team that goes 1/1 one week shouldn't count as much as a team that
    goes 20/30. Same join/merge shape as `_boxscore_rolling` above, from
    loader.load_team_game_comp_pct() via event_id.
    """
    id_cols = games[["game_id", "event_id", "date", "home_franchise", "away_franchise"]]

    home = id_cols.merge(
        _COMP_PCT_TABLE, left_on=["event_id", "home_franchise"], right_on=["event_id", "team"], how="left"
    )[["game_id", "date", "home_franchise", "comp", "att"]].rename(columns={"home_franchise": "team"})
    away = id_cols.merge(
        _COMP_PCT_TABLE, left_on=["event_id", "away_franchise"], right_on=["event_id", "team"], how="left"
    )[["game_id", "date", "away_franchise", "comp", "att"]].rename(columns={"away_franchise": "team"})
    log = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")

    grp = log.groupby("team", group_keys=False)
    roll_comp = grp["comp"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    roll_att = grp["att"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).sum())
    log["comp_pct_l10"] = (roll_comp / roll_att).replace([float("inf"), float("-inf")], pd.NA)

    home_out = log.merge(id_cols[["game_id", "home_franchise"]], on="game_id")
    home_out = home_out[home_out["team"] == home_out["home_franchise"]][
        ["game_id", "comp_pct_l10"]
    ].rename(columns={"comp_pct_l10": "home_comp_pct_l10"})

    away_out = log.merge(id_cols[["game_id", "away_franchise"]], on="game_id")
    away_out = away_out[away_out["team"] == away_out["away_franchise"]][
        ["game_id", "comp_pct_l10"]
    ].rename(columns={"comp_pct_l10": "away_comp_pct_l10"})

    return home_out.merge(away_out, on="game_id", how="outer")


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
        ["game_id", "ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak"]
    ].rename(columns={
        "ats_pct_l10": "home_ats_pct_l10", "win_pct_l10": "home_win_pct_l10",
        "pf_l10": "home_pf_l10", "pa_l10": "home_pa_l10",
        "rest_days": "home_rest_days", "streak": "home_streak",
    })
    away_feats = log[~log["is_home"]][
        ["game_id", "ats_pct_l10", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak"]
    ].rename(columns={
        "ats_pct_l10": "away_ats_pct_l10", "win_pct_l10": "away_win_pct_l10",
        "pf_l10": "away_pf_l10", "pa_l10": "away_pa_l10",
        "rest_days": "away_rest_days", "streak": "away_streak",
    })

    out = games.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out = out.merge(rating_history[["home_rating_pre", "away_rating_pre", "rating_diff_pre",
                                     "elo_home_win_prob"]], left_on="game_id", right_index=True, how="left")

    box = _boxscore_rolling(games)
    out = out.merge(box, on="game_id", how="left")
    out["home_yards_l10"] = out["home_yards_l10"].fillna(LEAGUE_AVG_TEAM_YARDS)
    out["away_yards_l10"] = out["away_yards_l10"].fillna(LEAGUE_AVG_TEAM_YARDS)
    out["home_int_thrown_l10"] = out["home_int_thrown_l10"].fillna(LEAGUE_AVG_TEAM_INT_THROWN)
    out["away_int_thrown_l10"] = out["away_int_thrown_l10"].fillna(LEAGUE_AVG_TEAM_INT_THROWN)
    out["yards_diff_l10"] = out["home_yards_l10"] - out["away_yards_l10"]
    out["int_thrown_diff_l10"] = out["home_int_thrown_l10"] - out["away_int_thrown_l10"]

    comp_pct = _comp_pct_rolling(games)
    out = out.merge(comp_pct, on="game_id", how="left")
    out["home_comp_pct_l10"] = out["home_comp_pct_l10"].fillna(LEAGUE_AVG_COMP_PCT)
    out["away_comp_pct_l10"] = out["away_comp_pct_l10"].fillna(LEAGUE_AVG_COMP_PCT)
    out["comp_pct_diff_l10"] = out["home_comp_pct_l10"] - out["away_comp_pct_l10"]

    # first game of a team's history has no prior rest date -> fill with a neutral bye-week value
    out["home_rest_days"] = out["home_rest_days"].fillna(7)
    out["away_rest_days"] = out["away_rest_days"].fillna(7)
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

    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]

    out["naive_total"], out["naive_margin"] = naive_score_features(
        out["home_pf_l10"], out["home_pa_l10"], out["away_pf_l10"], out["away_pa_l10"]
    )

    out["season_week_adj"] = season_week_adjusted(out)
    return out


def season_week_adjusted(games: pd.DataFrame) -> pd.Series:
    """
    Calendar-correct season-phase signal, adopted via hypothesis test
    "cfb_season_week_adj" (see decision_log.jsonl and this module's
    docstring). `week` as-is for regular-season games; bowl/postseason games
    (whose raw `week` is a degenerate placeholder of 1 for every single one
    of them in this dataset) are reassigned to that season's max
    regular-season week + 1, so they sort chronologically after the regular
    season the way they actually occur, rather than looking like week-1
    openers.
    """
    max_reg_week_by_season = games[~games["is_bowl"]].groupby("season")["week"].max()
    adj = games["week"].astype(float).copy()
    bowl_mask = games["is_bowl"]
    adj.loc[bowl_mask] = games.loc[bowl_mask, "season"].map(max_reg_week_by_season) + 1.0
    return adj


def naive_score_features(home_pf, home_pa, away_pf, away_pa):
    """
    Naive "implied score" baseline: each team's expected points = average of
    (its own scoring rate, its opponent's rate of allowing points) - a
    standard, simple total/margin prior that gives the ML model a strong
    starting signal for scoring pace, which team strength alone (rating
    differential) doesn't capture. Shared by `build_features` (historical,
    vectorized over a DataFrame) and any future manually-entered-game caller
    so the formula only lives in one place. Identical formula to the NFL
    version - sport-agnostic, just fed CFB-shaped inputs.
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
    Indexed by team (as given in the source data, no realignment mapping -
    see config.canonical_team), one row per team. Not exercised by verify.py
    (out of scope for this backend-only pass) but kept for interface parity
    with sports/nfl/features.py.
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
    "home_streak", "away_streak", "is_bowl", "is_neutral_venue",
    # Adopted (adopt_cautiously) via core/research.py hypothesis test
    # "boxscore_yards_turnovers" - see decision_log.jsonl and this module's
    # docstring for the measured margin/total-corr improvement and ROI dip
    # (within noise) that motivated keeping this.
    "home_yards_l10", "away_yards_l10", "yards_diff_l10",
    "home_int_thrown_l10", "away_int_thrown_l10", "int_thrown_diff_l10",
    # Adopted (adopt_cautiously) via hypothesis test "qb_completion_pct" -
    # see decision_log.jsonl and this module's docstring.
    "home_comp_pct_l10", "away_comp_pct_l10", "comp_pct_diff_l10",
    # Adopted ("adopt") via hypothesis test "cfb_season_week_adj" - see
    # decision_log.jsonl and this module's docstring.
    "season_week_adj",
]


def extra_matchup_features(home_fr, away_fr, game_date, home_row, away_row, is_playoff=False, **_ignored) -> dict:
    """
    Live-scoring counterpart for CFB going live (see core/dispatch.py's
    LIVE_SPORTS comment, 2026-08-25). Honest split, not a blanket fix:

    is_bowl: real, computable now. Same season_type==3 postseason
    convention this dataset's own is_bowl_game() uses (see config.py) is
    exactly what ESPN's own is_playoff already means for a live-synced
    event (see core/espn_client.py's season_type parsing) -- so it's a
    direct alias, not a fallback.

    season_week_adj, home/away_yards_l10, home/away_int_thrown_l10,
    home/away_comp_pct_l10 (and their _diff variants): NOT fixed here.
    These are real, hypothesis-tested, adopted features (see
    decision_log.jsonl) that this hook could theoretically supply, but
    doing so honestly needs real plumbing this pass didn't build:
    season_week_adj needs ESPN's real week.number threaded through
    espn_client.parse_events -> harness.py's two sync call sites ->
    core/dispatch.py -> here (a live-scoring signature change spanning
    every sport, not just CFB); the box-score-derived features need a
    live box-score data source that doesn't exist in this codebase at
    all yet (espn_client only ever pulled scoreboard/odds, never
    per-game stat lines). Both flagged by core/matchup.py's own
    missing-feature warning rather than silently guessed at under time
    pressure right after this exact bug class was found and fixed twice
    elsewhere today -- a real, accepted gap, tracked here, not a
    forgotten one.
    """
    return {"is_bowl": int(bool(is_playoff))}
