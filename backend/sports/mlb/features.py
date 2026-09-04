"""
Walk-forward-safe feature engineering for MLB games. Structurally identical
to sports/cfb/features.py - every rolling/trailing stat here is computed
with the current game excluded (via shift(1) before any rolling window), so
a game's feature row only ever reflects information that existed strictly
before that game's first pitch.

Differences from the CFB/NFL versions, all driven by the data rather than
stylistic: there is no `ats_pct_l10` at all (no odds data exists to compute
an ATS result from - see loader.py/config.py's module docstrings), no
`is_bowl`/`is_neutral_venue` (MLB's regular season, which is all this
dataset contains, has no bowl/playoff/neutral-site games), and no divisional
flag (MLB has real divisions, but nothing here currently uses them). In
their place, two cheap, real, non-fabricated context flags are carried
straight from the loader: `is_interleague` (AL team @ NL team or vice versa)
and `is_night` (night game). `LEAGUE_AVG_TEAM_SCORE` is MLB's actual scale
(~4.6 runs/team) rather than football's ~22-27 points/team.

STARTING PITCHER QUALITY: `home_sp_er_lN` / `away_sp_er_lN` (a starter's
rolling earned-runs-allowed PER START, not innings-normalized ERA - see
sports/mlb/starting_pitcher.py's module docstring for the exact scoping
and why) must already be columns on the `games` DataFrame passed into
`build_features` - call `starting_pitcher.attach_starter_quality(games)`
before this, same pattern as sports/nfl/weather.py's attach_weather()
being called before NFL's build_features. `sp_er_diff_lN` is computed here,
the same "diff" convention as `rest_diff`/`pyth_pct_diff` elsewhere in this
project. Adopted (adopt_cautiously) via a core.research hypothesis test -
see research_starting_pitcher.py and decision_log.jsonl.

MARKET-IMPLIED PROBABILITY: `market_fair_home_prob` (devigged home-moneyline
probability from the 2012-2021 odds archive, 0.5 - "no information" -
fallback for the ~73% of 1990-2025 history outside that archive's coverage)
is computed here directly from `games`' own `Home Odds Close`/`Away Odds
Close` columns (already present after `odds_loader.attach_odds(games)`,
called before `build_features` in every pipeline that reaches this point -
same ordering dependency as the starter-quality columns above). Adopted
(plain "adopt" - a real, noise-floor-clearing statistical fit improvement,
not just a harmless null) via a core.research hypothesis test - see
research_moneyline_dog_calibration.py and decision_log.jsonl. Reasoning in
brief: a real diagnostic found this model is well-calibrated when its pick
agrees with the market's favored side, but overconfident by ~9 percentage
points specifically when it disagrees (the market is pricing in real-time
information - injuries, lineup news, sharp money - this schedule/form-based
feature set structurally lacks) - letting the ML models see the market's
own probability directly, instead of only comparing against it after the
fact at bet-selection time, measurably improved margin_corr (+0.0078, well
above the 0.005 noise floor) without hurting ROI. This is the SAME 'Close'
snapshot price `core.edge_finder` already prices every backtested bet
against for this dataset (its only snapshot - see odds_loader.py) - known
pre-bet, not future information, so this is not leakage.

STARTING PITCHER SKILL (K-BB%): `home_sp_kbb_pct_lN` / `away_sp_kbb_pct_lN`
(a starter's rolling (K-BB)/batters-faced over their last 8 starts - see
sports/mlb/starter_kbb_quality.py's module docstring for the exact scoping)
must already be columns on `games` - call
`starter_kbb_quality.attach_starter_kbb_pct(games)` before this, same
ordering dependency as the ER-based starter feature above. A DIFFERENT,
more skill-isolated metric than home_sp_er_lN/away_sp_er_lN (K-BB% only
counts outcomes the pitcher himself overwhelmingly controls, stripping out
defense/ballpark/sequencing-luck noise a runs-allowed proxy can't
separate from true skill) - built as a genuinely new mechanism, not a
retest of the ER feature. `sp_kbb_pct_diff_lN` is home minus away (K-BB% is
a "higher is better" stat, the OPPOSITE polarity from ER's "lower is
better" - this diff's sign is deliberately flipped from sp_er_diff_lN's
away-minus-home so that positive still means "favors home" in both
columns, not a copy-pasted formula that would silently invert the signal).
Adopted (plain "adopt" - total_corr improved +0.0062, above the 0.005
noise floor, ROI improved +0.14pp, margin_corr moved -0.0015, within
noise either way) via a core.research hypothesis test - see
research_starter_kbb_pct.py and decision_log.jsonl
("starter_kbb_pct_rolling").
"""

import pandas as pd

from core import odds_math

ROLL_WINDOW = 10  # trailing games considered for form stats - same window as NFL/CFB for structural consistency, see module docstring for the tradeoff this implies given MLB's 162-game season

# League-average runs scored per team in this dataset (completed games,
# 1990-2025, non-tie): actual_total mean ~9.17 => ~4.58/team, rounded.
# Used only to fill a team's very first game in the data, where no trailing
# history exists yet.
LEAGUE_AVG_TEAM_SCORE = 4.6

# Typical days between a team's games (MLB teams play almost daily with
# occasional off days) - used only to fill a team's very first game, where
# no prior game date exists to diff against.
LEAGUE_AVG_REST_DAYS = 1.0

# Each team's CURRENT league (AL/NL), keyed by the same Retrosheet-style
# franchise codes config.canonical_team() resolves to. Derived empirically
# from the real historical dataset (see extra_matchup_features() below for
# the exact derivation and why it matters that this wasn't typed from
# memory) - not a static assumption, a verified snapshot of real data.
TEAM_LEAGUE = {"ANA": "AL", "ARI": "NL", "ATH": "AL", "ATL": "NL", "BAL": "AL", "BOS": "AL",
               "CHA": "AL", "CHN": "NL", "CIN": "NL", "CLE": "AL", "COL": "NL", "DET": "AL",
               "HOU": "AL", "KCA": "AL", "LAN": "NL", "MIA": "NL", "MIL": "NL", "MIN": "AL",
               "NYA": "AL", "NYN": "NL", "PHI": "NL", "PIT": "NL", "SDN": "NL", "SEA": "AL",
               "SFN": "NL", "SLN": "NL", "TBA": "AL", "TEX": "AL", "TOR": "AL", "WAS": "NL"}

# Each team's home ballpark IANA timezone - real, well-established public
# information (each team's home city), used only to convert a live game's
# real ESPN-synced UTC start time into local wall-clock time for a
# day/night classification. ATH covers both Oakland and Sacramento (its
# 2025+ interim home) since both are Pacific time.
TEAM_TIMEZONE = {
    "ANA": "America/Los_Angeles", "ARI": "America/Phoenix", "ATH": "America/Los_Angeles",
    "ATL": "America/New_York", "BAL": "America/New_York", "BOS": "America/New_York",
    "CHA": "America/Chicago", "CHN": "America/Chicago", "CIN": "America/New_York",
    "CLE": "America/New_York", "COL": "America/Denver", "DET": "America/New_York",
    "HOU": "America/Chicago", "KCA": "America/Chicago", "LAN": "America/Los_Angeles",
    "MIA": "America/New_York", "MIL": "America/Chicago", "MIN": "America/Chicago",
    "NYA": "America/New_York", "NYN": "America/New_York", "PHI": "America/New_York",
    "PIT": "America/New_York", "SDN": "America/Los_Angeles", "SEA": "America/Los_Angeles",
    "SFN": "America/Los_Angeles", "SLN": "America/Chicago", "TBA": "America/New_York",
    "TEX": "America/Chicago", "TOR": "America/New_York", "WAS": "America/New_York",
}

# Local start hour >= this counts as a night game. Standard baseball
# scheduling convention: day games cluster 12-3pm local, night games
# cluster 6-8pm local, with essentially nothing scheduled in between, so
# any reasonable cutoff in that gap is safe.
NIGHT_GAME_LOCAL_HOUR = 17


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (team, game) so rolling stats can be computed per-team."""
    cols = ["game_id", "date", "season", "home_win", "home_score", "away_score"]

    home = games[cols + ["home_franchise", "away_franchise"]].rename(columns={
        "home_franchise": "team", "away_franchise": "opponent",
        "home_win": "won", "home_score": "points_for", "away_score": "points_against",
    })
    home["is_home"] = True

    away = games[cols + ["away_franchise", "home_franchise"]].rename(columns={
        "away_franchise": "team", "home_franchise": "opponent",
        "home_win": "won", "home_score": "points_against", "away_score": "points_for",
    })
    away["won"] = 1 - away["won"]
    away["is_home"] = False

    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["team", "date"], kind="stable")


def _rolling_form(log: pd.DataFrame) -> pd.DataFrame:
    grp = log.groupby("team", group_keys=False)

    # shift(1) first so the current game is never included in its own rolling window
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
        ["game_id", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak"]
    ].rename(columns={
        "win_pct_l10": "home_win_pct_l10",
        "pf_l10": "home_pf_l10", "pa_l10": "home_pa_l10",
        "rest_days": "home_rest_days", "streak": "home_streak",
    })
    away_feats = log[~log["is_home"]][
        ["game_id", "win_pct_l10", "pf_l10", "pa_l10", "rest_days", "streak"]
    ].rename(columns={
        "win_pct_l10": "away_win_pct_l10",
        "pf_l10": "away_pf_l10", "pa_l10": "away_pa_l10",
        "rest_days": "away_rest_days", "streak": "away_streak",
    })

    out = games.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")
    out = out.merge(rating_history[["home_rating_pre", "away_rating_pre", "rating_diff_pre",
                                     "elo_home_win_prob"]], left_on="game_id", right_index=True, how="left")

    # first game of a team's history has no prior rest date -> fill with a neutral typical-rest value
    out["home_rest_days"] = out["home_rest_days"].fillna(LEAGUE_AVG_REST_DAYS)
    out["away_rest_days"] = out["away_rest_days"].fillna(LEAGUE_AVG_REST_DAYS)
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

    # Starting-pitcher rolling ER/start: home_sp_er_lN/away_sp_er_lN must
    # already be present on `games` (see module docstring - call
    # starting_pitcher.attach_starter_quality(games) beforehand). Positive
    # sp_er_diff_lN = home's starter has allowed FEWER earned runs per start
    # recently than away's -> favors home, same sign convention as rest_diff.
    if "home_sp_er_lN" in out.columns and "away_sp_er_lN" in out.columns:
        out["sp_er_diff_lN"] = out["away_sp_er_lN"] - out["home_sp_er_lN"]

    # Starting-pitcher rolling K-BB%: home_sp_kbb_pct_lN/away_sp_kbb_pct_lN
    # must already be present on `games` (see module docstring - call
    # starter_kbb_quality.attach_starter_kbb_pct(games) beforehand). K-BB% is
    # "higher is better", the opposite polarity from sp_er_lN's "lower is
    # better" - diff is home-minus-away here (not away-minus-home like
    # sp_er_diff_lN above) so positive still means "favors home" in both.
    if "home_sp_kbb_pct_lN" in out.columns and "away_sp_kbb_pct_lN" in out.columns:
        out["sp_kbb_pct_diff_lN"] = out["home_sp_kbb_pct_lN"] - out["away_sp_kbb_pct_lN"]

    out["market_fair_home_prob"] = market_implied_home_prob(out)

    return out


def market_implied_home_prob(games: pd.DataFrame) -> pd.Series:
    """Devigged home-moneyline probability from `Home Odds Close`/`Away Odds
    Close` (the 2012-2021 odds archive - see odds_loader.py; this must
    already be joined onto `games` before build_features is called, same
    ordering dependency starting_pitcher.attach_starter_quality has). 0.5
    ("no information") for the ~73% of 1990-2025 history outside that
    archive's coverage - the honest neutral fallback, not an invented value,
    same discipline as LEAGUE_AVG_SP_ER_PER_START. See this module's
    docstring for the hypothesis test that adopted this feature
    (research_moneyline_dog_calibration.py / decision_log.jsonl)."""
    has_ml = games["Home Odds Close"].notna() & games["Away Odds Close"].notna() if "Home Odds Close" in games.columns else pd.Series(False, index=games.index)
    out = pd.Series(0.5, index=games.index, dtype=float)
    if has_ml.any():
        fair = games.loc[has_ml].apply(
            lambda r: odds_math.devig_two_way(r["Home Odds Close"], r["Away Odds Close"]), axis=1
        )
        out.loc[has_ml] = [f[0] for f in fair]
    return out


def naive_score_features(home_pf, home_pa, away_pf, away_pa):
    """
    Naive "implied score" baseline: each team's expected runs = average of
    (its own scoring rate, its opponent's rate of allowing runs) - a
    standard, simple total/margin prior that gives the ML model a strong
    starting signal for scoring pace, which team strength alone (rating
    differential) doesn't capture. Identical formula to the NFL/CFB
    versions - sport-agnostic, just fed MLB-shaped (runs, not points)
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
    Indexed by canonical franchise. Not exercised by verify.py (out of scope
    for this backend-only pass) but kept for interface parity with
    sports/nfl/features.py and sports/cfb/features.py.
    """
    log = _team_game_log(games)
    grp = log.groupby("team", group_keys=False)
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
    return latest[["win_pct_l10", "pf_l10", "pa_l10", "streak", "date"]].rename(
        columns={"date": "last_game_date"}
    )


ML_FEATURE_COLS = [
    "rating_diff_pre", "rest_diff", "home_rest_days", "away_rest_days",
    "home_win_pct_l10", "away_win_pct_l10",
    "home_pf_l10", "home_pa_l10", "away_pf_l10", "away_pa_l10",
    "naive_total", "naive_margin",
    "home_streak", "away_streak", "is_interleague", "is_night",
    "home_sp_er_lN", "away_sp_er_lN", "sp_er_diff_lN",
    "home_sp_kbb_pct_lN", "away_sp_kbb_pct_lN", "sp_kbb_pct_diff_lN",
    "market_fair_home_prob",
]


def extra_matchup_features(home_fr, away_fr, game_date, home_row, away_row, **_ignored) -> dict:
    """
    Live-scoring counterpart to build_features' loader-derived sp_er_lN
    columns above. core/matchup.py (the generic scorer every live MLB pick
    goes through) calls this hook for a brand-new, not-yet-played game --
    which has no loader row to pull home_sp_er_lN/away_sp_er_lN from, since
    that requires starting_pitcher.attach_starter_quality(games), a
    dataset-wide batch join that only runs once, offline, during
    build_pipeline(). Without this hook, core/matchup.py's generic
    fill-anything-missing-with-a-default loop was silently filling both
    columns with 0.0 for every single live MLB prediction -- and 0.0 isn't
    a neutral "no signal" value here, it's a real, wrong one: it reads as
    "this starter has allowed zero earned runs across their last 8 starts,"
    an extreme outlier that basically never happens in the training data
    (league average is ~2.6). Fed to a HistGradientBoostingRegressor (which
    branches on exact thresholds, unlike a linear model that would just
    shrug off an off-distribution constant), that's a real, systematic bias
    on every live MLB pick, not just missing information. Confirmed by
    tracing core/matchup.py's fallback logic against this module's own
    ML_FEATURE_COLS.

    NOT fixed by actually computing each team's real confirmed starter's
    current rolling form live (that needs today's probable-pitcher
    identity, already surfaced for DISPLAY only via sports/mlb/probables.py,
    joined against that specific pitcher's own recent starts -- a real
    future enhancement, not attempted here). This hook instead falls back
    to the same LEAGUE_AVG_SP_ER_PER_START neutral value the historical
    pipeline already uses for any game outside its own event-file coverage
    -- the honest "no live signal available" value the model was actually
    trained to see for a missing starter, not an invented one.
    """
    from .starting_pitcher import LEAGUE_AVG_SP_ER_PER_START
    from .starter_kbb_quality import LEAGUE_AVG_SP_KBB_PCT
    row = {
        "home_sp_er_lN": LEAGUE_AVG_SP_ER_PER_START,
        "away_sp_er_lN": LEAGUE_AVG_SP_ER_PER_START,
        "sp_er_diff_lN": 0.0,
        # Same honest-neutral-fallback discipline as sp_er_lN above, and the
        # same real bug this hook exists to prevent (see this docstring) --
        # without it, core/matchup.py's generic fallback would silently fill
        # both with 0.0, which reads as "this starter has struck out every
        # batter he's ever walked zero of" (an extreme, wrong outlier), not
        # "no live signal available."
        "home_sp_kbb_pct_lN": LEAGUE_AVG_SP_KBB_PCT,
        "away_sp_kbb_pct_lN": LEAGUE_AVG_SP_KBB_PCT,
        "sp_kbb_pct_diff_lN": 0.0,
    }

    # is_interleague: real, computable now, not a fallback -- both flagged
    # by core/matchup.py's missing-feature warning as silently defaulting
    # to False for every live MLB pick (see that warning's own message).
    # TEAM_LEAGUE is derived EMPIRICALLY from the real historical dataset
    # (games.groupby("home_franchise")["home_league"].last()), not typed
    # from memory or general knowledge -- specifically because two teams
    # really have changed leagues within this dataset's own window (HOU
    # NL->AL in 2013, MIL AL->NL in 1998) and a memorized/guessed table
    # risks getting exactly those two wrong. Verified: exactly a 15/15 AL/NL
    # split, and taking each team's most recent recorded league correctly
    # resolves both real transitions to their CURRENT league.
    home_league = TEAM_LEAGUE.get(home_fr)
    away_league = TEAM_LEAGUE.get(away_fr)
    row["is_interleague"] = int(bool(home_league) and bool(away_league) and home_league != away_league)

    # is_night: the game's REAL scheduled start time (ESPN-synced, passed in
    # as `game_date`) converted to the home park's real local time, not a
    # fabricated value -- classified using a standard baseball scheduling
    # convention (day games cluster 12-3pm local, night games 6-8pm local,
    # essentially nothing in between, so any reasonable cutoff in that gap
    # is safe). TEAM_TIMEZONE is each team's real home city's IANA timezone
    # (well-established, low-ambiguity public information, unlike
    # TEAM_LEAGUE above) -- DST-aware via zoneinfo, not a fixed UTC offset.
    # Falls back to False (day) only if the home team isn't in the table at
    # all, which shouldn't happen for any real MLB franchise.
    tz_name = TEAM_TIMEZONE.get(home_fr)
    if tz_name is not None and game_date is not None:
        from zoneinfo import ZoneInfo
        gd = game_date
        if gd.tzinfo is None:
            gd = gd.tz_localize("UTC")
        local_dt = gd.tz_convert(ZoneInfo(tz_name))
        row["is_night"] = int(local_dt.hour >= NIGHT_GAME_LOCAL_HOUR)
    else:
        row["is_night"] = 0

    # market_fair_home_prob: this hook has no access to the live market odds
    # core/matchup.py's score_matchup() already holds for this exact game
    # (it's only merged into the model row AFTER build_matchup_feature_row()
    # returns, for the edge computation, not before it for prediction -- see
    # that module's docstring). Wiring the REAL live odds into this feature
    # would need market_odds threaded through build_matchup_feature_row()'s
    # signature, a core/matchup.py change with blast radius across every
    # other sport that shares it -- out of scope for this MLB-only pass.
    # Falls back to 0.5 ("no information"), the SAME neutral value the
    # historical pipeline uses for any game outside the 2012-2021 odds
    # archive's coverage -- avoids exactly the bug already caught twice in
    # this file's history (sp_er_lN/is_interleague silently defaulting to an
    # extreme, wrong value via core/matchup.py's generic 0.0/False
    # fallback). A real, scoped next step, not attempted here.
    row["market_fair_home_prob"] = 0.5

    return row
