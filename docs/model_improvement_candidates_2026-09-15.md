# Model Improvement Candidates — 2026-09-15

Research/ideation pass only. No code was modified, no pipeline was rebuilt, nothing here is
implemented. Written after reading the full `decision_log.jsonl` (122 entries), `core/research.py`,
`core/factor_taxonomy.py`, every sport's `features.py`, the three most-recently-built NFL/CFB feature
modules, and `core/ensemble.py`. Facts I could directly verify (by reading this repo's own code/data,
or via a live WebSearch) are marked **[confirmed]**. Everything else is my own externally-motivated
reasoning, marked **[reasoning]** — treat those as hypotheses to run through `core/research.py`'s
actual discipline, not conclusions.

Ground rule this whole document follows, inherited from `core/research.py`'s own docstring: a feature
idea only counts as a real candidate here if it has an external, domain-level reason to matter — not
"more data can't hurt." Several plausible-sounding ideas I considered and dropped are noted briefly
where relevant, so the absence isn't mistaken for an oversight.

---

## 0. Two structural facts worth stating up front

- **[confirmed]** `core/factor_taxonomy.py`'s `REGISTRY` only has an `"NFL"` entry. MLB/CFB/NBA/NHL have
  *zero* factor-taxonomy coverage — not "thin," literally absent. `coverage_report()` can't tell you
  where CFB or NBA is structurally weak today because nobody's populated their registries yet. This is
  cheap to fix (just transcribing each sport's own `ML_FEATURE_COLS` into `Factor(...)` entries, using
  the categorization logic already established for NFL) and would make every gap in this document
  mechanically re-discoverable instead of resting on a one-time manual read. Not a "model improvement"
  by itself, but the highest-leverage piece of due diligence not yet done.
- **[confirmed]** NHL already has a large, real MoneyPuck skater/goalie dataset sitting locally at
  `Datasets/NHL/2008_to_2024 copy 2.csv` (+ `2025 copy.csv`), 2008–2024, already join-solved against
  this project's own `game_id` via `sports/nhl/research_player_matchup.py`'s merge_asof method. I
  checked its header directly — it carries real per-shot **expected-goals (xG)** columns
  (`I_F_xGoals`, `OnIce_F_xGoals`, `OnIce_A_xGoals`, high/medium/low-danger xG splits, flurry- and
  score-adjusted variants). This is a big deal for section 1 below: it means NHL's single best
  next feature needs **zero new data acquisition** — the file is already on disk.

---

## 1. Per-sport feature gaps (MLB, NBA, NHL)

### MLB

**(a) Ballpark scoring-environment factor ("park factor")**
- **[reasoning]** Park factors are one of the oldest, most universally accepted adjustments in
  sabermetrics (Bill James popularized the idea; every modern site — FanGraphs, Baseball Reference,
  Statcast — publishes one). Two teams can have identical trailing scoring stats while playing in
  parks with genuinely different real run environments (Coors Field vs. a pitcher's park), and nothing
  in the current MLB feature set adjusts for that at all.
- **[confirmed]** This project's own MLB loader already carries a `park_id` column — decision log entry
  `travel_fatigue_short_rest_park_change` (2026-08-12) explicitly used it. That means a park factor
  needs **no external data source at all**: it's a straightforward empirical computation from games
  already loaded (each park's trailing/season average total runs relative to league average, the same
  "compute a trailing rate, shift/recenter" pattern every other feature in this project already uses).
- **Feasibility**: very high. Pure pandas, no new dependency, no license risk, no compiled-code wall.
  The closest thing to a blocker is making sure the park-id-to-venue mapping is stable across
  relocations (same discipline `TEAM_LEAGUE`/`TEAM_TIMEZONE` in `sports/mlb/features.py` already
  applies for AL/NL and timezone).

**(b) Team defensive efficiency, independent of the starter**
- **[reasoning]** The project's own `starter_kbb_pct_rolling` docstring (adopted, `sports/mlb/features.py`)
  explicitly names this as an *unaddressed* gap in its own reasoning: earned-runs-allowed and even
  K-BB% still can't separate a pitcher's own skill from "his defense turned balls in play into outs
  for him." Defensive Efficiency Ratio (DER = 1 − (balls in play that became hits) / (total balls in
  play)), a real, decades-old sabermetric team defense stat, is exactly the complementary signal — it
  measures the defense's own contribution, which nothing in `ML_FEATURE_COLS` currently isolates.
- **[confirmed]** The Retrosheet event files already parsed for `starting_pitcher.py`,
  `starter_kbb_quality.py`, and `research_bullpen_arm_quality.py` carry the batted-ball/out-type
  information needed to compute team-level DER directly — this is the **same data source already in
  the pipeline**, just a different aggregation, not a new acquisition.
- **Feasibility**: high, similar effort to the already-adopted K-BB% feature (same file family, same
  team already did the harder starter/reliever attribution work this would reuse).

**(c) Statcast-era batted-ball quality (barrel rate, exit velocity)**
- **[reasoning]** Real, modern sabermetric standard (Statcast's "barrel" classification correlates more
  tightly with sustained offensive/pitching skill than outcome stats like batting average).
- **[confirmed via WebSearch]** `pybaseball` is a real, pip-installable, actively maintained Python
  package (`pip install pybaseball`) that wraps Baseball Savant/Statcast, Baseball Reference, and
  FanGraphs scraping — pure Python + requests/pandas, no native compiler dependency (unlike the xgboost
  wall that blocked CFB's EPA this session).
- **Feasibility caveat**: Statcast only exists from **2015 onward** — a hard coverage ceiling, the same
  shape of problem NHL Edge hit this session (5-of-22 seasons real coverage, need an honest league-avg
  fallback for everything earlier, and an era-restricted re-test discipline per the NHL Edge decision
  log entries). Lower priority than (a)/(b) for that reason — real signal, but a smaller fraction of
  this project's 1990–2025 history would ever see a non-fallback value.

### NBA

**(a) The missing fourth Dean Oliver factor: free-throw rate**
- **[confirmed]** Dean Oliver's "Four Factors" (eFG%, TOV%, OREB%, FT rate) is the explicit,
  externally-cited framework two hypothesis tests already used this project (`four_factors_efg_tov`,
  `four_factors_oreb_pct`, both `adopt_cautiously`). Checking `sports/nba/features.py`'s `ML_FEATURE_COLS`
  directly: eFG%/TOV% and OREB% are covered; **FT rate — the fourth published factor — has never been
  tested**, confirmed by grep against `decision_log.jsonl` (no `ft_rate`/`free_throw` hypothesis
  anywhere).
- **Feasibility**: very high — same box-score dataset (`Datasets/NBA/nba-box-scores.csv`) already used
  for the other three factors and for `research_starters_out.py`. Cheapest possible next NBA test:
  literally completing a framework this project already adopted 3/4 of.

**(b) Pace-adjusted offensive/defensive rating**
- **[reasoning]** `home_pf_l10`/`home_pa_l10` are raw points, which conflate true offensive/defensive
  quality with a team's pace (a fast team scores and allows more points per game without being better
  or worse per possession) — this is the NBA-specific version of exactly the conflation problem
  `nfl_nflverse_trailing_epa_features`'s reasoning diagnosed for NFL (`pf_l10`/`pa_l10` "conflate real
  offensive/defensive quality with special-teams TDs... garbage-time scoring"). Points-per-100-
  possessions (ORTG/DRTG) is the standard NBA-analytics answer, used everywhere from Basketball
  Reference to broadcast graphics.
- **[confirmed via WebSearch]** `nba_api` is a real, MIT-licensed, pip-installable package
  (`pip install nba_api`) wrapping the official stats.nba.com endpoints, including team pace/rating
  endpoints (`leaguedashteamstats`).
- **Feasibility caveat**: real risk of the same WAF/User-Agent brittleness this project already hit
  with ESPN's feed (decision log entry 1) — stats.nba.com is known in the wider community for
  aggressive rate-limiting/blocking of non-browser clients. Worth a small pilot fetch before committing,
  not a blocker in principle. Also somewhat correlated with the Four-Factors features already adopted
  (both target "efficiency, not just points"), so likely smaller marginal fit gain than (a).

**(c) Same schedule-blindness gap as every non-NFL sport** — see section 2.

### NHL

**(a) Team-level expected goals (xG), from the MoneyPuck data already on disk**
- **[reasoning]** xG is the direct successor to the Corsi/Fenwick shot-differential family this project
  already adopted (`nhl_trailing_shot_diff_and_pp_rate`, real margin_corr gain) — modern hockey
  analytics (MoneyPuck, Evolving-Hockey, Natural Stat Trick) treats raw shot attempts as a first-
  generation possession proxy and xG (shot attempts weighted by real historical conversion rate given
  location/type/situation) as the more precise descendant, specifically because it separates shot
  *volume* from shot *quality* — two teams with identical shot differentials can have very different
  real scoring-chance quality.
- **[confirmed]** As noted in section 0: the exact data needed (`I_F_xGoals`/`OnIce_F_xGoals`/
  `OnIce_A_xGoals`, per player per game, situation-tagged) is **already downloaded** to
  `Datasets/NHL/2008_to_2024 copy 2.csv` / `2025 copy.csv`, and the game-id join (MoneyPuck's `gameId`
  → this project's `game_id`, via home/away team + date with a 1-day merge_asof tolerance) is already
  built and verified in `sports/nhl/research_player_matchup.py`. Building a *team-level* trailing xG
  feature means aggregating on-ice xG across all skaters for a team-game (summing `OnIce_F_xGoals`/
  `OnIce_A_xGoals` for `situation=="all"` rows) — a real aggregation step, but on data and a join that
  are both already solved, unlike every other candidate in this document.
- **Feasibility**: very high — no new dependency, no license question, no API key, no rate limit, no
  coverage ceiling (2008–2024 covers the large majority of this project's 2004–2026 NHL window, far
  better than NHL Edge's 5-season cap that already forced an era-restricted re-test this session).
  This is the strongest single candidate in this whole document — see the final ranking.

**(b) Same schedule-blindness gap as every non-NFL sport** — see section 2.

**(c) A note on what NOT to re-try**: NHL Edge tracking (skating speed, shot-location save%) was
already tried twice this session (full-history and era-restricted) and the era-restricted re-test
specifically *rejected* skating speed and left high-danger save% a clean null. I don't see a
well-motivated reformulation of either worth re-queuing right now — the MoneyPuck xG feature above is a
better-evidenced use of research time than a third pass at Edge.

---

## 2. Cross-sport structural ideas

**Opponent-adjustment (SOS) transfers directly, and is currently missing everywhere except NFL.**

`nfl_nflverse_trailing_epa_sos_adjustment` (adopted this session, decision log 2026-09-15) is a real,
validated win: subtract the opponent's own trailing allowed-rate at the time of each past game,
re-center to league average, then roll the adjusted per-game values — confirmed stable independently
across *both* halves of a season split (unlike the QB-continuity feature tested the same session, which
looked good in aggregate but failed exactly that check).

**[confirmed by reading each sport's `features.py`]**: none of MLB, NBA, NHL, or CFB apply any opponent
adjustment to their trailing form features today —

- MLB's `home_pf_l10`/`home_pa_l10`, `sp_er_lN`, `sp_kbb_pct_lN` are all raw trailing means, schedule-blind.
- NBA's `pf_l10`/`pa_l10` and the adopted Four-Factors features are raw trailing means, schedule-blind.
- NHL's `shot_diff_l10`, `pp_pct_l10`, `pk_pct_l10` are raw trailing means/ratios, schedule-blind.
- CFB's newly-adopted `off_success_rate_trail`/`def_success_rate_allowed_trail` (this session,
  `cfbfastr_features.py`) are raw trailing means — the exact same shape NFL's raw EPA trailing feature
  was in *before* the SOS-adjustment pass, and CFB arguably needs this more than any other sport, given
  how lopsided real strength-of-schedule is across FBS conferences (the same real signal that motivated
  this session's CFB low-history rating floor).

**[reasoning]** The method is sport-agnostic by construction — it only needs "a trailing rate stat for
team X" and "the opponent's own trailing rate on the complementary stat," both of which every sport
above already has. This is the single most directly transferable idea in this document: it's proven
methodology (not a hypothesis about what *might* work), applied to data these sports already have (not
a new acquisition), on a bug this project's own decision log just fixed for one sport this exact
session and can mechanically fix for four more.

**One caveat**: MLB/NHL/NBA's box-score-derived pf_l10/pa_l10 are noisier / more luck-contaminated than
NFL's real play-level EPA (see MLB's own DER gap above) — SOS-adjusting a noisy raw stat still leaves it
noisy, just schedule-fair. CFB is the cleanest transfer candidate (its `success_rate_trail` is already
the same "modern, play-level, EPA-adjacent" quality tier NFL's was) and MLB/NBA/NHL's raw pf_l10/pa_l10
are the weakest candidates for this specific technique — better spent on the sport-specific gaps in
section 1 first (park factor, DER, xG), with SOS-adjustment applied once those cleaner signals exist.

---

## 3. Confidence-label gaps

Per `core/ensemble.py`'s own docstrings, as of this session:

| Sport | Market | Status |
|---|---|---|
| NBA | moneyline | **Backwards, unresolved.** 3 mechanisms tried (submodel-agreement, market-agreement, edge-magnitude) and rejected — market-agreement flipped sign between season halves; edge-magnitude's best bucket had z~1.55. Gated to `Unvalidated`. |
| NBA | spread | **Backwards, unresolved.** Edge-magnitude non-monotonic; home/away directionally consistent but z~1.15; favorite/underdog directionally consistent (z~1.34) but never cleared the adoption bar. Gated to `Unvalidated`. |
| MLB | spread | **Backwards, unresolved.** Market-(dis)agreement shows a real hit-rate gap but ~zero ROI gap (pure odds-asymmetry); edge-magnitude/home-away/fav-dog all null. Gated to `Unvalidated`. |
| CFB | spread | **Backwards, unresolved — but for a *data*, not modeling, reason.** CFB spread odds are 100% synthetic (`market_fair_prob` exactly 0.500, constant -110) — there is no real two-sided price to build any market-relative mechanism against. Gated to `Unvalidated`. |
| CFB | total | **Gated to `Unvalidated` 2026-09-14**, precautionarily, on `watch`-level (not adopt-level) evidence: live-population subgroup test found z=-1.90 (needs ~3.2), stable direction, shrinkage weight 0.54. App owner chose to gate now rather than wait for more data. **Not yet reflected in `ensemble.py`'s own docstrings** — a small documentation-debt item worth a one-line addition next time that file is touched, since the module's own "unvalidated_confidence_tier" docstring still lists only the original four.|

**Live-data `watch` items that are candidates for the exact re-investigation that fixed NFL/MLB
moneyline and total this session, once more live samples accumulate:**

- `nba_spread_favorite_underdog_subgroup` (2026-09-03): z=+1.34, stable direction, shrunk estimate
  -0.60pp — directionally the same NBA spread signal checked 3 separate times and never quite cleared
  the bar. Worth a periodic re-check as `n` grows (the Bonferroni bar itself also *rises* as more tests
  run project-wide, so this isn't guaranteed to clear on volume alone — worth tracking both together).
- `mlb_spread_confidence_backwards_live` (2026-09-14): z=-1.73 on live forward-test data (n=99/112),
  stable direction, shrinkage weight 43%. **Worth flagging a real ambiguity I found while reading**: this
  entry's own reasoning states MLB spread "has never had a dedicated confidence-tier fix, just the old
  generic submodel-agreement `confidence_tier()`" — but `ensemble.py`'s `unvalidated_confidence_tier()`
  docstring and `edge_finder.py`'s `SPREAD_UNVALIDATED_SPORTS` (per the 2026-09-03 decision log entry)
  already include MLB. Either the live-audit picks in that check predate the 09-03 `Unvalidated` gate
  (the same "stale pending picks carry the old label" staleness class already found and fixed for CFB
  spread, decision log entry 91) or something regressed. Worth a direct one-line check
  (`SELECT DISTINCT confidence FROM forward_picks WHERE sport='MLB' AND market='spread' AND
  snapshotted_at > '2026-09-03'`) before spending any real investigation time on a new MLB-spread
  mechanism — chasing a signal that's actually a stale-label artifact would repeat the exact mistake
  CFB spread's parlay-poisoning incident already taught this project to check for first.
- `home_side_outperforms_away_side_cross_sport` (2026-09-14): z=+1.42, stable direction across CFB
  exclusion, shrunk estimate -6.97pp. A real, externally-motivated mechanism (home-field-advantage
  market inefficiency is well documented in betting literature) worth re-checking as the live sample
  grows, same "watch, don't force" treatment already given it.

---

## 4. What a CFBD API key unlocks

`CollegeFootballData.com`'s free tier (email signup, 1000 calls/month) was investigated twice this
session without ever obtaining a key (sandboxed environment, confirmed via a real unauthenticated
request returning 401). Two concrete things become newly testable once the app owner obtains one:

1. **Real, non-degenerate CFB spread/total odds**, via the `/lines` endpoint's `spread_open`/
   `over_under_open` fields (confirmed present in the schema, population *not* verified). This is the
   direct fix for CFB spread's `Unvalidated` gate above — that gate exists specifically because no real
   two-sided spread price exists in the current data; a real market price would let
   `spread_market_disagreement_confidence_tier`-style mechanisms be tried on CFB for the first time.
2. **Real CFB CLV** (open-vs-close movement) — CFB is currently one of the sports where `clv_pct` is
   structurally unmeasurable (see the 2026-09-03 CLV feasibility investigation), same root cause as (1).

Both are conditional on the *populated* fields actually differing from the flat lines already in this
project's own dataset — not yet confirmed, flagged honestly as such in the original investigation.

---

## 5. If I were prioritizing this list

1. **NHL team-level xG from the already-downloaded MoneyPuck file.** Highest score on both axes: real,
   externally-validated signal (xG is the direct modern successor to the Corsi/Fenwick signal that
   *already* produced a real, adopted fit gain in this exact sport) × essentially the lowest possible
   implementation cost in this document (data already on disk, join already solved, pure pandas, no
   new dependency, no API key, no rate limit, coverage spans nearly this project's entire NHL history
   unlike NHL Edge's 5-season ceiling that already forced an era-restricted re-test this session).
2. **MLB ballpark scoring-environment factor.** Near-zero cost (uses a column — `park_id` — already
   flowing through the pipeline for an unrelated feature), directly targeted at a gap the project's own
   K-BB% docstring explicitly names as unaddressed, no external dependency or license question at all.
3. **SOS/opponent-adjustment transfer, applied to CFB's `success_rate_trail` first.** Proven methodology
   (this exact technique just cleared a real, split-half-stable `adopt` for NFL this session) applied to
   the cleanest available non-NFL candidate — CFB's success-rate feature is the same "modern, play-level"
   quality tier NFL's raw EPA trailing feature was before its own SOS-adjustment, and CFB's real
   cross-conference strength disparity is exactly the kind of case this method is built for.
4. **MLB team defensive efficiency (DER) from the Retrosheet files already parsed.** Well-motivated
   (isolates defense from pitching, a gap explicitly named in this project's own recent work), reuses
   data-parsing infrastructure this session already built for the starter/bullpen features, so the
   marginal engineering lift is genuinely small.
5. **NBA free-throw rate — the missing fourth Dean Oliver factor.** Smallest lift of anything in this
   document (same box-score file already used for the other three factors), and it's the direct
   completion of a framework this project has already partially adopted twice — a cheap, well-motivated
   way to close out an already-open thread rather than start a new one.

**Notably NOT in the top 5, and why:** MLB Statcast/barrel-rate and NBA pace/ORTG-DRTG both have real
external support but real friction — Statcast's 2015+ coverage ceiling mirrors the NHL Edge problem this
project already hit and had to special-case; `nba_api`/stats.nba.com carries a realistic risk of the
same WAF/rate-limit friction ESPN's feed already presented (decision log entry 1), unverified until a
real pilot fetch is tried. The CFBD-key items are real and valuable but are gated on an action only the
app owner can take (getting the key), not on any research or engineering decision — worth doing the
moment the key exists, not rankable against the others until then.
