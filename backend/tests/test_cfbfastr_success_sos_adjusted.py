"""
Unit tests for sports/cfb/cfbfastr_features.py's opponent-adjusted (SOS)
trailing success-rate functions -- build_trailing_success_features_sos_adjusted()
and get_current_trailing_success_sos_adjusted() -- the direct CFB transfer
of NFL's opponent-adjustment method (see decision_log.jsonl's
"nfl_nflverse_trailing_epa_sos_adjustment" entry and
docs/model_improvement_candidates_2026-09-15.md's "Cross-sport structural
ideas" section, which flags this as the cleanest transfer candidate).

Mirrors tests/test_nflfastr_epa_sos_adjusted.py's structure and hand-derived
fixture almost exactly (same relative deltas, shifted by +0.5 so the
success-rate values look like plausible 0-1 rates) -- see that file's own
module docstring for the general shape being mirrored here.

Two teams (Ohio State, Michigan), playing each other 3 times (g0-g2), with
hand-derived expected values -- see the inline comment on the first test for
the full arithmetic. A 4th game (g3) against an unmapped/absent opponent
tests the one real CFB-specific difference from NFL: an opponent this
module could not map falls back to a no-op (league-average) adjustment for
that one game, rather than the team's own game being dropped from its
trailing window the way a plain inner self-join (NFL's approach) would.
"""

import pandas as pd
import pytest

from sports.cfb import cfbfastr_features as cf


@pytest.fixture
def synthetic_two_team_csv(tmp_path, monkeypatch):
    rows = [
        # game_id, team, off_success_rate, def_success_rate_allowed, season
        ("g0", "Ohio State", 0.60, 0.48),
        ("g0", "Michigan", 0.55, 0.50),
        ("g1", "Ohio State", 0.70, 0.47),
        ("g1", "Michigan", 0.58, 0.49),
        ("g2", "Ohio State", 0.80, 0.45),
        ("g2", "Michigan", 0.62, 0.52),
        # g3: Ohio State's 4th game, opponent NOT in this data at all
        # (simulates an unmapped/uncovered opponent -- see module docstring's
        # ~72% coverage limitation) -- only one row for this game_id.
        ("g3", "Ohio State", 0.75, 0.20),
    ]
    df = pd.DataFrame([{
        "game_id": g, "season": 2024, "team": team, "is_home": True,
        "off_success_rate": off_sr, "off_plays": 65,
        "def_success_rate_allowed": def_sr, "def_plays": 65,
    } for g, team, off_sr, def_sr in rows])
    path = tmp_path / "success.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(cf, "DATA_PATH", path)
    cf._team_name_cache = None
    cf._success_by_franchise_cache = None
    cf._adjusted_per_game_cache = None
    yield path
    cf._team_name_cache = None
    cf._success_by_franchise_cache = None
    cf._adjusted_per_game_cache = None


def _base_games():
    return pd.DataFrame([
        {"game_id": "vg0", "season": 2024, "home_franchise": "Ohio State Buckeyes",
         "away_franchise": "Michigan Wolverines", "date": pd.Timestamp("2024-09-01")},
        {"game_id": "vg1", "season": 2024, "home_franchise": "Ohio State Buckeyes",
         "away_franchise": "Michigan Wolverines", "date": pd.Timestamp("2024-09-08")},
        {"game_id": "vg2", "season": 2024, "home_franchise": "Ohio State Buckeyes",
         "away_franchise": "Michigan Wolverines", "date": pd.Timestamp("2024-09-15")},
    ])


class TestBuildTrailingSuccessFeaturesSosAdjusted:
    def test_hand_derived_adjustment_matches_exactly(self, synthetic_two_team_csv):
        # league_avg(def_success_rate_allowed) is computed over ALL mapped-
        # franchise rows in the fetched data (both teams' real games, 7
        # rows total here -- OSU's g0-g3, Michigan's g0-g2: [0.48, 0.47,
        # 0.45, 0.20, 0.50, 0.49, 0.52]), the same "real data mean" every
        # other trailing feature's fallback uses, not scoped to just the
        # games in this test's `games` frame -- sum 3.11 / 7 = 0.4442857...
        #
        # OSU's g0: opponent Michigan has NO trailing defense yet (its
        #   first game) -> falls back to league_avg, so the adjustment
        #   term cancels: adj = raw(0.60) - 0.4442857 + 0.4442857 = 0.60.
        # OSU's g1: opponent Michigan's trailing def entering g1 = mean of
        #   Michigan's prior def values (just g0's 0.50) = 0.50.
        #   adj = raw(0.70) - 0.50 + 0.4442857 = 0.6442857.
        # This function assigns EACH row the trailing value ENTERING that
        # specific already-recorded game (shift(1), same discipline the
        # raw build_trailing_success_features() test uses) -- so OSU's g2
        # gets the mean of STRICTLY PRIOR adjusted games only (g0, g1 --
        # never g2 itself): mean(0.60, 0.6442857) = 0.6221429.
        out = cf.build_trailing_success_features_sos_adjusted(_base_games(), n_games=10)
        assert out["home_off_success_rate_trail_sos"].iloc[2] == pytest.approx(0.6221428571, abs=1e-6)

    def test_unmapped_opponent_falls_back_to_no_op_adjustment_not_a_dropped_game(self, synthetic_two_team_csv):
        # OSU's g3 (opponent absent from this data entirely -- the real
        # ~72%-coverage gap this module's docstring documents) must still
        # be a real past game for OSU's OWN trailing window: its
        # adjustment collapses to raw (opp lookup falls back to
        # league_avg for both sides of the +/- so they cancel: adj_g3 =
        # 0.75), not silently vanish the way a plain inner self-join
        # (NFL's approach) would drop it.
        # OSU's own real per-game adjusted values, in order:
        #   g0 = 0.60, g1 = 0.6442857,
        #   g2 = raw(0.80) - trail_def_michigan_entering_g2 + league_avg
        #      = 0.80 - mean(0.50, 0.49) + 0.4442857 = 0.80 - 0.495 + 0.4442857
        #      = 0.7492857,
        #   g3 = 0.75 (unmodified, unmapped opponent).
        # "Current form" (n_games=10, takes all 4) = mean of those four =
        # 2.7435714 / 4 = 0.6858929.
        cf.build_trailing_success_features_sos_adjusted(_base_games(), n_games=10)
        result = cf.get_current_trailing_success_sos_adjusted("Ohio State Buckeyes", n_games=10)
        assert result["off_success_rate_trail_sos"] == pytest.approx(0.6858928571, abs=1e-6)

    def test_never_drops_or_reorders_existing_rows(self, synthetic_two_team_csv):
        games = pd.DataFrame([
            {"game_id": "vg0", "season": 2024, "home_franchise": "Ohio State Buckeyes",
             "away_franchise": "Michigan Wolverines", "date": pd.Timestamp("2024-09-01"), "extra_col": "keep me"},
            {"game_id": "vg1", "season": 2024, "home_franchise": "Michigan Wolverines",
             "away_franchise": "Ohio State Buckeyes", "date": pd.Timestamp("2024-09-08"), "extra_col": "keep me too"},
        ])
        out = cf.build_trailing_success_features_sos_adjusted(games, n_games=10)
        assert len(out) == 2
        assert list(out["game_id"]) == ["vg0", "vg1"]
        assert list(out["extra_col"]) == ["keep me", "keep me too"]

    def test_produces_the_four_expected_sos_columns(self, synthetic_two_team_csv):
        out = cf.build_trailing_success_features_sos_adjusted(_base_games(), n_games=10)
        for side in ("home", "away"):
            for col in cf.TRAILING_COLS:
                assert f"{side}_{col}_trail_sos" in out.columns

    def test_no_nulls_in_output(self, synthetic_two_team_csv):
        out = cf.build_trailing_success_features_sos_adjusted(_base_games(), n_games=10)
        sos_cols = [f"{side}_{col}_trail_sos" for side in ("home", "away") for col in cf.TRAILING_COLS]
        assert not out[sos_cols].isna().any().any()


class TestGetCurrentTrailingSuccessSosAdjusted:
    def test_raises_clear_error_before_any_build_call(self, synthetic_two_team_csv):
        with pytest.raises(RuntimeError):
            cf.get_current_trailing_success_sos_adjusted("Ohio State Buckeyes")

    def test_matches_the_same_hand_derived_value_as_training(self, synthetic_two_team_csv):
        # "Current form" is the mean of the last n ADJUSTED per-game
        # values -- same relationship get_current_trailing_success() has
        # to build_trailing_success_features(). Restricting to n_games=3
        # takes OSU's 3 MOST RECENT real games (g1, g2, g3 -- see the
        # unmapped-opponent test above for each one's adj value):
        # mean(0.6442857, 0.7492857, 0.75) = 2.1435714 / 3 = 0.7145238.
        cf.build_trailing_success_features_sos_adjusted(_base_games(), n_games=10)
        result = cf.get_current_trailing_success_sos_adjusted("Ohio State Buckeyes", n_games=3)
        assert result["off_success_rate_trail_sos"] == pytest.approx(0.7145238095, abs=1e-6)

    def test_unknown_franchise_gets_league_average_not_a_crash(self, synthetic_two_team_csv):
        cf.build_trailing_success_features_sos_adjusted(_base_games(), n_games=10)
        result = cf.get_current_trailing_success_sos_adjusted("Some Team Not In The Data")
        assert set(result.keys()) == {f"{c}_trail_sos" for c in cf.TRAILING_COLS}
        for v in result.values():
            assert isinstance(v, float)

    def test_cache_is_recomputed_every_build_call_not_frozen(self, synthetic_two_team_csv):
        # Real bug class this guards against (see nflfastr_features.
        # reset_caches()'s docstring for NFL's version of this exact bug):
        # CFB's own convention is to always re-read/re-cache on every real
        # build call, not populate-once -- calling build twice in a row
        # must not raise or silently reuse a stale table missing g3.
        cf.build_trailing_success_features_sos_adjusted(_base_games(), n_games=10)
        cf.build_trailing_success_features_sos_adjusted(_base_games(), n_games=10)
        result = cf.get_current_trailing_success_sos_adjusted("Ohio State Buckeyes", n_games=10)
        assert result["off_success_rate_trail_sos"] == pytest.approx(0.6858928571, abs=1e-6)
