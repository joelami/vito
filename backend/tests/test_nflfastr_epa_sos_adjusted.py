"""
Unit tests for sports/nfl/nflfastr_features.py's opponent-adjusted (SOS)
trailing EPA functions -- build_trailing_epa_features_sos_adjusted() and
get_current_trailing_epa_sos_adjusted() -- adopted 2026-09-15 (see
decision_log.jsonl's "nfl_nflverse_trailing_epa_sos_adjustment" entry).

Two teams (A, B), playing each other 3 times, with hand-derived expected
values -- see the module comment inline below for the full arithmetic.
Team B's own trailing defense evolves across its 3 games (0.00 -> -0.01
-> 0.02 raw def_epa_per_play_allowed, so its OWN trailing entering each
of those games is NaN/0.00/-0.005), which is what makes team A's
opponent-adjusted values differ game to game even though A always faces
"the same" opponent identity in this minimal fixture.
"""

import pandas as pd
import pytest

from sports.nfl import nflfastr_features as nf


@pytest.fixture
def synthetic_two_team_csv(tmp_path, monkeypatch):
    rows = [
        # game_id, team, off_epa, def_epa_allowed, date
        ("g0", "A", 0.10, -0.02, "2024-09-08"),
        ("g0", "B", 0.05, 0.00, "2024-09-08"),
        ("g1", "A", 0.20, -0.03, "2024-09-15"),
        ("g1", "B", 0.08, -0.01, "2024-09-15"),
        ("g2", "A", 0.30, -0.05, "2024-09-22"),
        ("g2", "B", 0.12, 0.02, "2024-09-22"),
    ]
    df = pd.DataFrame([{
        "game_id": g, "season": 2024, "team": team, "is_home": True,
        "off_epa_per_play": off_epa, "off_success_rate": 0.45,
        "off_plays": 60, "def_epa_per_play_allowed": def_epa, "def_success_rate_allowed": 0.45,
        "def_plays": 60, "gameday": date,
    } for g, team, off_epa, def_epa, date in rows])
    path = tmp_path / "epa.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(nf, "DATA_PATH", path)
    nf._team_game_epa_cache = None
    nf._adjusted_per_game_cache = None
    yield path
    nf._team_game_epa_cache = None
    nf._adjusted_per_game_cache = None


class TestBuildTrailingEpaFeaturesSosAdjusted:
    def test_hand_derived_adjustment_matches_exactly(self, synthetic_two_team_csv):
        # league_avg(def_epa_per_play_allowed) over all 6 rows
        # ([-0.02,-0.03,-0.05, 0.00,-0.01,0.02]) = -0.015.
        # A's g0: opponent B has NO trailing defense yet (B's first game)
        #   -> falls back to league_avg, so the adjustment term cancels:
        #   adj = raw(0.10) - (-0.015) + (-0.015) = 0.10.
        # A's g1: opponent B's trailing def entering g1 = mean of B's
        #   prior def values (just g0's 0.00) = 0.00.
        #   adj = raw(0.20) - 0.00 + (-0.015) = 0.185.
        # This function assigns EACH row the trailing value ENTERING that
        # specific already-recorded game (shift(1), same discipline
        # test_nflfastr_features.py's equivalent test uses) -- so A's g2
        # (2024-09-22) gets the mean of STRICTLY PRIOR adjusted games only
        # (g0, g1 -- never g2 itself): mean(0.10, 0.185) = 0.1425.
        games = pd.DataFrame([
            {"game_id": "vg0", "home_franchise": "Arizona Cardinals", "away_franchise": "Atlanta Falcons",
             "date": pd.Timestamp("2024-09-08")},
            {"game_id": "vg1", "home_franchise": "Arizona Cardinals", "away_franchise": "Atlanta Falcons",
             "date": pd.Timestamp("2024-09-15")},
            {"game_id": "vg2", "home_franchise": "Arizona Cardinals", "away_franchise": "Atlanta Falcons",
             "date": pd.Timestamp("2024-09-22")},
        ])
        monkeypatch_map = {"A": "Arizona Cardinals", "B": "Atlanta Falcons"}
        import sports.nfl.nflfastr_features as nf_mod
        orig_map = nf_mod.TEAM_CODE_TO_FRANCHISE
        nf_mod.TEAM_CODE_TO_FRANCHISE = {**orig_map, **monkeypatch_map}
        try:
            out = nf.build_trailing_epa_features_sos_adjusted(games, n_games=10)
        finally:
            nf_mod.TEAM_CODE_TO_FRANCHISE = orig_map
        assert out["home_off_epa_per_play_trail_sos"].iloc[2] == pytest.approx(0.1425, abs=1e-6)

    def test_never_drops_or_reorders_existing_rows(self, synthetic_two_team_csv):
        import sports.nfl.nflfastr_features as nf_mod
        orig_map = nf_mod.TEAM_CODE_TO_FRANCHISE
        nf_mod.TEAM_CODE_TO_FRANCHISE = {**orig_map, "A": "Arizona Cardinals", "B": "Atlanta Falcons"}
        try:
            games = pd.DataFrame([
                {"game_id": "vg0", "home_franchise": "Arizona Cardinals", "away_franchise": "Atlanta Falcons",
                 "date": pd.Timestamp("2024-09-08"), "extra_col": "keep me"},
                {"game_id": "vg1", "home_franchise": "Atlanta Falcons", "away_franchise": "Arizona Cardinals",
                 "date": pd.Timestamp("2024-09-15"), "extra_col": "keep me too"},
            ])
            out = nf.build_trailing_epa_features_sos_adjusted(games, n_games=10)
        finally:
            nf_mod.TEAM_CODE_TO_FRANCHISE = orig_map
        assert len(out) == 2
        assert list(out["game_id"]) == ["vg0", "vg1"]
        assert list(out["extra_col"]) == ["keep me", "keep me too"]

    def test_adds_exactly_the_eight_columns_ml_feature_cols_expects(self, synthetic_two_team_csv):
        from sports.nfl.features import ML_FEATURE_COLS
        import sports.nfl.nflfastr_features as nf_mod
        orig_map = nf_mod.TEAM_CODE_TO_FRANCHISE
        nf_mod.TEAM_CODE_TO_FRANCHISE = {**orig_map, "A": "Arizona Cardinals", "B": "Atlanta Falcons"}
        try:
            games = pd.DataFrame([{
                "game_id": "vg0", "home_franchise": "Arizona Cardinals", "away_franchise": "Atlanta Falcons",
                "date": pd.Timestamp("2024-09-08"),
            }])
            out = nf.build_trailing_epa_features_sos_adjusted(games, n_games=10)
        finally:
            nf_mod.TEAM_CODE_TO_FRANCHISE = orig_map
        sos_cols = [c for c in ML_FEATURE_COLS if c.endswith("_trail_sos")]
        assert len(sos_cols) == 8
        for col in sos_cols:
            assert col in out.columns


class TestGetCurrentTrailingEpaSosAdjusted:
    def test_matches_the_same_hand_derived_value_as_training(self, synthetic_two_team_csv):
        # "Current form" is the mean of the last n ADJUSTED per-game
        # values -- same relationship get_current_trailing_epa() has to
        # build_trailing_epa_features() -- so this should reproduce the
        # exact same 0.19166... derived above.
        import sports.nfl.nflfastr_features as nf_mod
        orig_map = nf_mod.TEAM_CODE_TO_FRANCHISE
        nf_mod.TEAM_CODE_TO_FRANCHISE = {**orig_map, "A": "Arizona Cardinals", "B": "Atlanta Falcons"}
        try:
            result = nf.get_current_trailing_epa_sos_adjusted("Arizona Cardinals", n_games=10)
        finally:
            nf_mod.TEAM_CODE_TO_FRANCHISE = orig_map
        assert result["off_epa_per_play_trail_sos"] == pytest.approx((0.10 + 0.185 + 0.29) / 3, abs=1e-6)

    def test_unknown_franchise_gets_league_average_not_a_crash(self, synthetic_two_team_csv):
        import sports.nfl.nflfastr_features as nf_mod
        orig_map = nf_mod.TEAM_CODE_TO_FRANCHISE
        nf_mod.TEAM_CODE_TO_FRANCHISE = {**orig_map, "A": "Arizona Cardinals", "B": "Atlanta Falcons"}
        try:
            result = nf.get_current_trailing_epa_sos_adjusted("Some Team Not In The Data")
        finally:
            nf_mod.TEAM_CODE_TO_FRANCHISE = orig_map
        assert set(result.keys()) == {f"{c}_trail_sos" for c in nf.TRAILING_COLS}
        for v in result.values():
            assert isinstance(v, float)
