"""
Unit tests for sports/mlb/defensive_efficiency.py -- hand-derived expected
values from a small synthetic Retrosheet event file, not just "it ran"
(see tests/test_nflfastr_epa_sos_adjusted.py for the style this follows).

Two synthetic games, written in real Retrosheet event-file shape (id/info/
play rows -- no start/sub rows needed, since DER is attributed via each
play's own vh_flag, not pitcher identity):

Game 1 (2020/07/28, number 0): home=AAA, away=BBB.
  vh=0 (BBB batting -> AAA fielding): "S8" (hit), "63" (fielding out)
    -> AAA fielding this game: bip=2, hits=1
  vh=1 (AAA batting -> BBB fielding): "D7" (hit), "8" (fielding out), "W" (walk, not BIP)
    -> BBB fielding this game: bip=2, hits=1
  Also includes "SB2" (vh=0) and "CS2(24)" (vh=1) -- real Retrosheet
  baserunning-event shapes that must NOT be counted as a ball in play at
  all (see defensive_efficiency.py's module docstring on why prefix,
  not exact, matching is required for these).

Game 2 (2020/07/29, number 0): home=AAA, away=CCC.
  vh=0 (CCC batting -> AAA fielding): "43" (out), "43" (out), "S9" (hit)
    -> AAA fielding this game: bip=3, hits=1

Hand-derived rolling DER for AAA (the team appearing in both games, as
the fielding/home side both times), shift(1)-then-rolling(10, min_periods=1):
  Game 1: AAA has no prior game at all -> NaN -> LEAGUE_AVG_DER fallback.
  Game 2: AAA's prior game (game 1) fielding tally: bip=2, hits=1
    -> der = 1 - 1/2 = 0.5
"""

import csv

import pandas as pd
import pytest

from sports.mlb import defensive_efficiency as de


GAME1_ROWS = [
    ["id", "AAA202007280"],
    ["info", "visteam", "BBB"],
    ["info", "hometeam", "AAA"],
    ["info", "date", "2020/07/28"],
    ["info", "number", "0"],
    ["play", "1", "0", "bbb001", "00", "X", "S8"],       # BBB batting -> AAA fielding: hit
    ["play", "1", "0", "bbb002", "00", "X", "63"],        # BBB batting -> AAA fielding: out
    ["play", "1", "0", "bbb002", "00", "X", "SB2"],       # baserunning only -- must be excluded entirely
    ["play", "1", "1", "aaa001", "00", "X", "D7"],        # AAA batting -> BBB fielding: hit
    ["play", "1", "1", "aaa002", "00", "X", "8"],         # AAA batting -> BBB fielding: out
    ["play", "1", "1", "aaa003", "00", "X", "W"],         # walk -- not a fielding chance
    ["play", "1", "1", "aaa003", "00", "X", "CS2(24)"],   # baserunning only -- must be excluded entirely
]

GAME2_ROWS = [
    ["id", "AAA202007290"],
    ["info", "visteam", "CCC"],
    ["info", "hometeam", "AAA"],
    ["info", "date", "2020/07/29"],
    ["info", "number", "0"],
    ["play", "1", "0", "ccc001", "00", "X", "43"],        # CCC batting -> AAA fielding: out
    ["play", "1", "0", "ccc002", "00", "X", "43"],        # out
    ["play", "1", "0", "ccc003", "00", "X", "S9"],        # hit
]


@pytest.fixture
def synthetic_event_file(tmp_path, monkeypatch):
    sub_dir = tmp_path / "2010seve"
    sub_dir.mkdir()
    path = sub_dir / "AAA2020.EVN"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        for row in GAME1_ROWS + GAME2_ROWS:
            writer.writerow(row)

    import sports.mlb.config as config_mod
    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path)
    de.reset_cache()
    yield path
    de.reset_cache()


class TestBaseCodeClassification:
    def test_hits_are_recognized(self):
        assert de._classify_base_code("S8") == "hit"
        assert de._classify_base_code("D7") == "hit"
        assert de._classify_base_code("T9") == "hit"
        assert de._classify_base_code("DGR") == "hit"
        assert de._classify_base_code("DGR9") == "hit"

    def test_non_bip_pa_codes_are_not_hits_or_bip(self):
        assert de._classify_base_code("K") == "non_bip"
        assert de._classify_base_code("K23") == "non_bip"     # suffixed strikeout variant
        assert de._classify_base_code("W") == "non_bip"
        assert de._classify_base_code("IW") == "non_bip"
        assert de._classify_base_code("HP") == "non_bip"
        assert de._classify_base_code("HR") == "non_bip"

    def test_baserunning_only_codes_excluded_even_when_suffixed(self):
        # The real bug this module's classification exists to avoid: these
        # carry base-number/parenthetical suffixes in the real data, so an
        # exact-string check (as starter_kbb_quality.py's NON_PA_BASE_CODES
        # uses) would miss them and wrongly count them as a fielded ball.
        assert de._classify_base_code("SB2") == "non_pa"
        assert de._classify_base_code("CS2(24)") == "non_pa"
        assert de._classify_base_code("CSH(242)") == "non_pa"
        assert de._classify_base_code("POCS1(1)") == "non_pa"

    def test_fielding_outs_and_error_fc_are_non_hit_bip(self):
        assert de._classify_base_code("63") == "bip_out"
        assert de._classify_base_code("8") == "bip_out"
        assert de._classify_base_code("E5") == "bip_out"
        assert de._classify_base_code("FC6") == "bip_out"


class TestLoadTeamGameDer:
    def test_hand_derived_bip_and_hit_counts_per_game(self, synthetic_event_file):
        der_games = de.load_team_game_der()
        assert len(der_games) == 2

        g1 = der_games[der_games["date"] == pd.Timestamp("2020-07-28")].iloc[0]
        assert g1["home_franchise"] == "AAA"
        assert g1["away_franchise"] == "BBB"
        assert g1["home_bip"] == 2 and g1["home_hits"] == 1   # S8 (hit) + 63 (out); SB2 excluded
        assert g1["away_bip"] == 2 and g1["away_hits"] == 1   # D7 (hit) + 8 (out); W, CS2(24) excluded

        g2 = der_games[der_games["date"] == pd.Timestamp("2020-07-29")].iloc[0]
        assert g2["home_bip"] == 3 and g2["home_hits"] == 1   # 43, 43 (outs) + S9 (hit)


class TestAttachDefensiveEfficiency:
    def test_hand_derived_rolling_der_matches_exactly(self, synthetic_event_file):
        games = pd.DataFrame([
            {"game_id": "vg1", "date": pd.Timestamp("2020-07-28"), "game_number": "0",
             "home_franchise": "AAA", "away_franchise": "BBB"},
            {"game_id": "vg2", "date": pd.Timestamp("2020-07-29"), "game_number": "0",
             "home_franchise": "AAA", "away_franchise": "CCC"},
        ])
        out = de.attach_defensive_efficiency(games)
        by_id = out.set_index("game_id")

        # Game 1: AAA has no prior game at all -> league-average fallback.
        assert by_id.loc["vg1", "home_der_lN"] == pytest.approx(de.LEAGUE_AVG_DER, abs=1e-9)
        # Game 2: AAA's rolling DER = 1 - (1 hit / 2 BIP) from game 1 = 0.5.
        assert by_id.loc["vg2", "home_der_lN"] == pytest.approx(0.5, abs=1e-9)
        assert by_id.loc["vg2", "der_diff_lN"] == pytest.approx(
            by_id.loc["vg2", "home_der_lN"] - by_id.loc["vg2", "away_der_lN"], abs=1e-9
        )

    def test_never_drops_or_reorders_existing_rows(self, synthetic_event_file):
        games = pd.DataFrame([
            {"game_id": "vg1", "date": pd.Timestamp("2020-07-28"), "game_number": "0",
             "home_franchise": "AAA", "away_franchise": "BBB", "extra_col": "keep0"},
            {"game_id": "vg2", "date": pd.Timestamp("2020-07-29"), "game_number": "0",
             "home_franchise": "AAA", "away_franchise": "CCC", "extra_col": "keep1"},
        ])
        out = de.attach_defensive_efficiency(games)
        assert len(out) == 2
        assert list(out["game_id"]) == ["vg1", "vg2"]
        assert list(out["extra_col"]) == ["keep0", "keep1"]

    def test_uncovered_game_gets_league_average_fallback(self, synthetic_event_file):
        games = pd.DataFrame([{
            "game_id": "vg_uncovered", "date": pd.Timestamp("1995-05-01"), "game_number": "0",
            "home_franchise": "ZZZ", "away_franchise": "YYY",
        }])
        out = de.attach_defensive_efficiency(games)
        assert out.loc[0, "home_der_lN"] == de.LEAGUE_AVG_DER
        assert out.loc[0, "away_der_lN"] == de.LEAGUE_AVG_DER
        assert out.loc[0, "der_diff_lN"] == 0.0


class TestGetCurrentDer:
    def test_matches_the_same_hand_derived_value_as_training(self, synthetic_event_file):
        # "Current form" includes the MOST RECENT game (not shifted) --
        # AAA's own tally across both its games: bip=2+3=5, hits=1+1=2
        # -> der = 1 - 2/5 = 0.6
        result = de.get_current_der("AAA")
        assert result == pytest.approx(1.0 - 2 / 5, abs=1e-9)

    def test_uses_cache(self, synthetic_event_file, monkeypatch):
        first = de.get_current_der("AAA")
        # corrupt the source so a cache miss would raise/produce garbage
        monkeypatch.setattr(de, "load_team_game_der", lambda: (_ for _ in ()).throw(AssertionError("cache miss")))
        second = de.get_current_der("AAA")
        assert second == pytest.approx(first, abs=1e-9)

    def test_unknown_team_gets_league_average_not_a_crash(self, synthetic_event_file):
        assert de.get_current_der("Some Team Not In The Data") == de.LEAGUE_AVG_DER
