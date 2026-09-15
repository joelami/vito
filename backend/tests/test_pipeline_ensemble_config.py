"""
Unit tests for pipeline.py's _sport_ensemble_config() -- the per-sport
Elo:ML blend-weight override mechanism adopted 2026-09-14 (see
core/research_ensemble_blend_weight.py and sports/nfl/config.py's
WEIGHT_ELO_SPREAD comment). The one property this must never regress:
a sport's config module with NO override attributes gets byte-identical
EnsembleConfig() defaults, exactly as if this mechanism didn't exist.
"""

from types import SimpleNamespace

from core import ensemble
from pipeline import _sport_ensemble_config


class TestSportEnsembleConfig:
    def test_a_sport_config_with_no_overrides_gets_the_plain_defaults(self):
        fake_config = SimpleNamespace()  # no WEIGHT_ELO_* attributes at all
        cfg = _sport_ensemble_config(fake_config)
        assert cfg == ensemble.EnsembleConfig()

    def test_a_single_override_only_changes_that_one_field(self):
        fake_config = SimpleNamespace(WEIGHT_ELO_SPREAD=0.3)
        cfg = _sport_ensemble_config(fake_config)
        assert cfg.weight_elo_spread == 0.3
        assert cfg.weight_elo_moneyline == 0.5
        assert cfg.weight_ml_total == 0.5

    def test_nfl_config_module_carries_the_real_adopted_spread_override(self):
        # Real, current wiring -- not a mock -- catches the override
        # silently disappearing from sports/nfl/config.py.
        from sports.nfl import config as nfl_config
        cfg = _sport_ensemble_config(nfl_config)
        assert cfg.weight_elo_spread == 0.3

    def test_cfb_config_module_has_no_override_and_stays_at_defaults(self):
        # Real, current wiring -- CFB's own blend-weight sweep found no
        # stable improvement on any market (see decision_log.jsonl), so
        # this must stay a plain default, not silently drift.
        from sports.cfb import config as cfb_config
        cfg = _sport_ensemble_config(cfb_config)
        assert cfg == ensemble.EnsembleConfig()
