# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for 1.5.0 features: architecture profiles and manual trial selection."""

import json

import pytest

from anlord.config import Settings
from anlord.native.config import NativeConfig
from anlord.profiles import (
    BASE_DEFAULTS,
    apply_architecture_profile,
    match_profile,
)

# ---------------------------------------------------------------------------
# Architecture profiles
# ---------------------------------------------------------------------------


def test_gemma_profile_detected_and_applied():
    settings = Settings()
    assert settings.abliteration_max_weight_limit is None  # auto
    profile = apply_architecture_profile(settings, "gemma3_text")
    assert profile is not None and profile.label == "gemma"
    assert settings.abliteration_max_weight_limit == 2.0


def test_gemma_profile_prefix_matches_all_gemma_types():
    for model_type in ("gemma", "gemma2", "gemma3", "gemma3_text"):
        assert match_profile(model_type) is not None


def test_qwen_profile_is_explicit_noop():
    settings = Settings()
    profile = apply_architecture_profile(settings, "qwen3")
    assert profile is not None and profile.label == "qwen"
    assert profile.overrides == {}
    assert settings.abliteration_max_weight_limit is None


def test_unknown_architecture_gets_no_profile():
    settings = Settings()
    assert apply_architecture_profile(settings, "llama") is None
    assert apply_architecture_profile(settings, "lfm2") is None
    assert settings.abliteration_max_weight_limit is None


def test_explicit_value_beats_profile():
    settings = Settings()
    settings.abliteration_max_weight_limit = 1.75
    profile = apply_architecture_profile(settings, "gemma3_text")
    assert profile is not None
    assert settings.abliteration_max_weight_limit == 1.75


def test_reproduction_mode_skips_profiles():
    settings = Settings()
    settings.reproduce = "reproduce/reproduce.json"
    assert apply_architecture_profile(settings, "gemma3_text") is None
    assert settings.abliteration_max_weight_limit is None


def test_native_config_resolves_auto_to_base():
    """No profile applied: NativeConfig falls back to the Qwen-tuned base."""
    settings = Settings()
    config = NativeConfig.from_anlord_settings(settings)
    assert config.max_weight_limit == BASE_DEFAULTS["abliteration_max_weight_limit"]


def test_native_config_uses_profile_value():
    settings = Settings()
    apply_architecture_profile(settings, "gemma3_text")
    config = NativeConfig.from_anlord_settings(settings)
    assert config.max_weight_limit == 2.0


def test_detect_model_type_from_local_config(tmp_path):
    from anlord.profiles import detect_model_type

    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"]}),
        encoding="utf-8",
    )
    assert detect_model_type(str(model_dir)) == "gemma3_text"


def test_detect_model_type_failure_returns_none(tmp_path):
    from anlord.profiles import detect_model_type

    assert detect_model_type(str(tmp_path / "does-not-exist")) is None


def test_settings_validation_still_rejects_bad_limits():
    with pytest.raises(ValueError, match="max_weight_limit"):
        Settings(abliteration_max_weight_limit=0.5)
    with pytest.raises(ValueError, match="trial_select_top"):
        Settings(abliteration_trial_select_top=-1)


def test_settings_record_explicit_limit_from_cli(tmp_path):
    """The CLI passes --max-weight-limit through; an explicit value must reach
    NativeConfig unchanged (profiles only fill the auto case)."""
    from anlord.cli.main import create_parser

    parser = create_parser()
    args = parser.parse_args(
        ["--model", "test/model", "--output", str(tmp_path), "--max-weight-limit", "2.5"]
    )
    assert args.max_weight_limit == 2.5
    settings = Settings(
        model=args.model,
        abliteration_max_weight_limit=getattr(args, "max_weight_limit", None),
    )
    assert settings.abliteration_max_weight_limit == 2.5
    profile = apply_architecture_profile(settings, "gemma3_text")
    assert profile is not None
    assert settings.abliteration_max_weight_limit == 2.5  # kept


# ---------------------------------------------------------------------------
# Manual trial selection + ceiling warning
# ---------------------------------------------------------------------------


def _make_study_with_trials():
    """A real multi-objective in-memory Optuna study with measurable trials."""
    import optuna
    from optuna.trial import create_trial

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["minimize", "minimize"])
    # (refusals, kl): rank order by refusals; the Pareto front is 0 and 2.
    specs = [
        {"refusals": 36, "kl": 0.1744},  # best refusals overall (Pareto)
        {"refusals": 50, "kl": 0.0593},  # best KL (Pareto)
        {"refusals": 42, "kl": 0.1017},  # Pareto
        {"refusals": 50, "kl": 0.2000},  # dominated by the first trial
    ]
    for i, spec in enumerate(specs):
        study.add_trial(
            create_trial(
                values=[spec["refusals"] / 100, spec["kl"]],
                user_attrs={
                    "index": i + 1,
                    "refusals": spec["refusals"],
                    "kl_divergence": spec["kl"],
                    "n_bad_prompts": 100,
                    "parameters": {
                        "mlp.down_proj": {
                            "max_weight": 1.494 if i == 0 else 1.0,
                            "max_weight_position": 13.0,
                            "min_weight": 0.7,
                            "min_weight_distance": 9.0,
                        }
                    },
                },
            )
        )
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    best_trials = sorted(study.best_trials, key=lambda tr: tuple(tr.values or ()))
    return best_trials, completed


def _make_abliter(tmp_path):
    from anlord.native.abliterator import NativeAbliterator

    config = NativeConfig(
        study_checkpoint_dir=str(tmp_path / "checkpoints"),
        trial_selection_topn=0,
    )
    return NativeAbliterator(config)


def test_select_trial_automatic_when_disabled(tmp_path):
    best_trials, completed = _make_study_with_trials()
    abliter = _make_abliter(tmp_path)
    assert abliter._select_trial(best_trials, completed) is best_trials[0]


def test_select_trial_manual_pick_dominated_low_kl(tmp_path):
    best_trials, completed = _make_study_with_trials()
    abliter = _make_abliter(tmp_path)
    abliter.config.trial_selection_topn = 4

    def ask(choices, default_rank):
        assert len(choices) == 4
        # The auto pick must be offered and marked in the rendered lines.
        assert any("[auto]" in line for line in choices)
        assert any("[Pareto]" in line for line in choices)
        return 4  # the dominated trial (50 refusals, KL 0.2)

    chosen = abliter._select_trial(best_trials, completed, ask=ask)
    assert chosen.user_attrs["refusals"] == 50
    assert chosen.user_attrs["kl_divergence"] == pytest.approx(0.2)


def test_select_trial_falls_back_to_auto_without_choice(tmp_path):
    best_trials, completed = _make_study_with_trials()
    abliter = _make_abliter(tmp_path)
    abliter.config.trial_selection_topn = 3
    assert (
        abliter._select_trial(best_trials, completed, ask=lambda c, d: None)
        is best_trials[0]
    )


def test_select_trial_single_candidate_uses_auto(tmp_path):
    best_trials, completed = _make_study_with_trials()
    abliter = _make_abliter(tmp_path)
    abliter.config.trial_selection_topn = 3
    only = [completed[0]]
    assert abliter._select_trial(best_trials, only, ask=lambda c, d: 1) is best_trials[0]


def test_select_trial_without_refusal_metrics_uses_auto(tmp_path):
    import optuna
    from optuna.trial import create_trial

    best_trials, _ = _make_study_with_trials()
    abliter = _make_abliter(tmp_path)
    abliter.config.trial_selection_topn = 3
    empty = optuna.create_study(directions=["minimize", "minimize"])
    empty.add_trial(create_trial(values=[0.3, 0.1], user_attrs={"index": 9}))
    trials = [t for t in empty.trials if t.state.name == "COMPLETE"]
    assert abliter._select_trial(best_trials, trials, ask=lambda c, d: 1) is best_trials[0]


def test_ceiling_warning_fires_for_saturated_chosen(tmp_path, capsys):
    best_trials, completed = _make_study_with_trials()
    abliter = _make_abliter(tmp_path)
    abliter.config.max_weight_limit = 1.5
    chosen = next(t for t in completed if t.user_attrs["refusals"] == 36)
    abliter._warn_bound_saturation(chosen, best_trials)
    out = capsys.readouterr().out
    assert "search ceiling (1.5)" in out
    assert "--max-weight-limit" in out
    assert "1/3 Pareto trials" in out


def test_ceiling_warning_silent_when_below_limit(tmp_path, capsys):
    best_trials, completed = _make_study_with_trials()
    abliter = _make_abliter(tmp_path)
    abliter.config.max_weight_limit = 2.5
    chosen = next(t for t in completed if t.user_attrs["refusals"] == 36)
    abliter._warn_bound_saturation(chosen, best_trials)
    assert "ceiling" not in capsys.readouterr().out


def test_trial_selection_config_wiring(tmp_path):
    """Settings.abliteration_trial_select_top must reach the native config."""
    settings = Settings()
    settings.abliteration_trial_select_top = 5
    config = NativeConfig.from_anlord_settings(settings)
    assert config.trial_selection_topn == 5
    assert NativeConfig.from_anlord_settings(Settings()).trial_selection_topn == 0
