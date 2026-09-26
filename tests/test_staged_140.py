# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for 1.4.0 features: staged search, silhouette-guided bounds, subset cache."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_native_parity import TINY_MODEL, _ensure_tiny_model  # noqa: E402

_ensure_tiny_model()


def _silhouette_position_low(silh):
    from anlord.native.abliterator import _silhouette_position_low

    return _silhouette_position_low(silh)


def test_silhouette_position_low_function():
    # Meaningful separation starts at the layer scoring >= half of the best.
    assert _silhouette_position_low([0.1, 0.2, 0.9, 0.88]) == 3.0
    assert _silhouette_position_low([0.9, 0.2, 0.1]) == 1.0
    # All-zero silhouettes: no meaningful layer -> layer 1.
    assert _silhouette_position_low([0.0, 0.0]) == 1.0


def test_staged_run_end_to_end(tmp_path):
    """Staged search: stage 1 = attention-only trials (MLP frozen at identity),
    stage 2 = MLP sampling on top of the frozen stage-1 attention winner."""
    from anlord.native.abliterator import NativeAbliterator
    from anlord.native.config import (
        DatasetSpecification,
        NativeConfig,
        QuantizationMethod,
        RowNormalization,
    )

    good_file = tmp_path / "good.txt"
    bad_file = tmp_path / "bad.txt"
    good_file.write_text("\n".join(f"Good prompt {i}" for i in range(6)) + "\n", encoding="utf-8")
    bad_file.write_text("\n".join(f"Bad request {i}" for i in range(6)) + "\n", encoding="utf-8")

    cfg = NativeConfig(
        model=str(TINY_MODEL.resolve()),
        dtypes=["float32"],
        quantization=QuantizationMethod.NONE,
        device_map="cpu",
        batch_size=2,
        row_normalization=RowNormalization.NONE,
        offload_outputs_to_cpu=False,
        seed=42,
        response_prefix="",
        n_trials=8,
        n_startup_trials=2,
        staged_search=True,
        staged_stage1_fraction=0.5,
        evaluation_pruning=False,
        search_seeds=False,
        good_prompts=DatasetSpecification(dataset=str(good_file), split="[:6]"),
        bad_prompts=DatasetSpecification(dataset=str(bad_file), split="[:6]"),
        study_checkpoint_dir=str(tmp_path / "checkpoints"),
        scorer_settings={
            "KeywordRate": {"prompts": {"dataset": str(bad_file), "split": "[:6]"}},
            "KLDivergence": {"prompts": {"dataset": str(good_file), "split": "[:6]"}},
        },
    )
    result = NativeAbliterator(cfg).run(tmp_path / "model_out")
    assert result.error is None

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    journal = list((tmp_path / "checkpoints").glob("*.jsonl"))[0]
    study = optuna.load_study(
        study_name="abliteration",
        storage=JournalStorage(JournalFileBackend(str(journal))),
    )
    trials = {t.user_attrs["index"]: t for t in study.trials if t.user_attrs.get("index")}

    stage1 = [t for i, t in sorted(trials.items()) if i <= 4]
    stage2 = [t for i, t in sorted(trials.items()) if i > 4]
    assert stage1 and stage2

    # Stage 1: attention-only — MLP frozen at identity (max_weight 0).
    for t in stage1:
        assert t.user_attrs["stage"] == "attn"
        assert t.user_attrs["parameters"]["mlp.down_proj"]["max_weight"] == 0.0

    # Stage 2: attention frozen at the stage-1 winner (within the rescale band).
    s1_best = min(stage1, key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]))
    s1_mw = s1_best.user_attrs["parameters"]["attn.o_proj"]["max_weight"]
    for t in stage2:
        assert t.user_attrs["stage"] == "mlp"
        rescale = t.user_attrs["attn_rescale"]
        assert 0.8 <= rescale <= 1.2
        got = t.user_attrs["parameters"]["attn.o_proj"]["max_weight"]
        assert got == pytest.approx(s1_mw * rescale, abs=1e-9)

    # MLP is genuinely sampled in stage 2 (0.0 is a legal sample — the
    # optimizer may switch the MLP off — but the values must vary).
    mlp_values = {
        t.user_attrs["parameters"]["mlp.down_proj"]["max_weight"] for t in stage2
    }
    assert len(mlp_values) > 1, mlp_values


def test_staged_auto_disable_with_single_group(tmp_path):
    """With attention-only components the staged search auto-disables."""
    from anlord.native.abliterator import NativeAbliterator
    from anlord.native.config import (
        DatasetSpecification,
        NativeConfig,
        QuantizationMethod,
        RowNormalization,
    )

    good_file = tmp_path / "good.txt"
    bad_file = tmp_path / "bad.txt"
    good_file.write_text("Good one\nGood two\n", encoding="utf-8")
    bad_file.write_text("Bad one\nBad two\n", encoding="utf-8")

    cfg = NativeConfig(
        model=str(TINY_MODEL.resolve()),
        dtypes=["float32"],
        quantization=QuantizationMethod.NONE,
        device_map="cpu",
        batch_size=2,
        row_normalization=RowNormalization.NONE,
        offload_outputs_to_cpu=False,
        seed=42,
        response_prefix="",
        n_trials=2,
        n_startup_trials=2,
        staged_search=True,
        abliteration_components=["attn"],
        good_prompts=DatasetSpecification(dataset=str(good_file), split="[:2]"),
        bad_prompts=DatasetSpecification(dataset=str(bad_file), split="[:2]"),
        study_checkpoint_dir=str(tmp_path / "checkpoints"),
        scorer_settings={
            "KeywordRate": {"prompts": {"dataset": str(bad_file), "split": "[:2]"}},
            "KLDivergence": {"prompts": {"dataset": str(good_file), "split": "[:2]"}},
        },
    )
    result = NativeAbliterator(cfg).run(tmp_path / "model_out")
    assert result.error is None
    # No stage attributes: the standard search ran.
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    journal = list((tmp_path / "checkpoints").glob("*.jsonl"))[0]
    study = optuna.load_study(
        study_name="abliteration",
        storage=JournalStorage(JournalFileBackend(str(journal))),
    )
    assert all("stage" not in t.user_attrs for t in study.trials)


def test_batch_cache_prefix_reuse(tmp_path):
    """The trial-scoped batch cache: prefix counts are exact prefixes of the
    full count, and the full evaluation reuses cached batches."""
    from anlord.native.config import (
        DatasetSpecification,
        NativeConfig,
        QuantizationMethod,
        RowNormalization,
    )
    from anlord.native.evaluator import Evaluator
    from anlord.native.model import Model

    good_file = tmp_path / "good.txt"
    bad_file = tmp_path / "bad.txt"
    good_file.write_text("Good one\nGood two\nGood three\nGood four\nGood five\nGood six\n", encoding="utf-8")
    bad_file.write_text(
        "Bad hacking one\nBad hacking two\nBad hacking three\n"
        "Bad hacking four\nBad hacking five\nBad hacking six\n",
        encoding="utf-8",
    )
    cfg = NativeConfig(
        model=str(TINY_MODEL.resolve()),
        dtypes=["float32"],
        quantization=QuantizationMethod.NONE,
        device_map="cpu",
        batch_size=2,
        row_normalization=RowNormalization.NONE,
        offload_outputs_to_cpu=False,
        seed=42,
        response_prefix="",
        good_prompts=DatasetSpecification(dataset=str(good_file), split="[:6]"),
        bad_prompts=DatasetSpecification(dataset=str(bad_file), split="[:6]"),
        scorer_settings={
            "KeywordRate": {"prompts": {"dataset": str(bad_file), "split": "[:6]"}},
            "KLDivergence": {"prompts": {"dataset": str(good_file), "split": "[:6]"}},
        },
    )
    model = Model(cfg)
    ev = Evaluator(cfg, model)

    # Prefix count is a lower bound of the full count.
    r2 = ev.quick_refusals(2)
    r4 = ev.quick_refusals(4)
    r6 = ev.quick_refusals(6)
    assert r2 is not None and r4 is not None and r6 is not None
    assert r2 <= r4 <= r6 <= 6

    # Batches are cached grid-aligned (bs=2 -> batches of 2 prompts).
    scorer = ev._scorer_entries[0].scorer
    assert set(scorer._batch_responses) == {0, 1, 2}

    # The full evaluation reuses the cached batches and yields the same count.
    full_count = ev.quick_refusals(6)
    assert full_count == r6
    scores = ev.get_scores()
    by_name = dict(scores)
    rate = by_name["Refusals"].value
    assert rate == pytest.approx(r6 / 6)
