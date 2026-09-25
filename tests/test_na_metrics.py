# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the "n/a" rendering of unmeasured refusal metrics (1.3.0)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_native_parity import TINY_MODEL, _ensure_tiny_model  # noqa: E402

_ensure_tiny_model()


def _kl_only_config(tmp_path: Path):
    from anlord.native.config import (
        DatasetSpecification,
        NativeConfig,
        QuantizationMethod,
        RowNormalization,
    )

    good_file = tmp_path / "good.txt"
    bad_file = tmp_path / "bad.txt"
    good_file.write_text("Good one\nGood two\nGood three\n", encoding="utf-8")
    bad_file.write_text("Bad one\nBad two\nBad three\n", encoding="utf-8")

    return NativeConfig(
        model=str(TINY_MODEL.resolve()),
        dtypes=["float32"],
        quantization=QuantizationMethod.NONE,
        device_map="cpu",
        batch_size=2,
        row_normalization=RowNormalization.NONE,
        offload_outputs_to_cpu=False,
        seed=42,
        response_prefix="",
        n_trials=1,
        n_startup_trials=1,
        evaluation_pruning=False,
        search_seeds=False,
        reproducibility_information="full",
        scorers=[
            {"plugin": "anlord.scorers.kl_divergence.KLDivergence", "optimization": "minimize"}
        ],
        good_prompts=DatasetSpecification(dataset=str(good_file), split="[:3]"),
        bad_prompts=DatasetSpecification(dataset=str(bad_file), split="[:3]"),
        scorer_settings={
            "KLDivergence": {"prompts": {"dataset": str(good_file), "split": "[:3]"}},
        },
        study_checkpoint_dir=str(tmp_path / "checkpoints"),
    )


@pytest.fixture(scope="module")
def kl_only_run(tmp_path_factory):
    from anlord.native.abliterator import NativeAbliterator

    tmp_path = tmp_path_factory.mktemp("kl_only")
    cfg = _kl_only_config(tmp_path)
    abliter = NativeAbliterator(cfg)
    result = abliter.run(tmp_path / "model_out")
    return tmp_path, result


def test_refusal_metrics_are_none_without_keyword_scorer(kl_only_run):
    tmp_path, result = kl_only_run
    assert result.error is None
    assert result.initial_refusals is None
    assert result.final_refusals is None
    assert result.kl_divergence is not None

    metrics = json.loads(
        (tmp_path / "model_out" / "native_abliteration_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["initial_refusals"] is None
    assert metrics["final_refusals"] is None
    assert metrics["total_prompts"] is None

    saved = json.loads(
        (tmp_path / "model_out" / "abliteration_reproduction.json").read_text(encoding="utf-8")
    )
    assert saved["metrics"]["initial_refusals"] is None
    assert saved["metrics"]["kl_divergence"] is not None


def test_reports_render_na_for_unknown_refusals(tmp_path):
    """CSV/HTML reports render "n/a" instead of fake zeros for unmeasured metrics."""
    from anlord.evaluation.comparison import compare_results
    from anlord.evaluation.evaluator import AbliterationResult, EvaluationResult
    from anlord.reports.formatters import format_csv
    from anlord.reports.generator import ReportGenerator

    baseline = EvaluationResult(
        model_id="m",
        evaluation_type="baseline",
        abliteration=AbliterationResult(model_id="m"),
    )
    abliterated = EvaluationResult(
        model_id="m",
        evaluation_type="abliterated",
        abliteration=AbliterationResult(model_id="m"),
    )
    comparison = compare_results(baseline, abliterated)

    csv_text = format_csv(comparison)
    assert "n/a" in csv_text
    # The fake "0 refusals" must not appear as a measured value.
    assert "Final Refusals,0,0" not in csv_text

    generator = ReportGenerator(tmp_path)
    html_text = generator._build_html(comparison)
    assert "n/a" in html_text
    csv_report = generator.generate_csv_report(comparison)
    assert "n/a" in csv_report.read_text(encoding="utf-8")


def test_abliteration_json_round_trip_preserves_none(tmp_path):
    """abliteration.json with null refusals loads back as None (not zeros)."""
    from types import SimpleNamespace

    from anlord.evaluation.evaluator import AbliterationResult
    from anlord.pipeline import AbliterationPipeline

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    payload = AbliterationResult(
        model_id="m",
        kl_divergence=0.0123,
    ).to_dict()
    (results_dir / "abliteration.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    holder = SimpleNamespace(settings=SimpleNamespace(get_results_dir=lambda: results_dir))
    loaded = AbliterationPipeline._load_abliteration_result(holder)
    assert loaded is not None
    assert loaded.initial_refusals is None
    assert loaded.final_refusals is None
    assert loaded.total_prompts is None
    assert loaded.kl_divergence == pytest.approx(0.0123)


def test_comparison_with_unknown_metrics_is_none_safe():
    """compare_results with unmeasured refusals must not crash or fake zeros."""
    from anlord.evaluation.comparison import compare_results
    from anlord.evaluation.evaluator import AbliterationResult, EvaluationResult

    baseline = EvaluationResult(
        model_id="m",
        evaluation_type="baseline",
        abliteration=AbliterationResult(model_id="m"),
    )
    abliterated = EvaluationResult(
        model_id="m",
        evaluation_type="abliterated",
        abliteration=AbliterationResult(model_id="m"),
    )
    comparison = compare_results(baseline, abliterated)
    assert comparison.initial_refusals_baseline is None
    assert comparison.final_refusals_abliterated is None
    assert comparison.refusal_reduction is None
    assert comparison.refusal_reduction_percent is None
