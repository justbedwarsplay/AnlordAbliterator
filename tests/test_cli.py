# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for command-line argument handling."""

from anlord.cli.main import create_parser, run_from_args


def test_skip_benchmarks_flag_is_available():
    args = create_parser().parse_args(["--model", "example/model", "--skip-benchmarks"])

    assert run_from_args(args).skip_benchmarks is True


def test_interactive_none_skips_all_benchmarks():
    from anlord.cli.prompts import InteractivePrompter

    prompter = InteractivePrompter()
    prompter.use_fallback = True
    prompter._input_fallback = lambda prompt, default="": "none"

    assert prompter.prompt_benchmarks() == []


def test_no_resume_flag_is_a_real_boolean_option():
    args = create_parser().parse_args(["--model", "example/model", "--no-resume"])

    assert run_from_args(args).resume is False


def test_cli_uses_default_benchmarks_and_quick_limit():
    args = create_parser().parse_args(["--model", "example/model"])
    settings = run_from_args(args)

    # Default benchmarks are the 6 classic tasks when no --tasks is given
    assert len(settings.benchmarks) == 6
    assert set(settings.benchmarks) == {"mmlu", "gsm8k", "hellaswag", "arc_challenge", "winogrande", "truthfulqa"}
    assert settings.limit == 100
    assert settings.prefetch_model is True
    assert settings.quantization == "auto"
    assert settings.abliteration_timeout == 43200
    assert settings.abliteration_backend == "native"


def test_model_prefetch_can_be_disabled():
    args = create_parser().parse_args(["--model", "example/model", "--no-prefetch-model"])

    assert run_from_args(args).prefetch_model is False


def test_full_mode_has_no_implicit_limit():
    args = create_parser().parse_args(
        ["--model", "example/model", "--mode", "full", "--limit", "10"]
    )

    assert run_from_args(args).limit is None


def test_abliteration_reuse_ignores_benchmark_limit(tmp_path):
    from anlord.config import Settings
    from anlord.evaluation.evaluator import AbliterationResult, EvaluationResult
    from anlord.pipeline import AbliterationPipeline

    settings = Settings(
        model="ornith-ai/Ornith-1.5-9B",
        output_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        limit=100,
        dtype="bfloat16",
        device="cuda",
        quantization="bnb_4bit",
        seed=42,
    )
    pipeline = object.__new__(AbliterationPipeline)
    pipeline.settings = settings
    existing = EvaluationResult(
        model_id="ornith-ai/Ornith-1.5-9B",
        evaluation_type="baseline",
        abliteration=AbliterationResult(model_id="ornith-ai/Ornith-1.5-9B", initial_refusals=90),
        config={
            "benchmarks": settings.benchmarks,
            "num_fewshot": 0,
            "limit": None,
            "dtype": "bfloat16",
            "device": "cuda",
            "batch_size": 1,
            "seed": 42,
            "quantization": "bnb_4bit",
            "model_commit": None,
            "cache_dir": str((tmp_path / "cache").resolve()),
        },
    )

    assert existing.abliteration_succeeded()
    assert pipeline._baseline_abliteration_matches(existing)
    assert pipeline._baseline_matches_current_run(existing) is False
    assert pipeline._abliteration_result_to_reuse(existing) is existing.abliteration


def test_abliteration_reuse_reads_standalone_metrics_file(tmp_path):
    from anlord.config import Settings
    from anlord.pipeline import AbliterationPipeline

    settings = Settings(
        model="ornith-ai/Ornith-1.5-9B",
        output_dir=tmp_path,
        cache_dir=tmp_path / "cache",
    )
    metrics_dir = settings.get_models_dir() / "abliteration_baseline"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "abliteration_evaluation.json").write_text(
        '{"model": "ornith-ai/Ornith-1.5-9B", "initial_refusals": 88, "final_refusals": 88, "total_prompts": 100}',
        encoding="utf-8",
    )
    pipeline = object.__new__(AbliterationPipeline)
    pipeline.settings = settings

    reused = pipeline._abliteration_result_to_reuse(None)

    assert reused is not None
    assert reused.initial_refusals == 88
