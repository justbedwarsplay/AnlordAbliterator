# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Real end-to-end test for the BenchmarkScore scorer: HFLM wiring, the
lm-eval harness call, and metric extraction — on the tiny model with a
sample-limited real benchmark (piqa). Requires network access for the
dataset download.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_native_parity import TINY_MODEL, _ensure_tiny_model  # noqa: E402

_ensure_tiny_model()

pytest.importorskip("lm_eval")


@pytest.fixture(scope="module")
def benchmark_evaluator(tmp_path_factory):
    from anlord.native.config import (
        DatasetSpecification,
        NativeConfig,
        QuantizationMethod,
        RowNormalization,
    )
    from anlord.native.evaluator import Evaluator
    from anlord.native.model import Model

    tmp_path = tmp_path_factory.mktemp("benchmark_prompts")
    good_file = tmp_path / "good.txt"
    good_file.write_text("Good one\nGood two\nGood three\nGood four\n", encoding="utf-8")
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("Bad one\nBad two\nBad three\nBad four\n", encoding="utf-8")

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
        scorers=[
            {
                "plugin": "anlord.scorers.benchmark_score.BenchmarkScore",
                "optimization": "maximize",
            }
        ],
        scorer_settings={
            "BenchmarkScore": {
                "score_name": "PIQA acc_norm",
                "task": "piqa",
                "metric": "acc_norm,none",
                "limit": 20,
            }
        },
        good_prompts=DatasetSpecification(dataset=str(good_file), split="[:4]"),
        bad_prompts=DatasetSpecification(dataset=str(bad_file), split="[:4]"),
    )
    model = Model(cfg)
    ev = Evaluator(cfg, model)
    yield ev, model
    del ev, model


def test_benchmark_score_objective_plumbing(benchmark_evaluator):
    ev, _model = benchmark_evaluator
    # BenchmarkScore is the only objective and it maximizes.
    assert ev.get_objective_names() == ["PIQA acc_norm"]

    from optuna.study import StudyDirection

    assert ev.get_objective_directions() == [StudyDirection.MAXIMIZE]
    # It is a reproducible built-in.
    assert ev.all_scorers_reproducible() is True
    assert ev.all_scorers_builtin() is True


def test_benchmark_score_real_lm_eval_run(benchmark_evaluator):
    ev, _model = benchmark_evaluator
    scores = ev.get_scores()
    assert [name for name, _ in scores] == ["PIQA acc_norm"]
    score = scores[0][1]
    # Real harness result: a probability-like accuracy in [0, 1].
    assert 0.0 <= score.value <= 1.0
    assert score.md_display  # "0.xxxx"
    assert float(score.md_display) == pytest.approx(score.value, abs=1e-9)
    # Objective values align with the score.
    assert ev.get_objective_values(scores) == (score.value,)


def test_benchmark_score_limit_is_recorded(benchmark_evaluator):
    ev, _model = benchmark_evaluator
    entry = ev._scorer_entries[0]
    assert entry.scorer.settings.limit == 20
