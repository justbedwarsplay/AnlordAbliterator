# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for 1.5.0 image abliteration: diffusers repos, assembly, text-encoder branch."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_native_parity import TINY_MODEL, _ensure_tiny_model

_ensure_tiny_model()

from anlord.image.diffusers import (
    assemble_pipeline_output,
    is_diffusers_repo,
    resolve_local_repo,
)

WEIGHT_FILES = ("model.safetensors", "config.json", "generation_config.json")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def _copy_existing(src: Path, dst: Path, names) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)


def _make_diffusers_repo(tmp_path: Path) -> Path:
    """A fake diffusers pipeline around the real tiny text encoder."""
    repo = tmp_path / "fake-t2i"
    repo.mkdir()
    (repo / "model_index.json").write_text(
        '{"_class_name": "TestPipeline",'
        ' "text_encoder": ["transformers", "LlamaForCausalLM"],'
        ' "tokenizer": ["transformers", "PreTrainedTokenizerFast"]}',
        encoding="utf-8",
    )
    _copy_existing(TINY_MODEL, repo / "text_encoder", WEIGHT_FILES)
    _copy_existing(TINY_MODEL, repo / "tokenizer", TOKENIZER_FILES)
    for component in ("transformer", "vae"):
        (repo / component).mkdir()
        (repo / component / "config.json").write_text("{}", encoding="utf-8")
        (repo / component / "model.safetensors").write_bytes(b"stub-weights")
    (repo / "scheduler").mkdir()
    (repo / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    return repo


def test_is_diffusers_repo_and_resolve(tmp_path):
    repo = _make_diffusers_repo(tmp_path)
    assert is_diffusers_repo(str(repo))
    assert resolve_local_repo(str(repo)) == repo
    # A plain causal-LM directory is not a diffusers pipeline.
    assert not is_diffusers_repo(str(TINY_MODEL))
    assert resolve_local_repo(str(TINY_MODEL)) is None


def test_is_model_directory_accepts_diffusers(tmp_path):
    from anlord.evaluation.evaluator import AbliterationResult

    repo = _make_diffusers_repo(tmp_path)
    assert AbliterationResult.is_model_directory(repo)
    # The marker without any weights anywhere must not validate.
    empty = tmp_path / "empty-pipeline"
    empty.mkdir()
    (empty / "model_index.json").write_text("{}", encoding="utf-8")
    assert not AbliterationResult.is_model_directory(empty)


def test_assemble_pipeline_output(tmp_path):
    repo = _make_diffusers_repo(tmp_path)
    pipeline_root = tmp_path / "out" / "pipeline"
    enc_out = pipeline_root / "text_encoder"
    enc_out.mkdir(parents=True)
    (enc_out / "model.safetensors").write_bytes(b"ablated-encoder-weights")
    (enc_out / "config.json").write_text("{}", encoding="utf-8")
    for artifact in ("abliteration_reproduction.json", "native_abliteration_metrics.json"):
        (enc_out / artifact).write_text("{}", encoding="utf-8")

    moved = assemble_pipeline_output(repo, pipeline_root)

    assert set(moved) == {"abliteration_reproduction.json", "native_abliteration_metrics.json"}
    assert (pipeline_root / "model_index.json").is_file()
    assert (pipeline_root / "transformer" / "model.safetensors").read_bytes() == b"stub-weights"
    assert (pipeline_root / "vae" / "config.json").is_file()
    assert (pipeline_root / "scheduler" / "scheduler_config.json").is_file()
    assert (pipeline_root / "tokenizer" / "tokenizer.json").is_file()
    # The ablated encoder stays in its subfolder; artifacts are moved to the root.
    assert (pipeline_root / "text_encoder" / "model.safetensors").read_bytes() == (
        b"ablated-encoder-weights"
    )
    assert not (pipeline_root / "text_encoder" / "native_abliteration_metrics.json").exists()
    assert (pipeline_root / "native_abliteration_metrics.json").is_file()


def test_cli_image_model_flag():
    from anlord.cli.main import create_parser

    parser = create_parser()
    assert parser.parse_args(["--model", "x/model"]).image_model is None
    assert parser.parse_args(["--model", "x/model", "--image-model"]).image_model is True
    assert parser.parse_args(["--model", "x/model", "--no-image-model"]).image_model is False


def test_resolve_image_target(tmp_path):
    from anlord.config import Settings
    from anlord.pipeline import AbliterationPipeline

    repo = _make_diffusers_repo(tmp_path)

    def make(model, image_model=None):
        settings = Settings(
            model=str(model),
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            image_model=image_model,
        )
        return AbliterationPipeline(settings)

    assert make(repo)._resolve_image_target() is True
    assert make(TINY_MODEL)._resolve_image_target() is False
    # Explicit flags override auto-detection in both directions.
    assert make(repo, image_model=False)._resolve_image_target() is False
    assert make(TINY_MODEL, image_model=True)._resolve_image_target() is True


def test_image_abliteration_end_to_end(tmp_path):
    """Engine on the encoder component + assembly = complete diffusers folder."""
    from anlord.native.abliterator import NativeAbliterator
    from anlord.native.config import (
        DatasetSpecification,
        NativeConfig,
        QuantizationMethod,
        RowNormalization,
    )

    repo = _make_diffusers_repo(tmp_path)
    original_weights = (repo / "text_encoder" / "model.safetensors").read_bytes()

    good_file = tmp_path / "good.txt"
    bad_file = tmp_path / "bad.txt"
    good_file.write_text("\n".join(f"Good prompt {i}" for i in range(6)) + "\n", encoding="utf-8")
    bad_file.write_text("\n".join(f"Bad request {i}" for i in range(6)) + "\n", encoding="utf-8")

    out_root = tmp_path / "out"
    pipeline_root = out_root / "pipeline"
    cfg = NativeConfig(
        model=str(repo / "text_encoder"),
        tokenizer_source=str(repo / "tokenizer"),
        dtypes=["float32"],
        quantization=QuantizationMethod.NONE,
        device_map="cpu",
        batch_size=2,
        row_normalization=RowNormalization.NONE,
        offload_outputs_to_cpu=False,
        seed=42,
        response_prefix="",
        n_trials=2,
        n_startup_trials=1,
        evaluation_pruning=False,
        search_seeds=False,
        good_prompts=DatasetSpecification(dataset=str(good_file), split="[:6]"),
        bad_prompts=DatasetSpecification(dataset=str(bad_file), split="[:6]"),
        study_checkpoint_dir=str(out_root / "checkpoints"),
        scorer_settings={
            "KeywordRate": {"prompts": {"dataset": str(bad_file), "split": "[:6]"}},
            "KLDivergence": {"prompts": {"dataset": str(good_file), "split": "[:6]"}},
        },
    )
    result = NativeAbliterator(cfg).run(output_dir=pipeline_root / "text_encoder")
    assert result.error is None
    assert result.final_refusals is not None

    moved = assemble_pipeline_output(repo, pipeline_root)
    assert "abliteration_reproduction.json" in moved
    # Complete diffusers layout with the ablated encoder inside.
    assert (pipeline_root / "model_index.json").is_file()
    assert (pipeline_root / "transformer" / "model.safetensors").is_file()
    assert (pipeline_root / "text_encoder" / "config.json").is_file()
    assert (pipeline_root / "abliteration_reproduction.json").is_file()
    # The exported encoder weights differ from the source; the source is intact.
    assert (pipeline_root / "text_encoder" / "model.safetensors").read_bytes() != original_weights
    assert (repo / "text_encoder" / "model.safetensors").read_bytes() == original_weights


def test_bundled_t2i_prompts():
    from anlord.image.diffusers import bundled_t2i_prompts

    harmless, harmful = bundled_t2i_prompts()
    assert harmless.is_file() and harmful.is_file()
    for path in (harmless, harmful):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 100
