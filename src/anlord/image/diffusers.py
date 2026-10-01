# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Diffusers text-to-image repository support for the image-abliteration branch.

Modern text-to-image models (Qwen-Image lineage, Z-Image, ...) ship as
diffusers pipelines whose ``text_encoder`` is an aligned causal LM (a
Qwen-family instruct model). Abliterating that encoder with the standard
refusal-direction machinery removes prompt-level censorship while the
transformer (DiT) and VAE stay untouched — the approach used by the
community's "AbliteratedTE" checkpoints.

This module:
- detects diffusers repositories (local paths, cached snapshots, or HF repos),
- downloads the full pipeline snapshot (the export needs every component),
- re-assembles a complete, ready-to-use diffusers folder around the ablated
  encoder produced by the native engine.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Marker file of a diffusers pipeline repository.
DIFFUSERS_MARKER = "model_index.json"
# Pipeline subfolder holding the text encoder (the only component we ablate).
ENCODER_DIR = "text_encoder"
TOKENIZER_DIR = "tokenizer"
# Engine artifacts written next to the exported encoder; moved to the pipeline
# root during assembly so the final folder mirrors a normal run's output.
_ARTIFACTS = (
    "abliteration_reproduction.json",
    "abliteration_pareto_front.json",
    "native_abliteration_metrics.json",
    "pareto_trials",
)


PROMPTS_DIR = Path(__file__).parent / "prompts"


def bundled_t2i_prompts() -> tuple[Path, Path]:
    """(harmless, harmful) bundled T2I-style prompt files, one prompt per line.

    Image-generation censorship is triggered by descriptive image prompts, not
    by the LLM-style request phrasing of the default text-model datasets, so
    the image branch uses these as its prompt defaults. Drop-in replaceable.
    """
    return PROMPTS_DIR / "t2i_harmless.txt", PROMPTS_DIR / "t2i_harmful.txt"


def _hub_cache(cache_dir: Path | str | None) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_local_repo(
    source: str,
    cache_dir: Path | str | None = None,
    model_commit: str | None = None,
) -> Path | None:
    """
    The local directory of a diffusers repo for `source`, or None.

    Accepts a local directory or an HF repo id already present in the local
    cache. Never touches the network — unknown sources return None (callers
    decide whether to download).
    """
    path = Path(source).expanduser()
    if path.is_dir():
        return path if (path / DIFFUSERS_MARKER).is_file() else None
    from ..models.downloader import find_local_snapshot

    snapshot = find_local_snapshot(_hub_cache(cache_dir), str(source), model_commit)
    if snapshot is not None and (snapshot / DIFFUSERS_MARKER).is_file():
        return snapshot
    return None


def is_diffusers_repo(
    source: str,
    cache_dir: Path | str | None = None,
    model_commit: str | None = None,
) -> bool:
    """
    Whether `source` (local path or HF repo id) is a diffusers pipeline.

    Local paths and cached snapshots are decided offline; unknown HF ids fall
    back to a best-effort remote check. Any failure returns False — detection
    must never block a run.
    """
    local = resolve_local_repo(source, cache_dir, model_commit)
    if local is not None:
        return True
    path = Path(source).expanduser()
    if path.is_dir():
        return False  # local model of another kind (no model_index.json)
    try:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN") or None
        return bool(
            HfApi(token=token).file_exists(
                str(source), DIFFUSERS_MARKER, revision=model_commit
            )
        )
    except Exception as error:
        logger.debug("Diffusers detection failed for %s: %s", source, error)
        return False


def prefetch_image_repo(
    model_id: str,
    cache_dir: Path | str | None,
    revision: str | None = None,
) -> str:
    """
    Full snapshot of the diffusers pipeline (local directory is returned as-is).

    The whole repository is downloaded on purpose: the final export must be a
    complete pipeline folder (transformer, VAE, scheduler, tokenizer), so
    encoder-only partial downloads would not produce a usable result.
    """
    local = Path(model_id).expanduser()
    if local.is_dir():
        return str(local.resolve())

    from huggingface_hub import snapshot_download

    hub_cache = Path(cache_dir).expanduser() / "hub" if cache_dir else None
    logger.info("Downloading full diffusers snapshot: %s", model_id)
    return snapshot_download(
        repo_id=model_id,
        cache_dir=str(hub_cache) if hub_cache else None,
        revision=revision,
        token=os.environ.get("HF_TOKEN") or None,
    )


def assemble_pipeline_output(source_repo: Path | str, pipeline_root: Path | str) -> list[str]:
    """
    Assemble the final diffusers folder around the ablated encoder.

    The native engine wrote the merged encoder into
    ``pipeline_root/text_encoder``. This copies every other component of the
    source repository (transformer, VAE, scheduler, tokenizer, model_index.json,
    ...) into ``pipeline_root``, then moves the engine's artifact files
    (metrics, reproduction bundle, Pareto trials) from ``text_encoder/`` up to
    the pipeline root. Returns the artifact names that were moved.
    """
    source_repo = Path(source_repo)
    pipeline_root = Path(pipeline_root)
    pipeline_root.mkdir(parents=True, exist_ok=True)

    for entry in sorted(source_repo.iterdir()):
        if entry.name == ENCODER_DIR:
            continue
        destination = pipeline_root / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, destination)

    encoder_output = pipeline_root / ENCODER_DIR
    moved: list[str] = []
    for name in _ARTIFACTS:
        artifact = encoder_output / name
        if artifact.exists():
            shutil.move(str(artifact), str(pipeline_root / name))
            moved.append(name)
    return moved
