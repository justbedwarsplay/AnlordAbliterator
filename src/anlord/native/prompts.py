# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Prompt loading — exact parity with abliteration/utils.py load_prompts.

Key requirement: offline parity without HF Hub. Abliteration loads from
`datasets.load_dataset(path, split=...)`. Our uploaded snapshots are stored
as `datasets--<id>/snapshots/<hash>/data/*.parquet` plus `cache/hub` mirror.
When HF is offline, `load_dataset` will fail to fetch the dataset script.
We therefore:

1. Try the exact Abliteration path first (is_hf_path -> load_dataset)
2. On ConnectionError / offline, fall back to local parquet files discovered
   under repo root `datasets--*/snapshots/*/data` or `cache/hub/...`
3. Apply split slicing identically via `ReadInstruction`

This preserves the *prompt strings* Abliteration would have loaded, which is what
matters for KL/refusal parity. Phase 3 requires that the same tokenization
pipeline sees the same strings in the same order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, TypeVar

from datasets import DatasetDict, ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from rich.console import Console

from .config import DatasetSpecification, NativeConfig

print = Console(highlight=False).print

T = TypeVar("T")


@dataclass
class Prompt:
    system: str
    user: str


def is_hf_path(path: str) -> bool:
    """Mirrors abliteration.utils.is_hf_path"""
    if Path(path).exists():
        return False
    try:
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(path)
        return True
    except Exception:
        return False


def get_split_slice(split_str: str, length: int) -> tuple[int, int]:
    split_name = split_str.split("[")[0]
    name_to_length = {split_name: length}
    absolute_instruction = ReadInstruction.from_spec(split_str).to_absolute(name_to_length)[0]
    return absolute_instruction.from_, absolute_instruction.to  # type: ignore[return-value]


def batchify(items: List[T], batch_size: int) -> List[List[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _find_parquet_for_dataset(dataset_id: str) -> List[Path]:
    """
    Locate local parquet mirrors for `mlabonne/harmful_behaviors` etc.
    Search order:
      1. ./cache/hub/datasets--.../snapshots/<hash>/data/*.parquet
      2. ./datasets--.../snapshots/<hash>/data/*.parquet (repo root tracked)
      3. ./cache/datasets/... (HF datasets cache — not used in this repo)
    """
    # turn "mlabonne/harmful_behaviors" -> "datasets--mlabonne--harmful_behaviors"
    hub_name = "datasets--" + dataset_id.replace("/", "--")
    candidates: List[Path] = []
    repo_root = Path(__file__).resolve().parents[3]  # src/anlord/native -> repo root
    search_roots = [
        repo_root / "cache" / "hub" / hub_name,
        repo_root / hub_name,
        Path.cwd() / "cache" / "hub" / hub_name,
        Path.cwd() / hub_name,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        # snapshots/<hash>/data/*.parquet
        for parquet in root.glob("snapshots/*/data/*.parquet"):
            candidates.append(parquet)
        # also consider refs/main -> snapshots/<hash>
        if not candidates:
            ref = root / "refs" / "main"
            if ref.exists():
                try:
                    h = ref.read_text().strip()
                    data_dir = root / "snapshots" / h / "data"
                    if data_dir.exists():
                        candidates.extend(data_dir.glob("*.parquet"))
                except Exception:
                    pass
    return sorted(candidates)


def _load_via_parquet_fallback(spec: DatasetSpecification) -> List[str]:
    """
    Offline fallback: load via `datasets.load_dataset('parquet', data_files=...)`
    then apply Abliteration split slicing logic.
    """
    dataset_id = spec.dataset
    parquets = _find_parquet_for_dataset(dataset_id)
    if not parquets:
        raise FileNotFoundError(
            f"No local parquet found for dataset {dataset_id!r}. "
            f"Searched cache/hub and repo root datasets-- mirrors."
        )

    # Need to decide train vs test based on split string.
    # Parquet files are named train-....parquet and test-....parquet.
    # We collect only those matching the split prefix before '['.
    split_str = spec.split or ""
    split_name = split_str.split("[")[0].strip() if split_str else ""
    # e.g. "train[:400]" -> split_name "train"
    #      "test[:100]"  -> "test"
    # If no split prefix (plain text file case), use all.
    if split_name in ("train", "test", ""):
        if split_name:
            filtered = [p for p in parquets if split_name in p.name]
            # if we filtered too aggressively and got 0, fall back to all
            if filtered:
                parquets = filtered
        # else no filtering
    # load
    data_files = [str(p) for p in parquets]
    # datasets will concat multiple parquet shards if needed
    ds = load_dataset("parquet", data_files=data_files, split="train")
    # ds may be Dataset; apply slice
    if spec.split is not None:
        start, end = get_split_slice(f"_{spec.split}" if not spec.split[0].isalpha() else spec.split, len(ds))
        # but spec.split already includes e.g. "train[:400]" — reuse directly
        # simpler: if split_str like "train[:400]", we already filtered train parquets, need to slice 0:400
        # Use ReadInstruction with synthetic split name
        s = spec.split
        # load_prompts does: get_split_slice(f"_{split}", len(prompts)) for text files,
        # but for dataset path it does: dataset = load_dataset(path, split=s); prompts=list(dataset[column])
        # So for parquet fallback we need to emulate loading the named split.
        # We already selected matching files; now slice appropriately.
        # Parse slice part
        if "[" in s:
            # extract slice like "[:400]"
            slice_part = s[s.find("[") :]
            synth = f"train{slice_part}" if "train" in s else f"test{slice_part}" if "test" in s else s
            start, end = get_split_slice(synth, len(ds))
            ds = ds.select(range(start, end))
    column = spec.column or "text"
    prompts = list(ds[column])
    return prompts


def _pin_dataset_commit(spec: DatasetSpecification) -> None:
    """
    Pin the dataset to its latest Hugging Face commit if not already pinned, so
    the exact dataset version is recorded for reproducibility.

    Fetching the commit hash requires internet access, but the dataset itself
    may be fully cached locally. If pinning fails, we proceed without pinning;
    an unpinned dataset disables the reproduction bundle during export.
    """
    if spec.commit is not None or not is_hf_path(spec.dataset):
        return
    try:
        import huggingface_hub
        import huggingface_hub.constants

        if getattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False):
            print(
                f"[yellow]Warning: Hugging Face Hub is offline; dataset [bold]{spec.dataset}[/] "
                "will not be pinned to a commit.[/]"
            )
            return
        spec.commit = huggingface_hub.dataset_info(spec.dataset).sha
    except Exception as error:
        print(
            f"[yellow]Warning: Could not fetch the latest commit hash for dataset "
            f"[bold]{spec.dataset}[/] ({error}). The dataset version will not be pinned.[/]"
        )


def load_prompts(config: NativeConfig, spec: DatasetSpecification) -> List[Prompt]:
    """
    Exact parity with abliteration.utils.load_prompts, plus offline parquet fallback.
    """
    path = spec.dataset
    split_str = spec.split

    # Plain text file path (one prompt per line)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as file:
            prompts = [line.strip() for line in file if line.strip()]
        if split_str is not None:
            start, end = get_split_slice(f"_{split_str}", len(prompts))
            prompts = prompts[start:end]
    else:
        if split_str is None:
            raise ValueError(f'The "split" field is required for datasets: {path}')
        if spec.column is None:
            raise ValueError(f'The "column" field is required for datasets: {path}')

        if is_hf_path(path):
            # Record the exact dataset version for reproducibility.
            _pin_dataset_commit(spec)
            # Fast path: if local parquet mirrors exist, use them immediately without hitting Hub
            # This avoids 5× TLS retries when Hub is unreachable (common in CI).
            parquets = _find_parquet_for_dataset(path)
            if parquets:
                try:
                    prompts = _load_via_parquet_fallback(spec)
                except Exception:
                    # fallback to Hub on parquet failure
                    dataset = load_dataset(
                        path,
                        name=spec.config,
                        revision=spec.commit,
                        split=split_str,
                    )
                    prompts = list(dataset[spec.column])  # type: ignore[index]
            else:
                try:
                    dataset = load_dataset(
                        path,
                        name=spec.config,
                        revision=spec.commit,
                        split=split_str,
                    )
                    prompts = list(dataset[spec.column])  # type: ignore[index]
                except Exception as e:
                    # Detect offline / hub failure -> fallback to parquet
                    err_str = str(e).lower()
                    if any(k in err_str for k in ("offline", "couldn't reach", "connection", "ssl", "tls", "eof", "resolve")) or "ConnectionError" in type(e).__name__:
                        # Try local parquet
                        try:
                            prompts = _load_via_parquet_fallback(spec)
                        except Exception as fallback_e:
                            raise RuntimeError(
                                f"Failed to load dataset {path!r} via Hub ({e}) "
                                f"and fallback parquet also failed ({fallback_e})"
                            ) from e
                    else:
                        raise
        elif Path(path, DATASET_STATE_JSON_FILENAME).exists():
            dataset = load_from_disk(path)
            assert not isinstance(dataset, DatasetDict), "Loading dataset dicts is not supported"
            start, end = get_split_slice(split_str, len(dataset))  # type: ignore[arg-type]
            dataset = dataset[start:end]  # type: ignore[index]
            prompts = list(dataset[spec.column])  # type: ignore[index]
        else:
            dataset = load_dataset(
                path,
                split=split_str,
                verification_mode=VerificationMode.NO_CHECKS,
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
            prompts = list(dataset[spec.column])  # type: ignore[index]

    if spec.prefix:
        prompts = [f"{spec.prefix} {prompt}" for prompt in prompts]
    if spec.suffix:
        prompts = [f"{prompt} {spec.suffix}" for prompt in prompts]

    system_prompt = config.system_prompt if spec.system_prompt is None else spec.system_prompt
    return [Prompt(system=system_prompt, user=prompt) for prompt in prompts]

