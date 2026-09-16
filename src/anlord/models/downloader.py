# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model snapshot prefetching with verification and user-visible progress."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_METADATA_PATTERNS = [
    "*.json",
    "*.jinja",
    "*.model",
    "*.py",
    "*.txt",
    "*.tiktoken",
    "*.toml",
    "*.vocab",
    "*.bpe",
    "*.yaml",
    "*.yml",
]
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_MIN_WEIGHT_BYTES = 1_000_000


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int | None = None


@dataclass(frozen=True)
class SnapshotPlan:
    files: tuple[RemoteFile, ...]
    allow_patterns: list[str] | None
    weight_files: tuple[RemoteFile, ...]
    expected_bytes: int

    @property
    def expected_gb(self) -> float:
        return self.expected_bytes / (1024**3)


class VisibleTqdm:
    """Force Hugging Face progress bars to stay visible even without a TTY."""

    _lock = None

    @classmethod
    def get_lock(cls):
        if cls._lock is None:
            from tqdm.auto import tqdm
            cls._lock = tqdm.get_lock()
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        from tqdm.auto import tqdm
        tqdm.set_lock(lock)
        cls._lock = lock

    def __init__(self, *args, **kwargs):
        from tqdm.auto import tqdm

        kwargs["disable"] = False
        kwargs.setdefault("mininterval", 0.2)
        kwargs.setdefault("dynamic_ncols", True)
        self._bar = tqdm(*args, **kwargs)

    def __iter__(self):
        return iter(self._bar)

    def __enter__(self):
        self._bar.__enter__()
        return self

    def __exit__(self, *args):
        return self._bar.__exit__(*args)

    def update(self, n=1):
        return self._bar.update(n)

    def close(self):
        return self._bar.close()

    def set_description(self, *args, **kwargs):
        return self._bar.set_description(*args, **kwargs)

    def set_postfix(self, *args, **kwargs):
        return self._bar.set_postfix(*args, **kwargs)

    def refresh(self):
        return self._bar.refresh()

    def __getattr__(self, name):
        return getattr(self._bar, name)


def prefetch_model_snapshot(
    model_id: str,
    *,
    cache_dir: Path | str,
    revision: str | None = None,
) -> str:
    """Download a model snapshot into the shared cache with progress bars.

    Only the preferred weight format and standard model/tokenizer metadata are
    downloaded. Existing cache entries are verified against remote file sizes so
    a previous 0-byte or LFS-pointer cache cannot be treated as complete.
    """
    local_path = Path(model_id).expanduser()
    if local_path.exists():
        logger.info("Using local model: %s", local_path.resolve())
        return str(local_path.resolve())

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    hub_cache = Path(cache_dir).resolve() / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    logger.info("Inspecting model files on Hugging Face: %s", model_id)
    api = HfApi(token=token)
    try:
        remote_files = list_remote_model_files(api, model_id, revision=revision)
    except GatedRepoError as error:
        raise RuntimeError(
            "Access to this gated model was denied. Accept the model terms on Hugging Face "
            "and provide a read token at startup."
        ) from error
    plan = build_snapshot_plan(remote_files)
    reject_unsupported_weight_layout(model_id, remote_files, plan)
    if plan.allow_patterns is None and plan.weight_files:
        logger.warning(
            "Could not identify a standard model weight format; downloading the full repository"
        )
    elif plan.allow_patterns is not None:
        weight_patterns = [
            pattern for pattern in plan.allow_patterns if pattern not in _METADATA_PATTERNS
        ]
        logger.info("Selected model weight format: %s", ", ".join(weight_patterns))

    print(f"\nDownloading model snapshot: {model_id}")
    print(f"Cache: {hub_cache}")
    if plan.expected_bytes:
        print(
            f"Expected: {plan.expected_gb:.2f} GB "
            f"({len(plan.weight_files)} weight files, {len(plan.files)} selected files)"
        )
    else:
        print(f"Selected files: {len(plan.files) or 'full repository'}")

    existing = find_local_snapshot(hub_cache, model_id, revision=revision)
    if existing:
        missing = incomplete_snapshot_files(existing, plan.files)
        present_bytes = sum(_local_size(existing / item.name) for item in plan.files)
        print(
            f"Local snapshot: {present_bytes / (1024**3):.2f} GB present, "
            f"{len(missing)} incomplete/missing files"
        )
        if not missing:
            print("Model snapshot is already complete. Skipping download.\n")
            logger.info("Model snapshot ready: %s", existing)
            _warn_about_extra_shards(hub_cache, model_id, plan)
            return str(existing)
        logger.warning("Cached snapshot is incomplete; downloading missing files")
    else:
        _ensure_disk_space(hub_cache, plan.expected_bytes)

    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=hub_cache,
            token=token,
            allow_patterns=plan.allow_patterns,
            max_workers=4,
            tqdm_class=VisibleTqdm,
        )
    except GatedRepoError as error:
        raise RuntimeError(
            "Model download was denied. Accept the model terms on Hugging Face and "
            "provide a read token at startup."
        ) from error
    except HfHubHTTPError as error:
        raise RuntimeError(
            f"Hugging Face download failed for {model_id!r}. Check the network and token."
        ) from error

    snapshot = Path(snapshot_path)
    missing = incomplete_snapshot_files(snapshot, plan.files)
    if missing:
        missing_names = [item.name for item in missing]
        missing_bytes = sum(item.size or 0 for item in missing)
        logger.warning(
            "Snapshot still incomplete after download (%s files, %.2f GB). Forcing re-download.",
            len(missing),
            missing_bytes / (1024**3),
        )
        _ensure_disk_space(hub_cache, missing_bytes)
        snapshot_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=hub_cache,
            token=token,
            allow_patterns=missing_names,
            force_download=True,
            max_workers=2,
            tqdm_class=VisibleTqdm,
        )
        snapshot = Path(snapshot_path)
        still_missing = incomplete_snapshot_files(snapshot, plan.files)
        if still_missing:
            names = ", ".join(item.name for item in still_missing[:8])
            extra = "" if len(still_missing) <= 8 else f" and {len(still_missing) - 8} more"
            raise RuntimeError(
                "Model snapshot is incomplete after a forced re-download. Missing or truncated: "
                f"{names}{extra}. Delete the cache folder and retry."
            )

    present_bytes = sum(_local_size(snapshot / item.name) for item in plan.files)
    print(f"Verified snapshot: {present_bytes / (1024**3):.2f} GB")
    print("Model snapshot is ready.\n")
    logger.info("Model snapshot ready: %s", snapshot_path)
    _warn_about_extra_shards(hub_cache, model_id, plan)
    return snapshot_path


def list_remote_model_files(api, model_id: str, revision: str | None = None) -> list[RemoteFile]:
    """Return remote filenames, with sizes when Hugging Face provides them."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        info = api.model_info(model_id, revision=revision, files_metadata=True)
        siblings = getattr(info, "siblings", None) or []
        files = [
            RemoteFile(name=sibling.rfilename, size=getattr(sibling, "size", None))
            for sibling in siblings
            if getattr(sibling, "rfilename", None)
        ]
        if files:
            return files
    except GatedRepoError:
        raise
    except (HfHubHTTPError, AttributeError, TypeError, ValueError) as error:
        logger.debug("model_info(files_metadata=True) failed, falling back: %s", error)

    try:
        names = api.list_repo_files(repo_id=model_id, revision=revision)
    except GatedRepoError:
        raise
    except HfHubHTTPError as error:
        raise RuntimeError(
            f"Could not inspect Hugging Face model {model_id!r}. Check the model ID, "
            "network connection, and optional HF token."
        ) from error
    return [RemoteFile(name=name) for name in names]


def build_snapshot_plan(remote_files: Iterable[RemoteFile | str]) -> SnapshotPlan:
    files = [
        item if isinstance(item, RemoteFile) else RemoteFile(name=item) for item in remote_files
    ]
    allow_patterns = select_download_patterns(item.name for item in files)
    if allow_patterns is None:
        selected = tuple(files)
    else:
        selected = tuple(item for item in files if _matches_any(item.name, allow_patterns))
    weight_files = tuple(item for item in selected if _is_weight_file(item.name))
    expected_bytes = sum(item.size or 0 for item in selected)
    return SnapshotPlan(
        files=selected,
        allow_patterns=allow_patterns,
        weight_files=weight_files,
        expected_bytes=expected_bytes,
    )


def reject_unsupported_weight_layout(
    model_id: str,
    remote_files: Iterable[RemoteFile | str],
    plan: SnapshotPlan,
) -> None:
    """Fail early when the repo cannot be loaded by Abliteration/Transformers."""
    names = [
        item.name if isinstance(item, RemoteFile) else item for item in remote_files
    ]
    has_gguf = any(name.lower().endswith(".gguf") for name in names)
    has_transformers_weights = any(_is_weight_file(name) for name in names)
    if has_gguf and not has_transformers_weights:
        suggested = model_id.removesuffix("-GGUF").removesuffix("-gguf")
        if suggested == model_id:
            suggested = "the matching non-GGUF Hugging Face repo"
        raise RuntimeError(
            f"{model_id} is a GGUF repository. Abliteration and Anlord Abliterator need "
            "Transformers weights (.safetensors), not a llama.cpp quant. "
            f"Abliterate {suggested} instead, then convert the exported model to "
            "Q4_K_M with llama.cpp if you want that GGUF file. "
            "You cannot pick one GGUF quant from a multi-file repo for abliteration."
        )
    if plan.allow_patterns is None and not plan.weight_files:
        logger.warning(
            "Could not identify a standard model weight format; downloading the full repository"
        )


def select_download_patterns(repo_files: Iterable[str]) -> list[str] | None:
    """Select one preferred model weight format plus required metadata."""
    filenames = list(repo_files)
    if any(filename.endswith(".safetensors") for filename in filenames):
        weight_patterns = ["*.safetensors"]
    elif any(filename.endswith(".bin") for filename in filenames):
        weight_patterns = ["*.bin"]
    elif any(filename.endswith((".pt", ".pth")) for filename in filenames):
        weight_patterns = ["*.pt", "*.pth"]
    else:
        return None
    return [*_METADATA_PATTERNS, *weight_patterns]


def find_local_snapshot(
    hub_cache: Path | str,
    model_id: str,
    revision: str | None = None,
) -> Path | None:
    repo_dir = Path(hub_cache) / f"models--{model_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    if revision:
        candidate = snapshots / revision
        if candidate.is_dir():
            return candidate
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def incomplete_snapshot_files(snapshot: Path, expected: Iterable[RemoteFile]) -> list[RemoteFile]:
    missing: list[RemoteFile] = []
    for item in expected:
        path = snapshot / item.name
        if _is_incomplete_file(path, item):
            missing.append(item)
    return missing


def _is_incomplete_file(path: Path, item: RemoteFile) -> bool:
    if not path.exists():
        return True
    size = _local_size(path)
    if size <= 0:
        return True
    if _looks_like_lfs_pointer(path):
        return True
    if item.size and size < int(item.size * 0.98):
        return True
    if _is_weight_file(item.name) and not item.size and size < _MIN_WEIGHT_BYTES:
        return True
    return False


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        return path.read_text(encoding="utf-8", errors="ignore").startswith(
            "version https://git-lfs.github.com"
        )
    except OSError:
        return True


def _local_size(path: Path) -> int:
    try:
        if path.is_file() or path.is_symlink():
            return path.stat().st_size
    except OSError:
        return 0
    return 0


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    basename = Path(name).name
    return any(fnmatch(name, pattern) or fnmatch(basename, pattern) for pattern in patterns)


def _is_weight_file(name: str) -> bool:
    return name.endswith(_WEIGHT_SUFFIXES)


def _ensure_disk_space(target: Path, needed_bytes: int) -> None:
    if needed_bytes <= 0:
        return
    free = shutil.disk_usage(target).free
    required = needed_bytes + 2 * (1024**3)
    if free < required:
        raise RuntimeError(
            f"Not enough disk space in {target}. Need about {required / (1024**3):.1f} GB, "
            f"have {free / (1024**3):.1f} GB free."
        )


def _warn_about_extra_shards(hub_cache: Path, model_id: str, plan: SnapshotPlan) -> None:
    repo_dir = Path(hub_cache) / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.exists():
        return
    local_weights = list(repo_dir.rglob("*.safetensors")) + list(repo_dir.rglob("*.bin"))
    remote_count = len(plan.weight_files)
    if remote_count and len(local_weights) > remote_count * 2:
        logger.warning(
            "Found %s local weight files for %s, but the Hugging Face repo publishes %s. "
            "This usually means a duplicated Hugging Face cache. Do not delete individual shards.",
            len(local_weights),
            model_id,
            remote_count,
        )
