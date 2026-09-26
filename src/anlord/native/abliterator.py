# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Native abliterator — full abliteration pipeline.

Implements:
  model → prompts → residual extraction → residual geometry analysis →
  refusal directions → parameterized abliteration → scorer evaluation →
  Optuna/TPE (scorer-derived objectives) → Pareto front → best params →
  final abliteration → export (merge or adapter) → reproduction bundle

The optimization objectives are provided by the configurable scorer plugins
(see anlord.native.scorer and anlord.native.scorers); a reproduction mode
re-applies a previously found ablation from a reproduce.json file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import warnings
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from os.path import commonprefix
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import optuna
from optuna import Trial, TrialPruned  # type: ignore
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import TrialState
from rich.console import Console

from .analyzer import ResidualAnalyzer
from .config import ExportStrategy, NativeConfig, QuantizationMethod
from .model import AbliterationParameters, Model
from .evaluator import Evaluator
from .prompts import load_prompts
from .reproduce import (
    REPRODUCTION_SCHEMA_VERSION,
    collect_model_hashes,
    create_reproduction_folder,
    native_config_to_reproduction_dict,
    verify_model_hashes,
)
from .utils import (
    empty_cache,
    format_duration,
    print_memory_usage,
    set_seed,
)

print = Console(highlight=False).print
logger = logging.getLogger(__name__)

# Native config fields that are pure runtime diagnostics: they do not affect
# the produced model, so reproduction keeps the local values instead of
# restoring them from the recorded native config.
_DISPLAY_ONLY_NATIVE_FIELDS = frozenset(
    {
        "print_residual_geometry",
        "plot_residuals",
        "residual_plot_path",
        "residual_plot_title",
        "residual_plot_style",
        "print_debug_information",
    }
)


def _silhouette_position_low(silh_scores: list[float]) -> float:
    """
    1-based layer index where meaningful good/bad cluster separation starts
    (layers scoring at least half of the best silhouette). Used as the lower
    bound of the max_weight_position search range.
    """
    top = max(silh_scores)
    if top <= 0:
        return 1.0
    meaningful = [i for i, s in enumerate(silh_scores, start=1) if s >= 0.5 * top]
    return float(min(meaningful)) if meaningful else 1.0


def _stage_best_params(study, stage_value, components) -> Optional[dict]:
    """
    The best (fewest refusals, then lowest KL) parameter set among completed
    trials of the given staged-search stage, restricted to `components`.
    """
    best = None
    for tr in study.trials:
        if tr.state != TrialState.COMPLETE:
            continue
        if tr.user_attrs.get("stage") != stage_value:
            continue
        refusals = tr.user_attrs.get("refusals")
        kl = tr.user_attrs.get("kl_divergence")
        if refusals is None or kl is None:
            continue
        key = (int(refusals), float(kl))
        if best is None or key < best[0]:
            best = (key, tr)
    if best is None:
        return None
    raw = best[1].user_attrs.get("parameters") or {}
    from .model import AbliterationParameters as _AP

    return {c: _AP(**raw[c]) for c in components if c in raw}


def max_weight_bounds(component: str, max_weight_limit: float) -> tuple[float, float]:
    """
    Search bounds for a component's max_weight.

    Lower: the MLP can be fully disabled (negative bound clamped to 0 in the
    objective), attention has a floor of 0.8. Upper: the configurable limit,
    kept strictly above the lower bound.
    """
    lower = -0.25 if component == "mlp.down_proj" else 0.8
    upper = max(float(max_weight_limit), lower + 0.05)
    return lower, upper


def evaluator_refusals_available(config: NativeConfig) -> bool:
    """
    True when a keyword-rate-like scorer is configured, i.e. integer refusal
    counts are measurable. Decides n/a vs real-number reporting.
    """
    for scorer_config in config.scorers:
        plugin = scorer_config.plugin
        if plugin.startswith("anlord.") and plugin.rsplit(".", 1)[-1] == "KeywordRate":
            return True
    return False


def _completed_metric_points(study) -> list[tuple[int, float]]:
    """(refusal count, KL) pairs of all completed trials, for the dominance
    checks of the multi-fidelity pruning."""
    points: list[tuple[int, float]] = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        refusals = trial.user_attrs.get("refusals")
        kl = trial.user_attrs.get("kl_divergence")
        if refusals is None or kl is None:
            continue
        points.append((int(refusals), float(kl)))
    return points


def _pruning_schedule(
    total_prompts: int, batch_size: int, fractions: list[float]
) -> list[int]:
    """
    Prefix lengths for the progressive refusal evaluation.

    Steps are rounded up to the generation batch grid so the prefix counts are
    produced by exactly the same batch composition as the full evaluation (and
    are therefore exact lower bounds of the full refusal count). Steps that
    round up to the full evaluation are dropped.
    """
    steps: list[int] = []
    if total_prompts <= 0 or not fractions or batch_size is None or batch_size <= 0:
        return steps
    for fraction in fractions:
        k = max(1, math.ceil(total_prompts * float(fraction)))
        k = math.ceil(k / batch_size) * batch_size
        k = min(k, total_prompts)
        if 0 < k < total_prompts and k not in steps:
            steps.append(k)
    return sorted(steps)


class NativeAbliteratorResult:
    def __init__(
        self,
        model_id: str,
        abliterated_model_path: Optional[str] = None,
        initial_refusals: Optional[int] = None,
        final_refusals: Optional[int] = None,
        total_prompts: Optional[int] = None,
        kl_divergence: Optional[float] = None,
        trials: int = 0,
        best_trial: int = 0,
        config: dict | None = None,
        error: Optional[str] = None,
        hash_verification: Optional[dict[str, str]] = None,
        reproduction_mode: bool = False,
        score_verification: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.abliterated_model_path = abliterated_model_path
        self.initial_refusals = initial_refusals
        self.final_refusals = final_refusals
        self.total_prompts = total_prompts
        self.kl_divergence = kl_divergence
        self.trials = trials
        self.best_trial = best_trial
        self.config = config or {}
        self.error = error
        self.hash_verification = hash_verification
        self.reproduction_mode = reproduction_mode
        self.score_verification = score_verification

    def to_dict(self) -> dict:
        data = {
            "model": self.model_id,
            "abliterated_model": self.abliterated_model_path,
            "initial_refusals": self.initial_refusals,
            "final_refusals": self.final_refusals,
            "total_prompts": self.total_prompts,
            "kl_divergence": self.kl_divergence,
            "trials": self.trials,
            "best_trial": self.best_trial,
            "config": self.config,
            "error": self.error,
        }
        if self.hash_verification is not None:
            data["hash_verification"] = self.hash_verification
        if self.reproduction_mode:
            data["reproduction_mode"] = True
        if self.score_verification is not None:
            data["score_verification"] = self.score_verification
        return data


class NativeAbliterator:
    """
    1:1 native replication. Usage mirrors AbliterationWrapper but without subprocess.
    """

    def __init__(self, config: NativeConfig, anlord_settings=None):
        self.config = config
        # Top-level Anlord settings (used for reproduction metadata); may be None
        # when the abliterator is constructed directly from a NativeConfig.
        self._anlord_settings = anlord_settings
        # Ensure directories exist
        Path(config.study_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_anlord_settings(cls, settings) -> "NativeAbliterator":
        cfg = NativeConfig.from_anlord_settings(settings)
        return cls(cfg, anlord_settings=settings)

    def _setup_torch(self) -> None:
        torch.set_grad_enabled(False)
        torch._dynamo.config.cache_size_limit = 64  # type: ignore
        # silence transformers
        import transformers

        transformers.logging.set_verbosity_error()
        logging.getLogger("lm_eval").setLevel(logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)  # type: ignore

    def _maybe_autotune_batch_size(self, model: Model, good_prompts, bad_prompts) -> None:
        if self.config.batch_size != 0:
            return
        print()
        print("Determining optimal batch size...")
        if self.config.max_batch_size <= 1:
            self.config.batch_size = 1
            print("* Chosen batch size: [bold]1[/] (capped by max_batch_size)")
            return

        on_cuda = False
        if torch.cuda.is_available():
            try:
                on_cuda = next(model.model.parameters()).device.type == "cuda"
            except StopIteration:
                on_cuda = False
        if on_cuda:
            self._autotune_batch_size_predictive(model, good_prompts, bad_prompts)
        else:
            # No VRAM metering (CPU device or unknown accelerator): fall back to
            # probe-and-catch, which relies on OOM being a recoverable exception
            # outside bitsandbytes kernels.
            self._autotune_batch_size_by_probe(model, good_prompts, bad_prompts)

    @staticmethod
    def _format_bytes(num_bytes: float) -> str:
        value = float(num_bytes)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if value < 1024.0 or unit == "GiB":
                return f"{value:.2f} {unit}" if unit == "GiB" else f"{value:.0f} {unit}"
            value /= 1024.0
        return f"{value:.2f} GiB"

    def _autotune_batch_size_predictive(self, model: Model, good_prompts, bad_prompts) -> None:
        """VRAM-aware autotune — THE algorithm for every CUDA model (quantized or not).

        Measure the resident model footprint (no batch) and the peak inference footprint
        at batch 1; their difference is the per-sequence VRAM coefficient. The next power
        of two is attempted only when ``resident + coefficient * size`` still fits into
        the usable headroom, so an oversized batch is rejected on paper instead of
        OOM-ing the process (a bitsandbytes OOM on Windows can surface as an uncatchable
        0xC0000005 that kills the whole run). After every successful attempt the
        coefficient is refined, keeping the maximum seen value to stay conservative:

            batch 1 -> measure diff(resident, peak_1) -> coeff c
            predict peak(2) = resident + c*2  -> fits? try 2 -> refine c from peak_2
            predict peak(4) = resident + c*4  -> fits? try 4 -> refine c
            predict peak(8) = resident + c*8  -> does not fit -> settle on 4

        Quantized (bnb_4bit) loads keep a larger safety margin (20% vs 12%) because
        their OOM is the one that can be an uncatchable access violation; everything
        else about the algorithm is identical for both paths.

        Selection is two-dimensional: a size must fit the VRAM prediction AND generate
        at least 85% of the best tokens/s seen so far (throughput vs batch size is
        concave — past the peak, KV-cache traffic and the longest-sequence tail only
        make bigger batches slower). The best-performing size that fits wins.
        """
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        torch.cuda.synchronize()
        base_allocated = torch.cuda.memory_allocated()

        # Safety margin: allocator fragmentation, cuBLAS/cuDNN workspaces, other
        # processes and the desktop on shared GPUs. Extra generous for 4-bit because
        # its OOM can be an uncatchable access violation on Windows.
        quantized = self.config.quantization == QuantizationMethod.BNB_4BIT
        margin_fraction = 0.20 if quantized else 0.12
        margin = max(int(total_bytes * margin_fraction), 768 * 1024 * 1024)
        usable_extra = max(free_bytes - margin, 0)
        print(
            f"* Model resident: [bold]{self._format_bytes(base_allocated)}[/] | "
            f"VRAM headroom: [bold]{self._format_bytes(usable_extra)}[/] "
            f"({self._format_bytes(free_bytes)} free minus {self._format_bytes(margin)} margin)"
        )

        batch_size = 1
        best_batch_size = 1
        best_performance = -1.0
        per_unit_bytes: Optional[int] = None

        while batch_size <= self.config.max_batch_size:
            if per_unit_bytes is not None and per_unit_bytes * batch_size > usable_extra:
                print(
                    f"* Batch [bold]{batch_size}[/]: predicted peak ~{self._format_bytes(per_unit_bytes * batch_size)} "
                    f"above the model exceeds usable headroom {self._format_bytes(usable_extra)} — "
                    f"keeping [bold]{best_batch_size}[/]"
                )
                break
            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]
            try:
                # warmup (also the real memory attempt; an OOM here is caught below)
                model.get_responses(prompts)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                responses = model.get_responses(prompts)
                end = time.perf_counter()
                torch.cuda.synchronize()
            except Exception as e:
                if batch_size == 1:
                    raise
                msg = str(e).strip() or repr(e)
                if "\n" in msg:
                    print(f"[red]Failed:\n{msg}[/]")
                else:
                    print(f"[red]Failed ({msg})[/]")
                break
            peak_bytes = torch.cuda.max_memory_allocated()
            extra_bytes = max(peak_bytes - base_allocated, 0)
            coefficient = int(extra_bytes / batch_size)
            per_unit_bytes = max(per_unit_bytes or 0, coefficient)
            response_lengths = [len(model.tokenizer.encode(r)) for r in responses]
            performance = sum(response_lengths) / (end - start) if (end - start) > 0 else 0
            print(
                f"* Batch [bold]{batch_size}[/]: [green]Ok[/] "
                f"([bold]{performance:.0f}[/] tokens/s, peak +{self._format_bytes(extra_bytes)}, "
                f"coefficient {self._format_bytes(coefficient)}/seq)"
            )
            if performance > best_performance:
                best_performance = performance
                best_batch_size = batch_size
            elif batch_size > 1 and performance < best_performance * 0.85:
                # Throughput vs batch size is concave: once a larger batch generates
                # meaningfully fewer tokens/s than the best seen (KV-cache memory
                # traffic and the longest-sequence generation tail dominate), every
                # larger size only gets worse — probing further wastes time.
                print(
                    f"* Batch throughput regressed below 85% of the best "
                    f"({performance:.0f} vs {best_performance:.0f} tokens/s) — keeping [bold]{best_batch_size}[/]"
                )
                break
            batch_size *= 2

        self.config.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{self.config.batch_size}[/]")

    def _autotune_batch_size_by_probe(self, model: Model, good_prompts, bad_prompts) -> None:
        """Legacy autotune without VRAM metering: probe powers of two until failure."""
        batch_size = 1
        best_batch_size = -1
        best_performance = -1
        while batch_size <= self.config.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")
            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]
            try:
                # warmup
                model.get_responses(prompts)
                start = time.perf_counter()
                responses = model.get_responses(prompts)
                end = time.perf_counter()
            except Exception as e:
                if batch_size == 1:
                    raise
                msg = str(e).strip() or repr(e)
                if "\n" in msg:
                    print(f"[red]Failed:\n{msg}[/]")
                else:
                    print(f"[red]Failed ({msg})[/]")
                break
            response_lengths = [len(model.tokenizer.encode(r)) for r in responses]
            performance = sum(response_lengths) / (end - start) if (end - start) > 0 else 0
            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")
            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance
            batch_size *= 2
        self.config.batch_size = max(best_batch_size, 1)
        print(f"* Chosen batch size: [bold]{self.config.batch_size}[/]")

    def _maybe_detect_prefix(self, model: Model, good_prompts, bad_prompts) -> None:
        if self.config.response_prefix is not None:
            return
        print()
        print("Checking for common response prefix...")
        prefix_check_prompts = good_prompts[:100] + bad_prompts[:100]
        responses = model.get_responses_batched(prefix_check_prompts)
        response_prefix = commonprefix(responses).rstrip(" ")
        if response_prefix:
            print(f"* Prefix found: [bold]{response_prefix!r}[/]")
            for cot_initializer, closed_cot_block in self.config.chain_of_thought_skips:
                if response_prefix.startswith(cot_initializer):
                    response_prefix = closed_cot_block
                    print(f"* Closed Chain-of-Thought block: [bold]{response_prefix!r}[/]")
                    print("* Rechecking with prefix...")
                    old = self.config.response_prefix
                    self.config.response_prefix = response_prefix
                    responses = model.get_responses_batched(prefix_check_prompts)
                    additional = commonprefix(responses).rstrip(" ")
                    if additional:
                        response_prefix += additional
                        print(f"* Extended prefix found: [bold]{response_prefix!r}[/]")
                    self.config.response_prefix = old
                    break
            self.config.response_prefix = response_prefix
        else:
            print("* None found")
            self.config.response_prefix = ""

    def _load_capability_proxy(self) -> list[dict]:
        from datasets import load_dataset
        ds_name = getattr(self.config, "capability_proxy_dataset", "cais/mmlu")
        subset = getattr(self.config, "capability_proxy_subset", "abstract_algebra")
        n = int(getattr(self.config, "capability_proxy_samples", 20))
        candidates = [
            (ds_name, subset),
            ("cais/mmlu", "abstract_algebra"),
            ("cais/mmlu", "all"),
        ]
        last_err = None
        for cand_ds, cand_sub in candidates:
            try:
                if cand_sub == "all":
                    ds = load_dataset(cand_ds, "all", split=f"test[:{n}]", trust_remote_code=True)
                else:
                    ds = load_dataset(cand_ds, cand_sub, split=f"test[:{n}]", trust_remote_code=True)
                examples = []
                for ex in ds:
                    q = ex.get("question", "")
                    choices = ex.get("choices", [])
                    ans = ex.get("answer", 0)
                    if isinstance(ans, str):
                        if ans in ["A","B","C","D"]:
                            ans = ["A","B","C","D"].index(ans)
                        else:
                            try:
                                ans = int(ans)
                            except:
                                ans = 0
                    examples.append({"question": q, "choices": choices, "answer": ans})
                if examples:
                    return examples
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Failed to load capability proxy dataset: {last_err}")

    def _score_capability(self, model: Model, proxy: list[dict]) -> float:
        import torch
        if not proxy:
            return 0.0
        letters = ["A","B","C","D"]
        correct = 0
        for ex in proxy:
            q = ex["question"]
            choices = ex["choices"][:4]
            ans = int(ex["answer"]) if isinstance(ex["answer"], int) else 0
            prompt = f"Question: {q}\n"
            for i, c in enumerate(choices):
                prompt += f"{letters[i]}. {c}\n"
            prompt += "Answer:"
            best_idx = 0
            best_score = float("-inf")
            for idx, _ in enumerate(choices):
                choice_text = f" {letters[idx]}"
                prompt_ids = model.tokenizer(prompt, return_tensors="pt").input_ids
                full = prompt + choice_text
                full_ids = model.tokenizer(full, return_tensors="pt").input_ids
                device = next(model.model.parameters()).device
                full_ids = full_ids.to(device)
                prompt_len = prompt_ids.shape[1]
                with torch.no_grad():
                    outputs = model.model(full_ids)
                    logits = outputs.logits
                    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                    target_ids = full_ids[:, 1:]
                    score = 0.0
                    for pos in range(prompt_len, full_ids.shape[1]):
                        tok_id = target_ids[0, pos-1].item()
                        lp = log_probs[0, pos-1, tok_id].item()
                        score += lp
                    score = score / max(1, full_ids.shape[1] - prompt_len)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx == ans:
                correct += 1
        return correct / len(proxy) if proxy else 0.0


    def _enqueue_seed_trials(self, study, model: Model) -> None:
        """
        Enqueues a few sensible starting configurations for a fresh study.

        The seeds span the region where optima are typically found (mid-to-late
        layers, moderate window, near-zero min weight), so the Pareto front
        forms immediately and TPE starts from informative observations instead
        of pure random exploration. Unrequested parameters (e.g. components
        excluded by the include list) are sampled normally.
        """
        last_layer_index = len(model.get_layers()) - 1
        components = model.get_abliterable_components()
        limit = float(getattr(self.config, "max_weight_limit", 1.5) or 1.5)

        def component_params(max_weight: float, pos: float, min_frac: float, dist: float):
            params = {}
            for component in components:
                params[f"{component}.max_weight"] = min(max_weight, limit)
                params[f"{component}.max_weight_position"] = min(
                    max(pos, 0.6 * last_layer_index), 1.0 * last_layer_index
                )
                params[f"{component}.min_weight"] = min_frac
                params[f"{component}.min_weight_distance"] = min(
                    max(dist, 1.0), max(0.6 * last_layer_index, 1.0)
                )
            return params

        seeds = [
            {
                "direction_scope": "global",
                "direction_index": 0.75 * last_layer_index,
                **component_params(1.0, 0.75 * last_layer_index, 0.05, 0.3 * last_layer_index),
            },
            {
                "direction_scope": "global",
                "direction_index": 0.8 * last_layer_index,
                **component_params(1.3, 0.8 * last_layer_index, 0.0, 0.45 * last_layer_index),
            },
            {
                "direction_scope": "per layer",
                **component_params(1.1, 0.7 * last_layer_index, 0.0, 0.5 * last_layer_index),
            },
        ]

        if getattr(self.config, "staged_search", False):
            # Stage 1 is attention-only: neutralize MLP parts of the seeds so
            # they stay consistent with the staged design.
            for seed in seeds:
                for key in list(seed):
                    if key.startswith("mlp.") and (key.endswith("max_weight") or key.endswith("min_weight")):
                        seed[key] = 0.0

        print(f"* Enqueueing {len(seeds)} seed trials for the fresh study")
        for seed in seeds[: self.config.n_trials]:
            try:
                study.enqueue_trial(seed, skip_if_exists=True)
            except Exception as e:
                print(f"[yellow]Could not enqueue seed trial: {e}[/]")
                break

    def _compute_refusal_directions(
        self,
        model: Model,
        good_prompts,
        bad_prompts,
    ):
        """
        Calculates per-layer refusal directions from good/bad prompt residuals.

        The residuals are always obtained in full and the directions are derived
        through a single canonical code path, so the produced model is identical
        regardless of whether the residual geometry analysis / PaCMAP plots are
        enabled. The analysis runs only when the corresponding config flags are
        set. Supports the Anlord-specific multi-vector refusal subspace
        (refusal_subspace_rank > 1).
        """
        print()
        subspace_rank = int(getattr(self.config, "refusal_subspace_rank", 1) or 1)
        if subspace_rank > 1:
            print(f"Calculating subspace refusal directions (k={subspace_rank}) via SVD...")
        else:
            print("Calculating per-layer refusal directions...")

        print("* Obtaining residuals for good prompts...")
        good_residuals = model.get_residuals_batched(good_prompts)
        print("* Obtaining residuals for bad prompts...")
        bad_residuals = model.get_residuals_batched(bad_prompts)

        analyzer = None
        if subspace_rank > 1:
            # Subspace: per-layer SVD on (bad - good_mean)
            Ls = good_residuals.shape[1]
            H = good_residuals.shape[2]
            k = min(subspace_rank, H, good_residuals.shape[0], bad_residuals.shape[0])
            refusal_directions = torch.zeros(Ls, k, good_residuals.shape[2], dtype=torch.float32)
            for l in range(Ls):
                good_l = good_residuals[:, l, :].float()
                bad_l = bad_residuals[:, l, :].float()
                good_mean_l = good_l.mean(dim=0)
                D = bad_l - good_mean_l.unsqueeze(0)
                D = D - D.mean(dim=0, keepdim=True)
                try:
                    q = min(k + 4, min(D.shape))
                    U, S, V = torch.svd_lowrank(D, q=q, niter=4)
                    Vk = V[:, :k].T
                except Exception:
                    try:
                        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
                        Vk = Vh[:k, :]
                    except Exception as e:
                        print(f"* SVD failed for layer {l}: {e}, falling back to mean diff")
                        diff = (bad_l.mean(dim=0) - good_mean_l)
                        Vk = F.normalize(diff.unsqueeze(0), p=2, dim=1).repeat(k, 1)
                        Vk[1:] = 0
                Vk = F.normalize(Vk, p=2, dim=1)
                if self.config.orthogonalize_direction:
                    good_dir = F.normalize(good_mean_l, p=2, dim=0)
                    proj = (Vk @ good_dir)
                    Vk = Vk - proj.unsqueeze(1) * good_dir.unsqueeze(0)
                    Vk = F.normalize(Vk, p=2, dim=1)
                refusal_directions[l] = Vk
                del good_l, bad_l, D
            print(f"* Subspace directions: [{Ls}, {k}, {H}]")
            # The subspace path does not use the analyzer.
        else:
            if self.config.print_residual_geometry or self.config.plot_residuals:
                analyzer = ResidualAnalyzer(self.config, model, good_residuals, bad_residuals)
                if self.config.print_residual_geometry:
                    analyzer.print_residual_geometry()
                if self.config.plot_residuals:
                    analyzer.plot_residuals()

            if getattr(self.config, "silhouette_guided_bounds", False):
                # Per-layer silhouette of the good/bad residual clusters: the
                # first layer with meaningful separation lower-bounds the
                # max_weight_position search range (positions below it are dead
                # zones the optimizer would waste trials on).
                try:
                    from sklearn.metrics import silhouette_score

                    silh = []
                    n_layers = len(model.get_layers())
                    for li in range(1, n_layers + 1):
                        X = torch.cat(
                            [good_residuals[:, li, :], bad_residuals[:, li, :]]
                        ).detach().cpu().numpy()
                        labels = [0] * good_residuals.shape[0] + [1] * bad_residuals.shape[0]
                        silh.append(float(silhouette_score(X, labels)))
                    self._silhouette_position_low = _silhouette_position_low(silh)
                    print(
                        f"* Silhouette-guided position lower bound: layer "
                        f"{self._silhouette_position_low:.0f} of {n_layers}"
                    )
                except Exception as e:
                    print(f"[yellow]Silhouette guidance failed ({e}) — using default bounds[/]")

            # The direction source is run-identity (recorded in reproduction
            # bundles): "median" uses the per-component median of the residual
            # vectors, which is robust to massive activations.
            if getattr(self.config, "direction_source", "mean") == "median":
                print("* Using per-component median residuals for the refusal direction")
                good_means = good_residuals.median(dim=0).values
                bad_means = bad_residuals.median(dim=0).values
            else:
                good_means = good_residuals.mean(dim=0)
                bad_means = bad_residuals.mean(dim=0)
            # The full residuals are no longer needed after the analysis
            # and the means have been computed.
            del good_residuals, bad_residuals
            refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
            if self.config.orthogonalize_direction:
                good_directions = F.normalize(good_means, p=2, dim=1)
                projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
                refusal_directions = refusal_directions - projection_vector.unsqueeze(1) * good_directions
                refusal_directions = F.normalize(refusal_directions, p=2, dim=1)
                del good_directions, projection_vector
            del good_means, bad_means

        # Release the analyzer (and any residuals it still references).
        del analyzer
        empty_cache()
        return refusal_directions

    def _canonicalize_dead_lora_entries(self, model: Model) -> None:
        """
        Zero lora_A wherever lora_B is all-zero.

        Those entries contribute nothing to the model (delta = B @ A = 0), but
        their leftover values from previous trials make adapter exports
        non-reproducible bit-for-bit, because reset_model() only zeroes lora_B
        while modules outside min_weight_distance keep their previous lora_A.
        Canonicalizing the dead entries makes adapter files hash-identical
        between the original run and its reproduction.
        """
        from peft.tuners.lora.layer import Linear as LoraLinear

        for module in model.model.modules():
            if not isinstance(module, LoraLinear):
                continue
            lora_B = module.lora_B["default"].weight
            if not lora_B.abs().any():
                module.lora_A["default"].weight.zero_()

    def _export_model(self, model: Model, output_dir: Path) -> ExportStrategy:
        """
        Exports the abliterated model according to the configured export
        strategy: "merge" saves the full model with the abliteration LoRA
        merged into the weights, "adapter" saves the LoRA adapter only.

        If merging fails, an adapter is saved as a fallback so the work is not
        lost. Returns the strategy that was effectively used.
        """
        strategy = self.config.export_strategy or ExportStrategy.MERGE
        if not isinstance(strategy, ExportStrategy):
            strategy = ExportStrategy(strategy)
        print(f"* Saving model to [bold]{output_dir}[/] (export strategy: {strategy.value})...")

        if strategy == ExportStrategy.ADAPTER:
            print("* Saving LoRA adapter...")
            # Canonicalize dead entries so the adapter file is reproducible.
            self._canonicalize_dead_lora_entries(model)
            try:
                model.model.save_pretrained(
                    str(output_dir),
                    max_shard_size=self.config.max_shard_size,
                )
            except TypeError:
                # Some PEFT/transformers versions don't accept max_shard_size here.
                model.model.save_pretrained(str(output_dir))
            model.tokenizer.save_pretrained(str(output_dir))
            return strategy

        # MERGE (default)
        try:
            merged = model.get_merged_model()
            try:
                merged.save_pretrained(
                    str(output_dir),
                    safe_serialization=True,
                    max_shard_size=self.config.max_shard_size,
                )
            except TypeError:
                merged.save_pretrained(str(output_dir), safe_serialization=True)
            model.tokenizer.save_pretrained(str(output_dir))
            if model.processor is not None:
                try:
                    model.processor.save_pretrained(str(output_dir))
                except Exception:
                    pass
            del merged
            empty_cache()
        except Exception as e:
            # fallback: save peft adapter so the result is not lost
            print(f"[yellow]Merge failed ({e}), saving adapter instead[/]")
            try:
                model.model.save_pretrained(
                    str(output_dir),
                    max_shard_size=self.config.max_shard_size,
                )
            except TypeError:
                model.model.save_pretrained(str(output_dir))
            model.tokenizer.save_pretrained(str(output_dir))
            return ExportStrategy.ADAPTER
        return strategy

    def run(
        self,
        output_dir: Path | str,
        timeout: Optional[int] = None,
    ) -> NativeAbliteratorResult:
        """
        Execute full pipeline. Returns result with merged model at output_dir.
        Mirrors the optimization pipeline but without interactive prompts.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_torch()

        if self.config.seed is None:
            self.config.seed = random.randint(0, 2**32 - 1)
        set_seed(self.config.seed)

        # Study checkpoint handling
        os.makedirs(self.config.study_checkpoint_dir, exist_ok=True)
        sanitized_model = "".join(
            [(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in self.config.model]
        )
        if len(sanitized_model) > 64:
            # Long local model paths would push the journal (and its lock file
            # rename suffix) past the Windows MAX_PATH limit; keep the name
            # readable but bounded, disambiguated by a content hash.
            digest = hashlib.sha256(self.config.model.encode("utf-8")).hexdigest()[:16]
            sanitized_model = f"{sanitized_model[:48]}--{digest}"
        study_file = os.path.join(self.config.study_checkpoint_dir, sanitized_model + ".jsonl")
        lock_obj = JournalFileOpenLock(study_file)
        backend = JournalFileBackend(study_file, lock_obj=lock_obj)
        storage = JournalStorage(backend)
        try:
            existing = storage.get_all_studies()[0]
        except IndexError:
            existing = None

        if existing is not None and existing.user_attrs.get("finished"):
            print(
                "* A previous run of this model is finished; continuing will add "
                "more trials on top of it."
            )

        # Model loading
        model = Model(self.config)
        print()
        # prompt loading for refusal directions
        print(f"Loading good prompts from [bold]{self.config.good_prompts.dataset}[/]...")
        good_prompts = load_prompts(self.config, self.config.good_prompts)
        print(f"* [bold]{len(good_prompts)}[/] prompts loaded")
        print(f"Loading bad prompts from [bold]{self.config.bad_prompts.dataset}[/]...")
        bad_prompts = load_prompts(self.config, self.config.bad_prompts)
        print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

        # autotune batch
        self._maybe_autotune_batch_size(model, good_prompts, bad_prompts)
        # prefix detection
        self._maybe_detect_prefix(model, good_prompts, bad_prompts)

        # Evaluator — loads and initializes all configured scorers, then
        # computes baseline scores (including refusal counts and KL).
        evaluator = Evaluator(self.config, model)
        has_refusal_metrics = evaluator.refusal_prompt_total() > 0
        if evaluator.refusal_prompt_total() == 0:
            print(
                "[yellow]No keyword-rate scorer configured: refusal counts in results, "
                "comparisons and reports will be recorded as 0 (unknown).[/]"
            )

        # Capability proxy (feature 2): tiny MMLU to preserve capability
        capability_proxy = None
        if getattr(self.config, "capability_proxy_enabled", False):
            try:
                capability_proxy = self._load_capability_proxy()
                print(f"* Capability proxy: {self.config.capability_proxy_dataset}/{self.config.capability_proxy_subset} ({len(capability_proxy)} samples)")
                base_cap = self._score_capability(model, capability_proxy)
                print(f"* Base capability (proxy MMLU): {base_cap:.3f}")
            except Exception as e:
                print(f"[yellow]Capability proxy init failed ({e}), disabling[/]")
                capability_proxy = None

        # refusal directions (with subspace support)
        refusal_directions = self._compute_refusal_directions(
            model, good_prompts, bad_prompts
        )

        # Optuna study (scorer-derived objectives, plus capability proxy when enabled)
        study_directions = list(evaluator.get_objective_directions())
        if getattr(self.config, "capability_proxy_enabled", False) and capability_proxy is not None:
            study_directions.append(StudyDirection.MAXIMIZE)
        objective_labels = ", ".join(
            f"{name} {'↓' if d == StudyDirection.MINIMIZE else '↑'}"
            for name, d in zip(evaluator.get_objective_names(), evaluator.get_objective_directions())
        )
        if getattr(self.config, "capability_proxy_enabled", False) and capability_proxy is not None:
            objective_labels += " capability ↑"
        if not study_directions:
            raise RuntimeError(
                "No optimization objectives configured. At least one scorer must set "
                'optimization to "maximize" or "minimize".'
            )
        print(f"* Optuna objectives: {objective_labels}")

        # Staged search setup: split components into attention and MLP groups.
        all_components = model.get_abliterable_components()
        attn_components = [c for c in all_components if c.startswith("attn")]
        mlp_components = [c for c in all_components if c.startswith("mlp")]
        staged_active = bool(
            getattr(self.config, "staged_search", False)
            and attn_components
            and mlp_components
        )
        stage1_n = round(self.config.n_trials * float(getattr(self.config, "staged_stage1_fraction", 0.4) or 0.4))
        stage1_n = max(3, min(stage1_n, self.config.n_trials - 3))
        staged_attn_best: Optional[dict[str, "AbliterationParameters"]] = None
        if getattr(self.config, "staged_search", False) and not staged_active:
            print(
                "[yellow]Staged search requested but the model does not expose both attention "
                "and MLP components — running the standard search.[/]"
            )
        elif staged_active:
            print(
                f"* Staged search active: stage 1 = attention-only (trials 1-{stage1_n}), "
                f"stage 2 = MLP on top of the frozen attention winner (trials {stage1_n + 1}-{self.config.n_trials})"
            )

        study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=128,
                multivariate=True,
                # Group decomposition: a no-op for the static space, but required
                # for the staged search where stage-1 trials sample attention
                # params and stage-2 trials sample MLP params.
                group=True,
                seed=self.config.seed,
            ),
            directions=study_directions,
            storage=storage,
            study_name="abliteration",
            load_if_exists=True,
        )
        # An existing journal silently keeps its original directions (verified
        # on Optuna 4.x), so a scorer configuration change would surface as a
        # cryptic failure mid-run. Detect it and explain the way out.
        if [d.name for d in study.directions] != [d.name for d in study_directions]:
            raise RuntimeError(
                "The study journal "
                f"{study_file} was created with different optimization objectives "
                f"({', '.join(d.name for d in study.directions)}) than the current "
                f"configuration ({', '.join(d.name for d in study_directions)}). "
                "Delete the journal file to start a fresh study, or restore the "
                "previous scorer configuration to continue it."
            )

        # Multi-fidelity pruning setup. Disabled with the capability proxy:
        # the front is then three-dimensional and the capability of a trial is
        # unknown until its full evaluation, so no sound dominance test exists.
        pruning_active = bool(
            self.config.evaluation_pruning
            and not (getattr(self.config, "capability_proxy_enabled", False) and capability_proxy is not None)
            and self.config.batch_size
            and self.config.batch_size > 0
        )
        pruning_schedule: list[int] = []
        if pruning_active:
            refusal_total = evaluator.refusal_prompt_total()
            pruning_schedule = _pruning_schedule(
                refusal_total, self.config.batch_size, list(self.config.pruning_fractions)
            )
            if not pruning_schedule:
                pruning_active = False
            else:
                print(
                    f"* Early pruning of dominated trials active: refusal prefixes "
                    f"{pruning_schedule} of {refusal_total} prompts"
                )

        # Seed trials: a fresh study starts from a few sensible configurations
        # instead of spending its startup trials on pure random exploration.
        if self.config.search_seeds and len(study.trials) == 0:
            self._enqueue_seed_trials(study, model)

        # persist settings json for resume parity
        try:
            # NativeConfig -> dict -> json
            import json as _json

            study.set_user_attr("settings", _json.dumps(self.config.to_dict()))
            study.set_user_attr("finished", False)
        except Exception:
            pass

        start_index = len(
            [t for t in study.trials if t.state != TrialState.WAITING]
        )
        trial_index = start_index
        start_time = time.perf_counter()
        last_trial_end = start_time
        trial_durations: list[float] = []
        if start_index > 0:
            print()
            print("Resuming existing study.")

        # Objective — scorers provide the scores; capability proxy may add one more
        def objective(trial: Trial) -> tuple[float, ...]:
            nonlocal trial_index, last_trial_end, staged_attn_best
            trial_index += 1
            trial.set_user_attr("index", trial_index)

            direction_scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
            last_layer_index = len(model.get_layers()) - 1
            direction_index = trial.suggest_float("direction_index", 0.4 * last_layer_index, 0.9 * last_layer_index)
            if direction_scope == "per layer":
                direction_index = None

            # Staged search: stage 1 samples attention components only (MLP
            # frozen at identity); stage 2 freezes the stage-1 attention winner
            # (with a +/-20% max_weight rescale) and samples the MLP components.
            stage = 1
            attn_rescale = 1.0
            if staged_active:
                stage = 2 if trial_index > stage1_n else 1
                trial.set_user_attr("stage", "attn" if stage == 1 else "mlp")
                if stage == 2 and staged_attn_best is None:
                    staged_attn_best = _stage_best_params(
                        study, "attn", [c for c in all_components if c.startswith("attn")]
                    )
                    if staged_attn_best is None:
                        print(
                            "[yellow]Stage 1 produced no completed attention trials — "
                            "falling back to the standard full search.[/]"
                        )
                if stage == 2 and staged_attn_best is not None:
                    attn_rescale = trial.suggest_float("attn.max_weight_rescale", 0.8, 1.2)
                    trial.set_user_attr("attn_rescale", attn_rescale)

            parameters: dict[str, AbliterationParameters] = {}
            for component in model.get_abliterable_components():
                # Staged search overrides per component:
                # - stage 1: MLP components are frozen at identity (no ablation),
                #   so stage 1 is a pure attention-only search;
                # - stage 2: attention components are frozen at the stage-1 winner
                #   (with a +/-20% max_weight rescale), MLP is sampled freely.
                if staged_active and stage == 1 and component.startswith("mlp"):
                    parameters[component] = AbliterationParameters(
                        max_weight=0.0,
                        max_weight_position=0.6 * last_layer_index,
                        min_weight=0.0,
                        min_weight_distance=1.0,
                    )
                    continue
                if staged_active and stage == 2 and component.startswith("attn") and staged_attn_best and component in staged_attn_best:
                    base = staged_attn_best[component]
                    parameters[component] = AbliterationParameters(
                        max_weight=base.max_weight * attn_rescale,
                        max_weight_position=base.max_weight_position,
                        min_weight=base.min_weight * attn_rescale,
                        min_weight_distance=base.min_weight_distance,
                    )
                    continue
                # The parameter ranges are based on experiments with various models
                # and much wider ranges. They are not set in stone and might have to
                # be adjusted for future models.
                #
                # The MLP gets a negative lower bound that is then clamped to 0, so
                # the optimizer can fully disable its ablation. The clamp puts a
                # positive probability mass on exactly 0 (the continuous sampler
                # would otherwise reach 0 with probability zero). Ablating the MLP is
                # often unnecessary for removing refusals and tends to damage model
                # intelligence more than ablating the attention output, so on many
                # models the optimum is to leave it (mostly) untouched.
                max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8
                # The upper bound is configurable: attention-only runs benefit
                # from a higher limit because attention has to carry the whole
                # ablation by itself (see docs/optimization_ideas.md).
                max_weight_upper = max_weight_bounds(component, self.config.max_weight_limit)[1]
                max_weight = max(
                    0.0,
                    trial.suggest_float(f"{component}.max_weight", max_weight_lower_bound, max_weight_upper),
                )
                pos_low = 0.6 * last_layer_index
                if getattr(self, "_silhouette_position_low", None) is not None:
                    pos_low = min(max(1.0, self._silhouette_position_low), last_layer_index)
                max_weight_position = trial.suggest_float(f"{component}.max_weight_position", pos_low, 1.0 * last_layer_index)
                min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
                min_weight_distance = trial.suggest_float(
                    f"{component}.min_weight_distance",
                    1.0,
                    max(0.6 * last_layer_index, 1.0),
                )
                parameters[component] = AbliterationParameters(
                    max_weight=max_weight,
                    max_weight_position=max_weight_position,
                    min_weight=(min_weight * max_weight),
                    min_weight_distance=min_weight_distance,
                )

            trial.set_user_attr("direction_index", direction_index)
            trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

            print()
            print(f"Running trial [bold]{trial_index}[/] of [bold]{self.config.n_trials}[/]...")
            print("* Parameters:")
            for name, value in trial.params.items():
                print(f"  * {name} = [bold]{value}[/]")
            print("* Resetting model...")
            model.reset_model()
            print("* Abliterating...")
            model.abliterate(refusal_directions, direction_index, parameters)
            print("* Evaluating...")
            evaluator.clear_response_caches()

            # Multi-fidelity early abandonment: a trial that is already
            # provably dominated by some completed trial never reaches the
            # Pareto front, so it is pruned before paying for the expensive
            # refusal generation. See docs/optimization_ideas.md for the
            # soundness argument (prefix counts are exact lower bounds).
            if pruning_active and trial_index > self.config.n_startup_trials:
                front_points = _completed_metric_points(study)
                if front_points:
                    kl_value = evaluator.quick_kl_divergence()
                    if kl_value is not None:
                        zero_refusal_kls = [
                            kl for refusals, kl in front_points if refusals <= 0
                        ]
                        if zero_refusal_kls and kl_value >= min(zero_refusal_kls):
                            print(
                                f"  * [yellow]Pruned after KL step: KL {kl_value:.4f} is above the "
                                f"zero-refusal front point ({min(zero_refusal_kls):.4f})[/]"
                            )
                            raise TrialPruned()
                        total = evaluator.refusal_prompt_total()
                        for step_index, k in enumerate(pruning_schedule):
                            r_k = evaluator.quick_refusals(k)
                            if r_k is None:
                                break
                            # Record the prefix count as an intermediate value so
                            # TPE can rank pruned trials meaningfully. Trial.report
                            # refuses multi-objective studies, but the underlying
                            # storage API (used by _get_pruned_trial_score) does not
                            # care about the study dimensionality; fall back
                            # silently if that ever changes.
                            try:
                                study._storage.set_trial_intermediate_value(
                                    trial._trial_id, step_index, float(r_k)
                                )
                            except Exception:
                                pass
                            dominating = [
                                (refusals, kl)
                                for refusals, kl in front_points
                                if refusals <= r_k and kl <= kl_value
                            ]
                            if dominating:
                                refusals_f, kl_f = dominating[0]
                                print(
                                    f"  * [yellow]Pruned after {k}/{total} prompts: "
                                    f"refusals >= {r_k} with KL {kl_value:.4f} is dominated by "
                                    f"(refusals {refusals_f}, KL {kl_f:.4f})[/]"
                                )
                                trial.set_user_attr("pruned_at", k)
                                raise TrialPruned()

            scores = evaluator.get_scores()
            objective_values = list(evaluator.get_objective_values(scores))
            for name, score in scores:
                print(f"  * [bold]{name}:[/] [green]{score.rich_display}[/]")

            elapsed = time.perf_counter() - start_time
            # Rolling-window ETA: the last 20 trials decide the projection, so
            # the estimate adapts when the per-trial speed changes mid-run
            # (e.g. GPU throttling ending, batch-size effects).
            trial_durations.append(elapsed - last_trial_end)
            last_trial_end = elapsed
            recent = trial_durations[-20:]
            remaining = (sum(recent) / len(recent)) * (self.config.n_trials - trial_index) if (trial_index - start_index) > 0 else 0
            print()
            print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed)}[/][/]")
            if trial_index < self.config.n_trials:
                print(f"[grey50]Estimated remaining time: [bold]{format_duration(remaining)}[/][/]")
            print_memory_usage()

            # Capability proxy (additional objective to preserve capability)
            if capability_proxy is not None:
                try:
                    cap_score = self._score_capability(model, capability_proxy)
                    print(f"  * Capability (proxy MMLU): {cap_score:.3f}")
                except Exception as e:
                    print(f"[yellow]Capability scoring failed ({e})[/]")
                    cap_score = 0.0
                objective_values.append(cap_score)
                trial.set_user_attr("capability", cap_score)

            trial.set_user_attr("scores", evaluator.get_paired_score_records(scores))

            # Legacy metric attributes (plain refusals/KL) for pipeline
            # compatibility. None means "not measured" (no matching scorer);
            # dominance-pruning front points skip such trials.
            trial.set_user_attr("kl_divergence", evaluator.last_kl_divergence)
            trial.set_user_attr("refusals", evaluator.last_refusals)
            trial.set_user_attr(
                "base_refusals",
                evaluator.base_refusals if evaluator.refusal_prompt_total() > 0 else None,
            )
            trial.set_user_attr(
                "n_bad_prompts",
                len(evaluator.bad_prompts) if evaluator.refusal_prompt_total() > 0 else None,
            )

            return tuple(objective_values)

        def objective_wrapper(trial: Trial) -> tuple[float, float] | tuple[float, float, float]:
            try:
                return objective(trial)
            except KeyboardInterrupt:
                trial.study.stop()
                raise TrialPruned()

        # Waiting (enqueued seed) trials are not yet executed: count only the
        # actually run trials against the requested n_trials.
        executed_before = len(
            [t for t in study.trials if t.state != TrialState.WAITING]
        )
        n_needed = self.config.n_trials - executed_before
        if n_needed > 0:
            # timeout handling
            try:
                study.optimize(objective_wrapper, n_trials=n_needed, timeout=timeout)
            except KeyboardInterrupt:
                pass

        executed_after = len(
            [t for t in study.trials if t.state != TrialState.WAITING]
        )
        if executed_after >= self.config.n_trials:
            try:
                study.set_user_attr("finished", True)
            except Exception:
                pass

        # Pareto front — Optuna computes it for multi-objective studies (this
        # covers any number of scorer-derived objectives plus the optional
        # capability proxy). Sort by objective values for a deterministic pick.
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            raise RuntimeError("No completed trials")

        best_trials = sorted(
            study.best_trials,
            key=lambda tr: tuple(tr.values or ()),
        )
        print(f"* Pareto front: {len(best_trials)}/{len(completed)} trials")

        # Automatic selection: choose the first Pareto trial (best objective values).
        chosen = best_trials[0]
        print()
        print("[bold green]Optimization finished![/]")
        cap_str = f" cap {chosen.user_attrs.get('capability', 0):.3f}" if "capability" in chosen.user_attrs else ""
        refusals_str = (
            f"{chosen.user_attrs['refusals']}/{len(evaluator.bad_prompts)}"
            if chosen.user_attrs.get("refusals") is not None
            else "n/a"
        )
        kl_str = (
            f"{chosen.user_attrs['kl_divergence']:.4f}"
            if chosen.user_attrs.get("kl_divergence") is not None
            else "n/a"
        )
        print(f"* Best trial: [bold]{chosen.user_attrs['index']}[/] refusals {refusals_str} KL {kl_str}{cap_str}")

        # Re-apply best trial to model for export
        print("* Resetting model for final export...")
        model.reset_model()
        print("* Abliterating with best parameters...")
        # reconstruct parameters dict
        best_params_raw = chosen.user_attrs["parameters"]
        best_params: dict[str, AbliterationParameters] = {
            k: AbliterationParameters(**v) for k, v in best_params_raw.items()
        }
        best_direction_index = chosen.user_attrs["direction_index"]
        model.abliterate(refusal_directions, best_direction_index, best_params)

        # Export (strategy: merge the LoRA into full weights, or save the adapter)
        strategy = self._export_model(model, output_dir)
        effective_strategy = strategy.value if hasattr(strategy, "value") else str(strategy)

        # Reproduction bundle — allows re-applying this exact ablation later via
        # --reproduce. Only generated when every input is pinned and reproducible:
        # HF model and dataset paths with pinned commits, reproducible built-in
        # scorers, and no external plugins.
        if self.config.reproducibility_information != "none":
            dataset_specifications = [
                self.config.good_prompts,
                self.config.bad_prompts,
                *evaluator.get_dataset_specifications(),
            ]
            from .prompts import is_hf_path

            is_reproducible = (
                is_hf_path(self.config.model)
                and all(
                    is_hf_path(spec.dataset) and spec.commit is not None
                    for spec in dataset_specifications
                )
                and evaluator.all_scorers_reproducible()
                and evaluator.all_scorers_builtin()
            )
            if is_reproducible and self._anlord_settings is not None:
                model_hashes = collect_model_hashes(output_dir)
                reproduction_dir = create_reproduction_folder(
                    output_dir,
                    self.config,
                    self._anlord_settings,
                    checkpoint_path=study_file,
                    trial=chosen,
                    model_hashes=model_hashes,
                    include_system_information=(
                        self.config.reproducibility_information == "full"
                    ),
                    metrics={
                        "initial_refusals": evaluator.base_refusals if has_refusal_metrics else None,
                        "final_refusals": chosen.user_attrs["refusals"] if has_refusal_metrics else None,
                        "total_prompts": len(evaluator.bad_prompts) if has_refusal_metrics else None,
                        "kl_divergence": chosen.user_attrs["kl_divergence"],
                        "trials": len(study.trials),
                        "best_trial": chosen.user_attrs["index"],
                    },
                )
                print(f"* Reproduction information saved to [bold]{reproduction_dir}[/].")
            elif not is_reproducible:
                print(
                    "[yellow]Reproduction information not generated: the model/datasets are not "
                    "Hugging Face paths with pinned commits, or not all scorers are reproducible "
                    "built-in plugins.[/]"
                )

        # Always save the full ablation information: the chosen trial's complete
        # parameters (direction scope/index, per-component weights) plus every
        # Pareto-optimal trial, each in a --reproduce-compatible file. This is
        # written unconditionally — unlike the shareable reproduce/ bundle — so
        # no found ablation is ever lost to a summary metrics file.
        try:
            base_metrics = {
                "model": self.config.model,
                "initial_refusals": evaluator.base_refusals if has_refusal_metrics else None,
                "total_prompts": len(evaluator.bad_prompts) if has_refusal_metrics else None,
                "trials": len(study.trials),
            }

            def _trial_payload(trial) -> dict:
                index = trial.user_attrs["index"]
                return {
                    "version": REPRODUCTION_SCHEMA_VERSION,
                    "timestamp": datetime.now(timezone.utc)
                    .replace(microsecond=0, tzinfo=None)
                    .isoformat(),
                    "trial_index": index,
                    "parameters": {
                        "direction_index": trial.user_attrs["direction_index"],
                        "abliteration_parameters": trial.user_attrs["parameters"],
                    },
                    "scores": trial.user_attrs.get("scores", []),
                    "metrics": {
                        **base_metrics,
                        "final_refusals": trial.user_attrs.get("refusals"),
                        "kl_divergence": trial.user_attrs.get("kl_divergence"),
                        "best_trial": index,
                    },
                    "native_config": native_config_to_reproduction_dict(self.config),
                    "hashes": {},
                }

            chosen_payload = _trial_payload(chosen)
            chosen_payload["hashes"] = collect_model_hashes(output_dir)
            (output_dir / "abliteration_reproduction.json").write_text(
                json.dumps(chosen_payload, indent=2)
            )

            pareto_dir = output_dir / "pareto_trials"
            pareto_dir.mkdir(parents=True, exist_ok=True)
            pareto_entries = []
            for trial in best_trials:
                index = trial.user_attrs["index"]
                filename = f"trial_{index}.json"
                (pareto_dir / filename).write_text(
                    json.dumps(_trial_payload(trial), indent=2)
                )
                pareto_entries.append(
                    {
                        "trial_index": index,
                        "objective_values": list(trial.values or ()),
                        "refusals": trial.user_attrs.get("refusals"),
                        "kl_divergence": trial.user_attrs.get("kl_divergence"),
                        "capability": trial.user_attrs.get("capability"),
                        "reproduction_file": f"pareto_trials/{filename}",
                    }
                )
            (output_dir / "abliteration_pareto_front.json").write_text(
                json.dumps(
                    {
                        "objective_names": evaluator.get_objective_names(),
                        "chosen_trial": chosen.user_attrs["index"],
                        "trials": pareto_entries,
                    },
                    indent=2,
                )
            )
            print(
                f"* Full ablation parameters saved to [bold]{output_dir / 'abliteration_reproduction.json'}[/] "
                f"({len(pareto_entries)} Pareto trials in [bold]{pareto_dir}[/])."
            )
        except Exception as e:
            print(f"[yellow]Could not write trial parameter files: {e}[/]")

        # Save metrics. Refusal counts are None when no refusal-measuring
        # scorer is configured (rendered as "n/a" downstream, not fake zeros).
        has_refusal_metrics = evaluator.refusal_prompt_total() > 0
        result = NativeAbliteratorResult(
            model_id=self.config.model,
            abliterated_model_path=str(output_dir),
            initial_refusals=evaluator.base_refusals if has_refusal_metrics else None,
            final_refusals=chosen.user_attrs["refusals"] if has_refusal_metrics else None,
            total_prompts=len(evaluator.bad_prompts) if has_refusal_metrics else None,
            kl_divergence=chosen.user_attrs["kl_divergence"],
            trials=len(study.trials),
            best_trial=chosen.user_attrs["index"],
            config=self.config.to_dict(),
        )
        result.config["export_strategy_effective"] = effective_strategy
        # also write metrics file like bridge does
        try:
            (output_dir / "native_abliteration_metrics.json").write_text(json.dumps(result.to_dict(), indent=2))
        except Exception:
            pass

        return result

    def run_reproduction(
        self,
        output_dir: Path | str,
        reproduction_information: dict,
    ) -> NativeAbliteratorResult:
        """
        Reproduction mode: instead of running the optimization, restore the
        ablation parameters stored in `reproduction_information` (a loaded
        reproduce.json file), re-apply the found ablation to the model, export
        it using the configured export strategy, and verify the SHA-256 hashes
        of the exported weight files against the recorded ones.

        The caller is responsible for building `self.config` from the settings
        stored in the reproduction information (see
        anlord.native.reproduce.settings_from_reproduction) and for running the
        environment check.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Restore native-level settings recorded with the original run (response
        # prefix, prompts, normalization, batch size, ...). Values recorded as
        # None (local paths) are skipped, keeping the local ones; display-only
        # diagnostics are likewise kept from the local configuration.
        native_overrides = reproduction_information.get("native_config") or {}
        if native_overrides:
            native_overrides = {
                key: value
                for key, value in native_overrides.items()
                if key not in _DISPLAY_ONLY_NATIVE_FIELDS
            }
            self.config.update_from_dict(native_overrides)

        self._setup_torch()

        parameters_info = reproduction_information["parameters"]
        direction_index = parameters_info["direction_index"]
        raw_parameters = parameters_info["abliteration_parameters"]
        parameters: dict[str, AbliterationParameters] = {
            component: AbliterationParameters(**values)
            for component, values in raw_parameters.items()
        }
        original_hashes: dict[str, str] = reproduction_information.get("hashes") or {}
        metrics = reproduction_information.get("metrics") or {}

        if self.config.seed is None:
            self.config.seed = random.randint(0, 2**32 - 1)
        set_seed(self.config.seed)

        # Model loading
        model = Model(self.config)

        # Whether refusal metrics are measurable at all (a keyword-rate-like
        # scorer is configured). Decides the n/a vs real-number reporting for
        # every result artifact below.
        # Behavioral verification support: when the reproduction information
        # contains recorded scores, load the same scorers so the reproduced
        # model's behavior can be compared against the original run. Baselines
        # are computed here, on the still-clean model.
        evaluator = None
        if reproduction_information.get("scores"):
            evaluator = Evaluator(self.config, model)

        print()
        print(f"Loading good prompts from [bold]{self.config.good_prompts.dataset}[/]...")
        good_prompts = load_prompts(self.config, self.config.good_prompts)
        print(f"* [bold]{len(good_prompts)}[/] prompts loaded")
        print(f"Loading bad prompts from [bold]{self.config.bad_prompts.dataset}[/]...")
        bad_prompts = load_prompts(self.config, self.config.bad_prompts)
        print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

        # Batch size: reproduction needs a batch size, but autotuning is not
        # reproducible; fall back to 1 if it was left at "auto".
        if self.config.batch_size == 0:
            print("* Batch size auto: using 1 for reproduction (autotuning is not deterministic)")
            self.config.batch_size = 1

        if self.config.response_prefix is None:
            # Reproduction must use the exact recorded prefix; if the settings
            # lost it, re-detect it deterministically from the model.
            self._maybe_detect_prefix(model, good_prompts, bad_prompts)

        print()
        print("Restoring model from reproduction information...")
        print("* Parameters:")
        for component, values in raw_parameters.items():
            for name, value in values.items():
                print(f"  * {component}.{name} = [bold]{value}[/]")
        print(f"  * direction_index = [bold]{direction_index if direction_index is not None else 'per layer'}[/]")

        refusal_directions = self._compute_refusal_directions(
            model, good_prompts, bad_prompts
        )

        print("* Resetting model...")
        model.reset_model()
        print("* Abliterating with restored parameters...")
        model.abliterate(refusal_directions, direction_index, parameters)

        # Behavioral verification: re-measure every recorded scorer on the
        # reproduced model and compare with the original run's numbers.
        score_verification = None
        if evaluator is not None and reproduction_information.get("scores"):
            print("* Verifying reproduced behavior against recorded scores...")
            scores = evaluator.get_scores()
            recorded = {s["name"]: s for s in reproduction_information["scores"]}
            score_verification = {"scores": {}, "baselines": {}, "all_match": True}

            def _compare(section, name, expected_value, got_value, expected_display, got_display):
                match = (
                    expected_value is not None
                    and got_value is not None
                    and math.isclose(float(got_value), float(expected_value), rel_tol=0.02, abs_tol=1e-6)
                )
                score_verification[section][name] = {
                    "expected": expected_value,
                    "got": got_value,
                    "expected_display": expected_display,
                    "got_display": got_display,
                    "match": match,
                }
                if not match:
                    score_verification["all_match"] = False
                status = "[green]MATCH[/]" if match else "[red]MISMATCH[/]"
                print(f"  * {name}: expected {expected_display}, reproduced {got_display} — {status}")

            # Clean-model baselines recorded with the original run (proves the
            # right base model was loaded) — except baselines that are 0 by
            # definition (e.g. KL divergence of a model against itself).
            for name, baseline_score in evaluator.baseline_scores:
                expected = (
                    recorded.get(name, {}).get("baseline", {}).get("value")
                )
                if expected == 0:
                    continue
                _compare(
                    "baselines", name, expected, baseline_score.value,
                    recorded.get(name, {}).get("baseline", {}).get("md_display", "?"),
                    baseline_score.md_display,
                )
            for name, score in scores:
                expected = recorded.get(name, {}).get("score", {}).get("value")
                _compare(
                    "scores", name, expected, score.value,
                    recorded.get(name, {}).get("score", {}).get("md_display", "?"),
                    score.md_display,
                )
            if score_verification["all_match"]:
                print("[green]Behavior verified: all reproduced scores match the original run.[/]")
            else:
                print("[yellow]Behavior differs from the original run — see MISMATCH lines above.[/]")

        strategy = self._export_model(model, output_dir)

        # Verify that the exported weights match the originally published ones.
        hash_verification = verify_model_hashes(output_dir, original_hashes)

        def _metric_or_none(key, cast):
            value = metrics.get(key)
            return cast(value) if value is not None else None

        total_prompts = (
            _metric_or_none("total_prompts", int)
            if metrics.get("total_prompts") is not None
            else len(bad_prompts)
        )
        result = NativeAbliteratorResult(
            model_id=self.config.model,
            abliterated_model_path=str(output_dir),
            initial_refusals=_metric_or_none("initial_refusals", int),
            final_refusals=_metric_or_none("final_refusals", int),
            total_prompts=total_prompts,
            kl_divergence=_metric_or_none("kl_divergence", float),
            trials=int(metrics.get("trials") or 0),
            best_trial=int(metrics.get("best_trial") or 0),
            config=self.config.to_dict(),
            hash_verification=hash_verification,
            reproduction_mode=True,
            score_verification=score_verification,
        )
        result.config["export_strategy_effective"] = (
            strategy.value if hasattr(strategy, "value") else str(strategy)
        )
        try:
            (output_dir / "native_abliteration_metrics.json").write_text(
                json.dumps(result.to_dict(), indent=2)
            )
        except Exception:
            pass

        matched = sum(1 for status in hash_verification.values() if status == "match")
        if original_hashes and matched == len(original_hashes):
            print("[bold green]Reproduction verified: all weight file hashes match.[/]")
        elif original_hashes:
            print(
                f"[yellow]Reproduction finished with hash differences "
                f"({matched}/{len(original_hashes)} files match).[/]"
            )

        return result
