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


class NativeAbliteratorResult:
    def __init__(
        self,
        model_id: str,
        abliterated_model_path: Optional[str] = None,
        initial_refusals: int = 0,
        final_refusals: int = 0,
        total_prompts: int = 0,
        kl_divergence: float = 0.0,
        trials: int = 0,
        best_trial: int = 0,
        config: dict | None = None,
        error: Optional[str] = None,
        hash_verification: Optional[dict[str, str]] = None,
        reproduction_mode: bool = False,
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

        # If resuming, Abliteration reloads settings from study. We do the same for parity
        # but also keep Anlord-specific fields (model, cache). Simplified: if study
        # exists and not finished, we reuse its settings json but preserve n_trials.
        # For native we always allow resume; if study finished, we will reuse it.
        # This matches Anlord's AbliterationWrapper which handles resume via study file.
        if existing is not None:
            # Check finished flag
            try:
                finished = existing.user_attrs.get("finished", False)
            except Exception:
                finished = False
            # If settings stored, we could load them, but for native we keep current config
            # to honor Anlord's model/cache overrides. This is intentional divergence
            # documented in docs/abliteration_implementation.md
            pass

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

        # Capability proxy (feature 2): tiny MMLU to preserve capability
        capability_proxy = None
        if getattr(self.config, "capability_proxy_enabled", False):
            try:
                capability_proxy = self._load_capability_proxy()
                print(f"* Capability proxy: {self.config.capability_proxy_dataset}/{self.config.capability_proxy_subset} ({len(capability_proxy)} samples)")
                base_cap = self._score_capability(model, capability_proxy)
                print(f"* Base capability (proxy MMLU): {base_cap:.3f}")
                evaluator.base_capability = base_cap  # type: ignore
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
        study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=128,
                multivariate=True,
                seed=self.config.seed,
            ),
            directions=study_directions,
            storage=storage,
            study_name="abliteration",
            load_if_exists=True,
        )
        # persist settings json for resume parity
        try:
            # NativeConfig -> dict -> json
            import json as _json

            study.set_user_attr("settings", _json.dumps(self.config.to_dict()))
            study.set_user_attr("finished", False)
        except Exception:
            pass

        start_index = len(study.trials)
        trial_index = start_index
        start_time = time.perf_counter()
        if start_index > 0:
            print()
            print("Resuming existing study.")

        # Objective — scorers provide the scores; capability proxy may add one more
        def objective(trial: Trial) -> tuple[float, ...]:
            nonlocal trial_index
            trial_index += 1
            trial.set_user_attr("index", trial_index)

            direction_scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
            last_layer_index = len(model.get_layers()) - 1
            direction_index = trial.suggest_float("direction_index", 0.4 * last_layer_index, 0.9 * last_layer_index)
            if direction_scope == "per layer":
                direction_index = None

            parameters: dict[str, AbliterationParameters] = {}
            for component in model.get_abliterable_components():
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
                max_weight = max(
                    0.0,
                    trial.suggest_float(f"{component}.max_weight", max_weight_lower_bound, 1.5),
                )
                max_weight_position = trial.suggest_float(f"{component}.max_weight_position", 0.6 * last_layer_index, 1.0 * last_layer_index)
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
            scores = evaluator.get_scores()
            objective_values = list(evaluator.get_objective_values(scores))
            for name, score in scores:
                print(f"  * [bold]{name}:[/] [green]{score.rich_display}[/]")

            elapsed = time.perf_counter() - start_time
            remaining = (elapsed / (trial_index - start_index)) * (self.config.n_trials - trial_index) if (trial_index - start_index) > 0 else 0
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

            # Legacy metric attributes (plain refusals/KL) for pipeline compatibility.
            kl_divergence = evaluator.last_kl_divergence if evaluator.last_kl_divergence is not None else 0.0
            refusals = evaluator.last_refusals if evaluator.last_refusals is not None else 0
            trial.set_user_attr("kl_divergence", kl_divergence)
            trial.set_user_attr("refusals", refusals)
            trial.set_user_attr("base_refusals", evaluator.base_refusals)
            trial.set_user_attr("n_bad_prompts", len(evaluator.bad_prompts))

            return tuple(objective_values)

        def objective_wrapper(trial: Trial) -> tuple[float, float] | tuple[float, float, float]:
            try:
                return objective(trial)
            except KeyboardInterrupt:
                trial.study.stop()
                raise TrialPruned()

        n_needed = self.config.n_trials - len(study.trials)
        if n_needed > 0:
            # timeout handling
            try:
                study.optimize(objective_wrapper, n_trials=n_needed, timeout=timeout)
            except KeyboardInterrupt:
                pass

        if len(study.trials) == self.config.n_trials:
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
        print(f"* Best trial: [bold]{chosen.user_attrs['index']}[/] refusals {chosen.user_attrs['refusals']}/{len(evaluator.bad_prompts)} KL {chosen.user_attrs['kl_divergence']:.4f}{cap_str}")

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
                        "initial_refusals": evaluator.base_refusals,
                        "final_refusals": chosen.user_attrs["refusals"],
                        "total_prompts": len(evaluator.bad_prompts),
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
                "initial_refusals": evaluator.base_refusals,
                "total_prompts": len(evaluator.bad_prompts),
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

        # Save metrics
        result = NativeAbliteratorResult(
            model_id=self.config.model,
            abliterated_model_path=str(output_dir),
            initial_refusals=evaluator.base_refusals,
            final_refusals=chosen.user_attrs["refusals"],
            total_prompts=len(evaluator.bad_prompts),
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

        strategy = self._export_model(model, output_dir)

        # Verify that the exported weights match the originally published ones.
        hash_verification = verify_model_hashes(output_dir, original_hashes)

        total_prompts = int(metrics.get("total_prompts") or len(bad_prompts))
        result = NativeAbliteratorResult(
            model_id=self.config.model,
            abliterated_model_path=str(output_dir),
            initial_refusals=int(metrics.get("initial_refusals") or 0),
            final_refusals=int(metrics.get("final_refusals") or 0),
            total_prompts=total_prompts,
            kl_divergence=float(metrics.get("kl_divergence") or 0.0),
            trials=int(metrics.get("trials") or 0),
            best_trial=int(metrics.get("best_trial") or 0),
            config=self.config.to_dict(),
            hash_verification=hash_verification,
            reproduction_mode=True,
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
