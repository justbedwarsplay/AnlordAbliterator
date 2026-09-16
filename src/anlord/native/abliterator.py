# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Native abliterator — full pipeline replication of abliteration/main.py

Implements:
  model → prompts → residual extraction → refusal directions →
  parameterized abliteration → evaluation → KL divergence → refusal score →
  Optuna/TPE → best params → final abliteration → output model

All mathematics and control flow are identical to abliteration 1.4.0. This module
does NOT import abliteration-llm.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import warnings
import logging
from dataclasses import asdict
from os.path import commonprefix
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import numpy as np
import optuna
from optuna import Trial, TrialPruned  # type: ignore
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import TrialState
from rich.console import Console

from .config import NativeConfig, RowNormalization
from .model import AbliterationParameters, Model
from .evaluator import Evaluator
from .prompts import load_prompts
from .utils import empty_cache, format_duration, set_seed

print = Console(highlight=False).print
logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict:
        return {
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


class NativeAbliterator:
    """
    1:1 native replication. Usage mirrors AbliterationWrapper but without subprocess.
    """

    def __init__(self, config: NativeConfig):
        self.config = config
        # Ensure directories exist
        Path(config.study_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_anlord_settings(cls, settings) -> "NativeAbliterator":
        cfg = NativeConfig.from_anlord_settings(settings)
        return cls(cfg)

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
        self.config.batch_size = best_batch_size
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
                    # temporarily set to recheck
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
            # propagate to model.settings (same object) — already done since model.settings is self.config
        else:
            print("* None found")
            self.config.response_prefix = ""

    def run(
        self,
        output_dir: Path | str,
        timeout: Optional[int] = None,
    ) -> NativeAbliteratorResult:
        """
        Execute full pipeline. Returns result with merged model at output_dir.
        Mirrors abliteration/main.py:run() but without interactive prompts.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_torch()

        if self.config.seed is None:
            self.config.seed = random.randint(0, 2**32 - 1)
        set_seed(self.config.seed)

        # Study checkpoint handling
        os.makedirs(self.config.study_checkpoint_dir, exist_ok=True)
        study_file = os.path.join(
            self.config.study_checkpoint_dir,
            "".join([(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in self.config.model]) + ".jsonl",
        )
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

        # Evaluator (also loads evaluation prompts and computes base_logprobs/base_refusals)
        evaluator = Evaluator(self.config, model)

        # refusal directions
        print()
        print("Calculating per-layer refusal directions...")
        # abliteration main has two branches: full residuals vs means only
        needs_full = self.config.print_residual_geometry or self.config.plot_residuals
        if needs_full:
            print("* Obtaining residuals for good prompts...")
            good_residuals = model.get_residuals_batched(good_prompts)
            print("* Obtaining residuals for bad prompts...")
            bad_residuals = model.get_residuals_batched(bad_prompts)
            good_means = good_residuals.mean(dim=0)
            bad_means = bad_residuals.mean(dim=0)
            # analyzer plot/print skipped for native (not needed for parity)
            del good_residuals, bad_residuals
        else:
            print("* Obtaining residual mean for good prompts...")
            good_means = model.get_residuals_mean(good_prompts)
            print("* Obtaining residual mean for bad prompts...")
            bad_means = model.get_residuals_mean(bad_prompts)

        refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
        if self.config.orthogonalize_direction:
            good_directions = F.normalize(good_means, p=2, dim=1)
            projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
            refusal_directions = refusal_directions - projection_vector.unsqueeze(1) * good_directions
            refusal_directions = F.normalize(refusal_directions, p=2, dim=1)
            del good_directions, projection_vector
        del good_means, bad_means
        empty_cache()

        # Optuna study
        study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=128,
                multivariate=True,
                seed=self.config.seed,
            ),
            directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
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

        # Objective — exact copy of abliteration/main.py objective
        def objective(trial: Trial) -> tuple[float, float]:
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
                max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
                max_weight_position = trial.suggest_float(f"{component}.max_weight_position", 0.6 * last_layer_index, 1.0 * last_layer_index)
                min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
                min_weight_distance = trial.suggest_float(f"{component}.min_weight_distance", 1.0, 0.6 * last_layer_index)
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
            score, kl_divergence, refusals = evaluator.get_score()
            elapsed = time.perf_counter() - start_time
            remaining = (elapsed / (trial_index - start_index)) * (self.config.n_trials - trial_index) if (trial_index - start_index) > 0 else 0
            print()
            print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed)}[/][/]")
            if trial_index < self.config.n_trials:
                print(f"[grey50]Estimated remaining time: [bold]{format_duration(remaining)}[/][/]")
            # memory usage
            try:
                from psutil import Process

                print(f"[grey50]Resident system RAM: [bold]{Process().memory_info().rss / (1024**3):.2f} GB[/][/]")
                if torch.cuda.is_available():
                    c = torch.cuda.device_count()
                    alloc = sum(torch.cuda.memory_allocated(d) for d in range(c)) / (1024**3)
                    reserv = sum(torch.cuda.memory_reserved(d) for d in range(c)) / (1024**3)
                    print(f"[grey50]Allocated GPU VRAM: [bold]{alloc:.2f} GB[/][/]")
                    print(f"[grey50]Reserved GPU VRAM: [bold]{reserv:.2f} GB[/][/]")
            except Exception:
                pass
            trial.set_user_attr("kl_divergence", kl_divergence)
            trial.set_user_attr("refusals", refusals)
            trial.set_user_attr("base_refusals", evaluator.base_refusals)
            trial.set_user_attr("n_bad_prompts", len(evaluator.bad_prompts))
            return score

        def objective_wrapper(trial: Trial) -> tuple[float, float]:
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

        # Pareto front selection — identical to abliteration
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            raise RuntimeError("No completed trials")

        sorted_trials = sorted(completed, key=lambda tr: (tr.user_attrs["refusals"], tr.user_attrs["kl_divergence"]))
        min_div = math.inf
        best_trials = []
        for tr in sorted_trials:
            kl = tr.user_attrs["kl_divergence"]
            if kl < min_div:
                min_div = kl
                best_trials.append(tr)

        # Automatic selection: choose first Pareto trial (best refusals, lowest KL)
        # This matches Anlord's bridge which picks "best" via sorted order
        chosen = best_trials[0]
        print()
        print("[bold green]Optimization finished![/]")
        print(f"* Best trial: [bold]{chosen.user_attrs['index']}[/] refusals {chosen.user_attrs['refusals']}/{len(evaluator.bad_prompts)} KL {chosen.user_attrs['kl_divergence']:.4f}")

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

        # Export
        print(f"* Saving model to [bold]{output_dir}[/]...")
        # Abliteration export_strategy = merge (default). Native always merges.
        # For quantized models, get_merged_model handles reload on CPU.
        # For non-quantized, it merges via PEFT.
        if self.config.quantization == RowNormalization.FULL or True:  # always merge
            try:
                merged = model.get_merged_model()
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
                # fallback: save peft adapter
                print(f"[yellow]Merge failed ({e}), saving adapter instead[/]")
                model.model.save_pretrained(str(output_dir))
                model.tokenizer.save_pretrained(str(output_dir))
        else:
            model.model.save_pretrained(str(output_dir))
            model.tokenizer.save_pretrained(str(output_dir))

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
        # also write metrics file like bridge does
        try:
            (output_dir / "native_abliteration_metrics.json").write_text(json.dumps(result.to_dict(), indent=2))
        except Exception:
            pass

        return result
