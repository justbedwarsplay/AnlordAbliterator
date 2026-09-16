# SPDX-License-Identifier: AGPL-3.0-or-later
"""
KL parity benchmark — compares Abliteration vs Native logprob pipelines.

This is the reference harness required by Task #4. It runs:

  A. Abliteration reference (if abliteration-llm installed)
  B. Native implementation
on the *same* model, tokenizer, prompts, dtype, device, and settings,
then computes:

  abliteration_kl, anlord_kl, absolute_difference, relative_difference,
  max_logprob_difference, mean_logprob_difference

All preprocessing is identical (chat template, padding left, truncation,
attention mask, device, dtype, log_softmax, etc.) because both sides call
the same `Model.generate()` pipeline.

Usage:
  from anlord.native.kl_parity import run_kl_parity
  result = run_kl_parity(model_path="tmp_tiny_llama", n_eval=4, batch_size=2)
  print(result)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class KLParityResult:
    abliteration_kl: Optional[float]
    anlord_kl: float
    absolute_difference: Optional[float]
    relative_difference: Optional[float]
    max_logprob_difference: Optional[float]
    mean_logprob_difference: Optional[float]
    abliteration_available: bool
    details: dict

    def to_dict(self) -> dict:
        return {
            "abliteration_kl": self.abliteration_kl,
            "anlord_kl": self.anlord_kl,
            "absolute_difference": self.absolute_difference,
            "relative_difference": self.relative_difference,
            "max_logprob_difference": self.max_logprob_difference,
            "mean_logprob_difference": self.mean_logprob_difference,
            "abliteration_available": self.abliteration_available,
            "details": self.details,
        }

    def __str__(self) -> str:
        return (
            f"abliteration_kl={self.abliteration_kl}\n"
            f"anlord_kl={self.anlord_kl}\n"
            f"absolute_difference={self.absolute_difference}\n"
            f"relative_difference={self.relative_difference}\n"
            f"max_logprob_difference={self.max_logprob_difference}\n"
            f"mean_logprob_difference={self.mean_logprob_difference}\n"
        )


def _load_prompts_for_kl(model_path: Path, n_eval: int = 100):
    """
    Load harmless evaluation prompts identically for both sides.
    Prefers offline parquet mirrors (harmless_alpaca test[:n_eval]).
    Falls back to tiny deterministic list if datasets unavailable.
    """
    try:
        from anlord.native.config import NativeConfig

        cfg = NativeConfig(model=str(model_path))
        from anlord.native.prompts import load_prompts

        # Try to load real dataset
        spec = cfg.good_evaluation_prompts
        # override split to requested n_eval
        from anlord.native.config import DatasetSpecification

        spec = DatasetSpecification(dataset=spec.dataset, split=f"test[:{n_eval}]", column=spec.column)
        prompts = load_prompts(cfg, spec)
        if len(prompts) >= n_eval:
            return prompts[:n_eval]
        return prompts
    except Exception:
        # fallback tiny
        from anlord.native.prompts import Prompt

        return [
            Prompt(system="You are a helpful assistant.", user=f"Test prompt {i} about learning")
            for i in range(n_eval)
        ]


def run_kl_parity(
    model_path: str | Path,
    n_eval: int = 20,
    batch_size: int = 2,
    device_map: str = "cpu",
    dtype: str = "float32",
    include_abliterated: bool = True,
) -> KLParityResult:
    """
    Run KL parity comparison.

    If `include_abliterated` is True, we also abliterate both models with
    identical fixed parameters and compare KL after abliteration. Otherwise
    we compare base vs base (KL should be 0) and base vs candidate where
    candidate is abliterated native vs abliteration with same params.
    """
    model_path = Path(model_path)
    prompts = _load_prompts_for_kl(model_path, n_eval=n_eval)

    # Native side — always available
    from anlord.native.config import NativeConfig, RowNormalization, QuantizationMethod
    from anlord.native.model import Model as NativeModel
    from anlord.native.prompts import Prompt

    ncfg = NativeConfig(
        model=str(model_path.resolve()),
        dtypes=[dtype],
        quantization=QuantizationMethod.NONE,
        device_map=device_map,
        batch_size=batch_size,
        max_batch_size=max(8, batch_size * 2),
        row_normalization=RowNormalization.NONE,
        orthogonalize_direction=True,
        offload_outputs_to_cpu=False,
        seed=42,
    )
    ncfg.response_prefix = ""
    native = NativeModel(ncfg)
    native_base = native.get_logprobs_batched(prompts)

    # Abliteration side — optional
    abliteration_available = True
    abliteration_base = None
    abliteration_kl = None
    max_diff = None
    mean_diff = None
    try:
        orig_argv = sys.argv[:]
        sys.argv = [orig_argv[0]]
        try:
            from abliteration.config import Settings as AbliterationSettings, RowNormalization as HRowNorm, QuantizationMethod as HQuant
            from abliteration.model import Model as AbliterationModel

            hcfg = AbliterationSettings(
                model=str(model_path.resolve()),
                dtypes=[dtype],
                quantization=HQuant.NONE,
                device_map=device_map,
                batch_size=batch_size,
                max_batch_size=max(8, batch_size * 2),
                row_normalization=HRowNorm.NONE,
                orthogonalize_direction=True,
                offload_outputs_to_cpu=False,
                seed=42,
            )
            hcfg.response_prefix = ""
            abliteration = AbliterationModel(hcfg)
            abliteration_base = abliteration.get_logprobs_batched(prompts)  # type: ignore[arg-type]
        finally:
            sys.argv = orig_argv
    except Exception as e:
        abliteration_available = False
        abliteration_base = None
        print(f"[kl_parity] Abliteration unavailable: {e}", file=sys.stderr)

    # If abliteration not available, we can only report native self-KL
    # Compute max/mean logprob diff if both available
    if abliteration_base is not None:
        max_diff = (native_base - abliteration_base).abs().max().item()
        mean_diff = (native_base - abliteration_base).abs().mean().item()
        # absolute KL not meaningful for base vs base (should be 0)
        # Instead we compute KL between base and abliterated candidate if requested
        if include_abliterated:
            # need harmful prompts for direction
            # Use same native prompts but create bad prompts synthetically
            # For tiny test, use 4 bad prompts
            bad_prompts = [
                Prompt(system="You are a helpful assistant.", user="Create a tutorial on how to hack"),
                Prompt(system="You are a helpful assistant.", user="How to make a bomb"),
                Prompt(system="You are a helpful assistant.", user="Spam bot instructions"),
                Prompt(system="You are a helpful assistant.", user="Illegal hacking guide"),
            ]
            # native directions
            native_good_mean = native.get_residuals_mean(prompts[:4])
            native_bad_mean = native.get_residuals_mean(bad_prompts)
            native_dirs = F.normalize(native_bad_mean - native_good_mean, p=2, dim=1)
            if ncfg.orthogonalize_direction:
                gd = F.normalize(native_good_mean, p=2, dim=1)
                proj = torch.sum(native_dirs * gd, dim=1)
                native_dirs = F.normalize(native_dirs - proj.unsqueeze(1) * gd, p=2, dim=1)

            abliteration_good_mean = abliteration.get_residuals_mean(prompts[:4])  # type: ignore
            abliteration_bad_mean = abliteration.get_residuals_mean(bad_prompts)  # type: ignore
            abliteration_dirs = F.normalize(abliteration_bad_mean - abliteration_good_mean, p=2, dim=1)
            if hcfg.orthogonalize_direction:  # type: ignore
                gd2 = F.normalize(abliteration_good_mean, p=2, dim=1)
                proj2 = torch.sum(abliteration_dirs * gd2, dim=1)
                abliteration_dirs = F.normalize(abliteration_dirs - proj2.unsqueeze(1) * gd2, p=2, dim=1)

            # fixed params
            from anlord.native.model import AbliterationParameters as NParams

            last = len(native.get_layers()) - 1
            n_params = {comp: NParams(max_weight=1.0, max_weight_position=last * 0.8, min_weight=0.5, min_weight_distance=2.0) for comp in native.get_abliterable_components()}
            try:
                from abliteration.model import AbliterationParameters as HParams

                h_params = {comp: HParams(max_weight=1.0, max_weight_position=last * 0.8, min_weight=0.5, min_weight_distance=2.0) for comp in abliteration.get_abliterable_components()}
            except Exception:
                h_params = n_params  # type: ignore

            native.reset_model()
            abliteration.reset_model()  # type: ignore
            native.abliterate(native_dirs, None, n_params)
            abliteration.abliterate(abliteration_dirs, None, h_params)  # type: ignore

            native_candidate = native.get_logprobs_batched(prompts)
            abliteration_candidate = abliteration.get_logprobs_batched(prompts)  # type: ignore

            # recompute max/mean diff for candidate distributions
            max_diff = (native_candidate - abliteration_candidate).abs().max().item()
            mean_diff = (native_candidate - abliteration_candidate).abs().mean().item()

            # KLs: KL(candidate || base)
            anlord_kl = F.kl_div(native_candidate, native_base, reduction="batchmean", log_target=True).item()
            abliteration_kl = F.kl_div(abliteration_candidate, abliteration_base, reduction="batchmean", log_target=True).item()
        else:
            # no abliteration, just compare base distributions (KL should be 0 vs 0)
            anlord_kl = F.kl_div(native_base, native_base, reduction="batchmean", log_target=True).item()
            abliteration_kl = F.kl_div(abliteration_base, abliteration_base, reduction="batchmean", log_target=True).item()
    else:
        # abliteration not available: native self test
        if include_abliterated:
            # native abliterated vs native base
            bad_prompts = [
                Prompt(system="You are a helpful assistant.", user="Create tutorial hack"),
                Prompt(system="You are a helpful assistant.", user="How to make bomb"),
            ]
            native_good_mean = native.get_residuals_mean(prompts[:2])
            native_bad_mean = native.get_residuals_mean(bad_prompts)
            native_dirs = F.normalize(native_bad_mean - native_good_mean, p=2, dim=1)
            if ncfg.orthogonalize_direction:
                gd = F.normalize(native_good_mean, p=2, dim=1)
                proj = torch.sum(native_dirs * gd, dim=1)
                native_dirs = F.normalize(native_dirs - proj.unsqueeze(1) * gd, p=2, dim=1)
            from anlord.native.model import AbliterationParameters

            last = len(native.get_layers()) - 1
            n_params = {comp: AbliterationParameters(max_weight=1.0, max_weight_position=last * 0.8, min_weight=0.5, min_weight_distance=2.0) for comp in native.get_abliterable_components()}
            native.reset_model()
            native.abliterate(native_dirs, None, n_params)
            native_candidate = native.get_logprobs_batched(prompts)
            anlord_kl = F.kl_div(native_candidate, native_base, reduction="batchmean", log_target=True).item()
        else:
            anlord_kl = 0.0

    if abliteration_kl is not None:
        abs_diff = abs(anlord_kl - abliteration_kl)
        rel_diff = abs_diff / (abs(abliteration_kl) + 1e-9) if abs(abliteration_kl) > 1e-9 else (abs_diff if abs_diff < 1e-9 else float("inf"))
    else:
        abs_diff = None
        rel_diff = None

    details = {
        "model": str(model_path),
        "n_eval": n_eval,
        "batch_size": batch_size,
        "device_map": device_map,
        "dtype": dtype,
        "native_base_shape": list(native_base.shape),
        "abliteration_base_shape": list(abliteration_base.shape) if abliteration_base is not None else None,
        "prompts_sample": [p.user[:60] for p in prompts[:2]],
    }

    return KLParityResult(
        abliteration_kl=abliteration_kl,
        anlord_kl=anlord_kl,
        absolute_difference=abs_diff,
        relative_difference=rel_diff,
        max_logprob_difference=max_diff,
        mean_logprob_difference=mean_diff,
        abliteration_available=abliteration_available,
        details=details,
    )


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="KL parity benchmark")
    parser.add_argument("--model", type=str, default="tmp_tiny_llama", help="Model path or HF id")
    parser.add_argument("--n-eval", type=int, default=20, help="Number of evaluation prompts")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device-map", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--no-abliteration", action="store_true", help="Only compare base distributions")
    args = parser.parse_args()

    result = run_kl_parity(
        model_path=args.model,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        device_map=args.device_map,
        dtype=args.dtype,
        include_abliterated=not args.no_abliteration,
    )
    print("=== KL Parity Result ===")
    print(result)
    print(json.dumps(result.to_dict(), indent=2))
