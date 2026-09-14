# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-interactive bridge for the Heretic console application.

Heretic deliberately exposes a console entry point (``heretic.main:main``), not a
``heretic.__main__`` module.  It also asks the user to choose a Pareto-optimal
trial and an export action.  Anlord Abliterator runs Heretic in a subprocess, so this bridge
selects the best trial, exports it to the requested directory, and writes the
selected metrics in a small machine-readable file.

This module lives in a separate process on purpose: Heretic mutates global
state, parses ``sys.argv`` during settings construction, and keeps a large model
in memory.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


class _Answer:
    """Minimal questionary question object used by Heretic <= 1.0."""

    def __init__(self, value: Any):
        self.value = value

    def ask(self) -> Any:
        return self.value

    def unsafe_ask(self) -> Any:
        return self.value


def _choice_value(choice: Any) -> Any:
    return getattr(choice, "value", choice)


def _first_trial(choices: Iterable[Any]) -> Any | None:
    """Return the first actual Optuna trial from a list of menu choices."""
    for choice in choices:
        value = _choice_value(choice)
        if hasattr(value, "user_attrs"):
            return value
    return None


class _PromptController:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.selected_trial: Any | None = None
        self.model_saved = False

    def select(self, message: str, choices: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        """Answer Heretic menus deterministically."""
        del args, kwargs
        choices = list(choices)
        message_lower = message.lower()

        if "how would you like to proceed" in message_lower:
            # Resume Heretic's own valid checkpoint rather than throwing work away.
            values = [_choice_value(choice) for choice in choices]
            return "continue" if "continue" in values else values[0]

        if "which trial" in message_lower:
            if self.selected_trial is not None:
                return ""
            self.selected_trial = _first_trial(choices)
            return self.selected_trial or ""

        if "what do you want to do" in message_lower:
            if not self.model_saved:
                self.model_saved = True
                return "Save the model to a local folder"
            # Heretic 1.0 uses this exact string; newer versions accept an empty
            # value to return to the trial menu.
            values = [_choice_value(choice) for choice in choices]
            old_exit = "Nothing (return to trial selection menu)"
            return old_exit if old_exit in values else ""

        # This bridge never chooses upload/chat/benchmark actions.  Returning the
        # first choice is the safest forward-compatible behavior for a new prompt.
        return _choice_value(choices[0]) if choices else None

    def questionary_select(
        self, message: str, choices: Iterable[Any], *args: Any, **kwargs: Any
    ) -> _Answer:
        return _Answer(self.select(message, choices, *args, **kwargs))

    def path(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return str(self.output_dir)

    def questionary_path(self, *args: Any, **kwargs: Any) -> _Answer:
        return _Answer(self.path(*args, **kwargs))

    def metrics(self) -> dict[str, Any]:
        if self.selected_trial is None:
            return {}
        attrs = dict(getattr(self.selected_trial, "user_attrs", {}))
        return {
            "initial_refusals": attrs.get("base_refusals"),
            "final_refusals": attrs.get("refusals"),
            "total_prompts": attrs.get("n_bad_prompts"),
            "kl_divergence": attrs.get("kl_divergence"),
            "best_trial": attrs.get("index", 0),
        }


def _patch_prompts(heretic_main: Any, controller: _PromptController) -> None:
    """Patch both the modern Heretic helpers and the legacy questionary calls."""
    heretic_main.prompt_select = controller.select
    heretic_main.prompt_path = controller.path

    questionary = getattr(heretic_main, "questionary", None)
    if questionary is not None:
        questionary.select = controller.questionary_select
        questionary.path = controller.questionary_path


def _heretic_uses_mixed_offload() -> bool:
    device_map = os.environ.get("HERETIC_DEVICE_MAP", "auto")
    return device_map != "cuda" or bool(os.environ.get("HERETIC_MAX_MEMORY"))


def enable_quantized_cpu_offload(heretic_model: Any) -> None:
    """Allow bitsandbytes 4-bit layers to live on CPU when VRAM is too small.

    Heretic's BitsAndBytesConfig omits ``llm_int8_enable_fp32_cpu_offload``.
    Without it, Transformers refuses ``device_map=auto`` as soon as any module
    is placed on CPU or disk — the exact failure on an 8 GB laptop GPU.

    Do not enable this for GPU-only ``device_map=cuda`` loads: on Windows that
    mixed path is what access-violates with 0xC0000005.
    """
    original = getattr(heretic_model.Model, "_get_quantization_config", None)
    if original is None or getattr(original, "_anlord_cpu_offload", False):
        return

    mixed = _heretic_uses_mixed_offload()

    def _get_quantization_config(self, dtype: str) -> Any:
        config = original(self, dtype)
        if config is None:
            return None
        if hasattr(config, "llm_int8_enable_fp32_cpu_offload"):
            config.llm_int8_enable_fp32_cpu_offload = mixed
        if hasattr(config, "llm_int8_skip_modules"):
            skipped = list(getattr(config, "llm_int8_skip_modules", None) or [])
            for name in ("visual", "model.visual", "vision_tower", "multi_modal_projector"):
                if name not in skipped:
                    skipped.append(name)
            config.llm_int8_skip_modules = skipped
        if os.name == "nt" and hasattr(config, "bnb_4bit_use_double_quant"):
            # Extra FP32 absmax buffers push an 8 GB Windows load into 0xC0000005.
            config.bnb_4bit_use_double_quant = False
        return config

    _get_quantization_config._anlord_cpu_offload = True  # type: ignore[attr-defined]
    heretic_model.Model._get_quantization_config = _get_quantization_config


def ensure_chat_template() -> None:
    """Patch AutoTokenizer.from_pretrained to inject a chat template when missing.

    Some models (e.g. google/gemma-4-E2B) ship without a ``chat_template`` in
    their tokenizer config.  Heretic's ``generate()`` unconditionally calls
    ``tokenizer.apply_chat_template()``, which raises when the attribute is
    ``None``.  This patch sets a reasonable default for known model families.
    """
    from transformers import AutoTokenizer

    _GEMMA_TEMPLATE = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
        "{% elif message['role'] == 'model' %}"
        "<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
        "{% elif message['role'] == 'system' %}"
        "{{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )

    _TEMPLATES: dict[str, str] = {
        "google": _GEMMA_TEMPLATE,
    }

    original_from_pretrained = AutoTokenizer.from_pretrained

    @staticmethod  # type: ignore[misc]
    def patched_from_pretrained(*args: Any, **kwargs: Any) -> Any:
        tokenizer = original_from_pretrained(*args, **kwargs)
        if tokenizer.chat_template is not None:
            return tokenizer
        model_id = str(args[0]) if args else str(kwargs.get("pretrained_model_name_or_path", ""))
        for prefix, template in _TEMPLATES.items():
            if model_id.startswith(prefix):
                print(f"* Setting missing chat_template for {prefix} model")
                tokenizer.chat_template = template
                break
        return tokenizer

    AutoTokenizer.from_pretrained = patched_from_pretrained  # type: ignore[assignment]


def force_text_only_causal_lm(heretic_model: Any) -> None:
    """Load Qwen3.5-VL checkpoints as CausalLM so the vision tower is skipped.

    Heretic 1.4.0 sees ``vision_config`` and uses AutoModelForImageTextToText
    (~760 tensors). The same Ornith-1.5-9B checkpoint loads as CausalLM with
    ~427 tensors and fits 4-bit on an 8 GB laptop. Abliteration only needs the
    language layers.
    """
    original = getattr(heretic_model, "get_model_class", None)
    if original is None or getattr(original, "_anlord_text_only", False):
        return

    def get_model_class(model: str) -> Any:
        del model
        from transformers import AutoModelForCausalLM

        print("* Loading text-only CausalLM (vision tower skipped)")
        return AutoModelForCausalLM

    get_model_class._anlord_text_only = True  # type: ignore[attr-defined]
    heretic_model.get_model_class = get_model_class


def enable_weight_placement_check(heretic_model: Any) -> None:
    """Task 7: after from_pretrained, log device distribution and warn on CPU offload."""
    original_init = getattr(heretic_model.Model, "__init__", None)
    if original_init is None or getattr(original_init, "_anlord_weight_check", False):
        return

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        try:
            model = getattr(self, "model", None)
            if model is None:
                return
            devices: dict[str, int] = {}
            total = 0
            for _name, param in model.named_parameters():
                dev = str(param.device)
                devices[dev] = devices.get(dev, 0) + param.numel()
                total += param.numel()
            # Also check buffers? parameters covers most
            for dev, cnt in sorted(devices.items()):
                share = 100 * cnt / total if total else 0
                print(f"* Weight placement: {dev}: {cnt/1e9:.3f}B ({share:.1f}%)")
            cpu_cnt = devices.get("cpu", 0)
            # also check for device string containing cpu
            for k in list(devices.keys()):
                if k.startswith("cpu") and k != "cpu":
                    cpu_cnt += devices[k]
            if total and cpu_cnt:
                share = 100 * cpu_cnt / total
                if share > 5:
                    # Check CUDA available without creating context? Use torch.cuda.is_available via initialized check
                    try:
                        import torch
                        # is_initialized avoids creating context if not already
                        init = getattr(torch.cuda, "is_initialized", None)
                        cuda = False
                        if callable(init):
                            cuda = init() or torch.cuda.is_available()
                        else:
                            cuda = torch.cuda.is_available()
                        if cuda:
                            print(
                                f"* WARNING: {share:.1f}% weights are on CPU while CUDA is available — "
                                "each forward will drag weights via PCIe (slow). Remove max_memory or increase GPU limit."
                            )
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            print(f"* Weight placement check failed: {exc}")

    _patched_init._anlord_weight_check = True  # type: ignore[attr-defined]
    heretic_model.Model.__init__ = _patched_init  # type: ignore[assignment]


def enable_low_ram_lora_merge(heretic_model: Any) -> None:
    """Drop GPU 4-bit weights before Heretic reloads BF16 on CPU to merge LoRA."""
    original = getattr(heretic_model.Model, "get_merged_model", None)
    if original is None or getattr(original, "_anlord_low_ram_merge", False):
        return

    def get_merged_model(self) -> Any:
        import transformers

        originals: dict[str, Any] = {}

        def wrap(orig: Any) -> Any:
            def from_pretrained(*args: Any, **kwargs: Any) -> Any:
                if getattr(self, "model", None) is not None:
                    print("* Releasing GPU 4-bit weights before CPU BF16 merge")
                    self.model = None
                    gc.collect()
                    try:
                        import torch

                        initialized = getattr(torch.cuda, "is_initialized", None)
                        if callable(initialized) and initialized():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                kwargs.setdefault("low_cpu_mem_usage", True)
                return orig(*args, **kwargs)

            return from_pretrained

        for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            cls = getattr(transformers, name, None)
            if cls is None or not hasattr(cls, "from_pretrained"):
                continue
            originals[name] = cls.from_pretrained
            cls.from_pretrained = staticmethod(wrap(originals[name]))
        try:
            return original(self)
        finally:
            for name, orig in originals.items():
                getattr(transformers, name).from_pretrained = orig

    get_merged_model._anlord_low_ram_merge = True  # type: ignore[attr-defined]
    heretic_model.Model.get_merged_model = get_merged_model


class _PlainProgress:
    """Carriage-return progress that stays visible when stdout is piped."""

    def __init__(self, total: int, desc: str) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.n = 0
        self._started = time.monotonic()
        self._last_draw = 0.0
        self._draw(force=True)

    def update(self, n: int = 1) -> None:
        self.n = min(self.n + int(n), self.total) if self.total else self.n + int(n)
        self._draw()

    def close(self) -> None:
        self._draw(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _draw(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_draw < 0.2:
            return
        self._last_draw = now
        total = self.total or max(self.n, 1)
        ratio = min(1.0, self.n / total) if total else 0.0
        width = 28
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = now - self._started
        if self.n and elapsed > 0 and self.n < total:
            remaining = elapsed * (total - self.n) / self.n
            eta = f" ETA {remaining:0.0f}s"
        else:
            eta = ""
        sys.stdout.write(
            f"\r{self.desc}: {self.n}/{self.total} [{bar}] {ratio * 100:5.1f}%  "
            f"{elapsed:0.0f}s{eta}"
        )
        sys.stdout.flush()


def make_prompt_progress(total: int, desc: str) -> Any:
    """Prefer tqdm, but never hide the bar just because stdout is not a TTY."""
    try:
        from tqdm import tqdm

        return tqdm(
            total=total,
            desc=desc,
            unit="prompt",
            file=sys.stdout,
            disable=False,
            dynamic_ncols=False,
            ncols=88,
            mininterval=0.2,
            ascii=True,
            leave=True,
        )
    except Exception:
        return _PlainProgress(total, desc)


def enable_generation_progress(heretic_model: Any) -> None:
    """Show a live progress bar for Heretic's silent batched generation."""
    original_responses_batched = getattr(heretic_model.Model, "get_responses_batched", None)
    original_responses = getattr(heretic_model.Model, "get_responses", None)
    original_residuals_batched = getattr(heretic_model.Model, "get_residuals_batched", None)
    original_residuals = getattr(heretic_model.Model, "get_residuals", None)
    if original_responses_batched is None or getattr(
        original_responses_batched, "_anlord_progress", False
    ):
        return

    state: dict[str, Any] = {"bar": None}

    def _update(count: int) -> None:
        bar = state["bar"]
        if bar is not None:
            bar.update(count)

    def _run_batched(original, prompts, desc, args, kwargs):
        items = list(prompts)
        bar = make_prompt_progress(len(items), desc)
        state["bar"] = bar
        try:
            return original(items, *args, **kwargs)
        finally:
            bar.close()
            state["bar"] = None

    if original_responses is not None:

        def get_responses(self, prompts, *args, **kwargs):
            batch = list(prompts)
            result = original_responses(self, batch, *args, **kwargs)
            _update(len(batch))
            return result

        heretic_model.Model.get_responses = get_responses

    def get_responses_batched(self, prompts, *args, **kwargs):
        return _run_batched(
            lambda items, *inner_args, **inner_kwargs: original_responses_batched(
                self, items, *inner_args, **inner_kwargs
            ),
            prompts,
            "Generating responses",
            args,
            kwargs,
        )

    get_responses_batched._anlord_progress = True  # type: ignore[attr-defined]
    heretic_model.Model.get_responses_batched = get_responses_batched

    if original_residuals_batched is not None and original_residuals is not None:

        def get_residuals(self, prompts, *args, **kwargs):
            batch = list(prompts)
            result = original_residuals(self, batch, *args, **kwargs)
            _update(len(batch))
            return result

        def get_residuals_batched(self, prompts, *args, **kwargs):
            return _run_batched(
                lambda items, *inner_args, **inner_kwargs: original_residuals_batched(
                    self, items, *inner_args, **inner_kwargs
                ),
                prompts,
                "Computing residuals",
                args,
                kwargs,
            )

        heretic_model.Model.get_residuals = get_residuals
        heretic_model.Model.get_residuals_batched = get_residuals_batched


def _prepare_optional_dependencies() -> None:
    """Load runtime guards by file path so source-layout execution also works."""
    runtime_path = Path(__file__).resolve().parents[1] / "utils" / "runtime.py"
    spec = importlib.util.spec_from_file_location("_anlord_runtime", runtime_path)
    if spec is None or spec.loader is None:
        return
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.prepare_optional_ml_dependencies()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    _prepare_optional_dependencies()

    # Importing heretic.main is the supported Python equivalent of its console
    # entry point.  In particular, do not use ``python -m heretic``: the package
    # intentionally has no __main__.py.
    try:
        import heretic.main as heretic_main
        import heretic.model as heretic_model
    except ImportError as error:
        print(f"Heretic is not installed: {error}", file=sys.stderr)
        return 2

    ensure_chat_template()
    enable_quantized_cpu_offload(heretic_model)
    force_text_only_causal_lm(heretic_model)
    enable_low_ram_lora_merge(heretic_model)
    enable_weight_placement_check(heretic_model)
    enable_generation_progress(heretic_model)
    controller = _PromptController(args.output_dir or Path("heretic_output"))
    if not args.evaluate_only:
        _patch_prompts(heretic_main, controller)

    # Heretic's Settings parses the process command line.  Configuration is
    # supplied through HERETIC_* environment variables by HereticWrapper.
    sys.argv = ["heretic"]
    heretic_main.main()

    metrics = controller.metrics()
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not args.evaluate_only and not controller.model_saved:
        print("Heretic finished without exporting a model.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
