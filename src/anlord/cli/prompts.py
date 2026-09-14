# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Interactive prompts for Anlord Abliterator CLI.
Windows-friendly interface with terminal prompts.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    import questionary
    from questionary import Choice, Style

    QUESTIONARY_AVAILABLE = True
except ImportError:
    QUESTIONARY_AVAILABLE = False
    questionary = None

    class Choice:
        def __init__(self, title, value=None):
            self.title = title
            self.value = value if value is not None else title

    class Style:
        def __init__(self, rules=None):
            self.rules = rules or []


from ..config import RunMode, Settings

logger = logging.getLogger(__name__)

# Custom questionary style
CUSTOM_STYLE = Style(
    [
        ("qmark", "#00d4ff bold"),
        ("question", "#ffffff bold"),
        ("answer", "#00d4ff bold"),
        ("pointer", "#7c4dff bold"),
        ("highlighted", "#7c4dff bold"),
        ("selected", "#00d4ff"),
        ("separator", "#666666"),
        ("instruction", "#888888"),
        ("text", "#ffffff"),
    ]
)


class InteractivePrompter:
    """
    Interactive prompts for Anlord Abliterator configuration.

    Provides a Windows-friendly interface for setting up
    the abliteration pipeline.
    """

    def __init__(self):
        self.use_fallback = not QUESTIONARY_AVAILABLE

        if self.use_fallback:
            logger.warning("questionary not available, using basic input()")

    def prompt_model_id(self, default: str = "unsloth/gpt-oss-20b-BF16") -> str:
        """Prompt for model ID."""
        if self.use_fallback:
            return self._input_fallback("Enter HuggingFace model ID", default)

        return questionary.text(
            "Enter HuggingFace model ID:",
            default=default,
            style=CUSTOM_STYLE,
        ).ask()

    def prompt_output_dir(self, default: str = "./output") -> Path:
        """Prompt for output directory."""
        if self.use_fallback:
            path_str = self._input_fallback("Enter output directory", default)
            return Path(path_str)

        path_str = questionary.text(
            "Enter output directory:",
            default=default,
            style=CUSTOM_STYLE,
        ).ask()

        return Path(path_str)

    def prompt_benchmarks(self) -> list[str]:
        """Prompt for benchmark selection.

        New UX: first choose mode (All / None / Custom).  If Custom is chosen,
        a checkbox with individual benchmarks appears where each selection is
        immediate (Space) and a confirmation button appears at the bottom
        (Enter / second confirm), allowing an arbitrary custom list.
        """
        all_benchmarks = [
            "mmlu",
            "gsm8k",
            "hellaswag",
            "arc_challenge",
            "winogrande",
            "truthfulqa",
            "gpqa",
            "gpqa_diamond",
            "mmlu_pro",
            "ifstruct",
            "parsebench",
            "extractbench",
            "screenspot_pro",
            "mmmu_pro",
        ]
        benchmark_labels = {
            "mmlu": "MMLU (Multilingual)",
            "gsm8k": "GSM8K (Math)",
            "hellaswag": "HellaSwag (Reasoning)",
            "arc_challenge": "ARC-Challenge",
            "winogrande": "Winogrande",
            "truthfulqa": "TruthfulQA",
            "gpqa": "GPQA Diamond (Graduate QA) 11.9",
            "gpqa_diamond": "GPQA Diamond",
            "mmlu_pro": "MMLU-Pro 29.7",
            "ifstruct": "Ifstruct V1 (LiquidAI) 15.5",
            "parsebench": "ParseBench (Parse) 28.4",
            "extractbench": "ExtractBench 46.6",
            "screenspot_pro": "ScreenSpot-Pro 46.5",
            "mmmu_pro": "MMMU-Pro Vision 31.2",
        }

        if self.use_fallback:
            print("\nAvailable benchmarks:")
            for b in all_benchmarks:
                print(f"  - {b}  —  {benchmark_labels[b]}")
            print("  - all  (run all benchmarks)")
            print("  - none (skip all benchmarks)")
            print("  - or comma-separated custom list, e.g. mmlu,gsm8k")

            response = self._input_fallback(
                "Enter benchmarks (comma-separated, 'all', or 'none')", "none"
            )
            lowered = response.lower().strip()
            if lowered in {"none", "skip", "нет", "0", ""}:
                return []
            if lowered == "all":
                return all_benchmarks
            # custom list
            items = [item.strip().lower() for item in response.split(",") if item.strip()]
            # keep valid ones, allow full list
            valid = [x for x in items if x in all_benchmarks]
            # if user typed something invalid but not empty, return as-is (will be validated later)
            return valid if valid else items

        # Step 1: choose mode — All / None / Custom
        mode = questionary.select(
            "Benchmark selection:",
            choices=[
                Choice(title="✓  All benchmarks", value="all"),
                Choice(title="✗  Skip all benchmarks", value="none"),
                Choice(title="☰  Custom selection (choose manually)", value="custom"),
            ],
            style=CUSTOM_STYLE,
        ).ask()

        if mode == "all":
            return all_benchmarks
        if mode == "none" or mode is None:
            return []

        # Step 2: custom — checkbox where each Space directly selects,
        # and a confirmation button appears at the bottom (second confirm)
        while True:
            choices = [
                Choice(title=benchmark_labels[b], value=b) for b in all_benchmarks
            ]
            selected = questionary.checkbox(
                "Select benchmarks (Space to toggle, Enter to confirm):",
                choices=choices,
                style=CUSTOM_STYLE,
                validate=lambda ans: True if len(ans) > 0 else "Select at least one benchmark",
            ).ask()

            if not selected:
                # User pressed Enter with nothing selected — ask if they want to skip or retry
                if questionary.confirm(
                    "No benchmarks selected. Skip all benchmarks?",
                    default=False,
                    style=CUSTOM_STYLE,
                ).ask():
                    return []
                continue

            # Bottom confirmation button
            summary = ", ".join(selected)
            confirmed = questionary.confirm(
                f"Confirm selection: {summary} ?",
                default=True,
                style=CUSTOM_STYLE,
            ).ask()
            if confirmed:
                return selected
            # otherwise loop and let user re-pick

    def prompt_mode(self) -> RunMode:
        """Prompt for evaluation mode."""
        if self.use_fallback:
            print("\nEvaluation modes:")
            print("  1 - Quick (fast, limited samples)")
            print("  2 - Full (complete evaluation)")

            mode = self._input_fallback("Select mode (1/2)", "1")
            return RunMode.QUICK if mode == "1" else RunMode.FULL

        # Use simple text input to avoid default value issues
        mode_input = questionary.text(
            "Select mode (quick/full):",
            default="quick",
            style=CUSTOM_STYLE,
        ).ask()

        return RunMode.QUICK if mode_input.lower() == "quick" else RunMode.FULL

    def prompt_limit(self, mode: RunMode) -> Optional[int]:
        """Prompt for example limit."""
        if mode == RunMode.FULL:
            return None

        if self.use_fallback:
            limit_str = self._input_fallback("Enter example limit (or press Enter for 100)", "100")
            return int(limit_str) if limit_str.strip() else 100

        limit_str = questionary.text(
            "Enter example limit (default: 100):",
            default="100",
            style=CUSTOM_STYLE,
        ).ask()

        try:
            return int(limit_str) if limit_str.strip() else 100
        except ValueError:
            return 100

    def prompt_trials(self) -> int:
        """Prompt for number of Heretic trials."""
        if self.use_fallback:
            trials = self._input_fallback("Enter number of Heretic trials (default: 100)", "100")
            return int(trials) if trials.strip() else 100

        trials = questionary.text(
            "Enter number of Heretic trials (default: 100):",
            default="100",
            style=CUSTOM_STYLE,
        ).ask()

        try:
            return int(trials) if trials.strip() else 100
        except ValueError:
            return 100

    def prompt_dtype(self) -> str:
        """Prompt for model dtype."""
        if self.use_fallback:
            print("\nDtype options:")
            print("  1 - auto (Automatically select)")
            print("  2 - float16")
            print("  3 - bfloat16")
            print("  4 - float32")

            dtype_map = {"1": "auto", "2": "float16", "3": "bfloat16", "4": "float32"}
            choice = self._input_fallback("Select dtype (1/2/3/4)", "1")
            return dtype_map.get(choice, "auto")

        choices = [
            Choice(title="1. Auto (recommended)", value="auto"),
            Choice(title="2. Float16", value="float16"),
            Choice(title="3. BFloat16", value="bfloat16"),
            Choice(title="4. Float32", value="float32"),
        ]

        selected = questionary.select(
            "Select model dtype:",
            choices=choices,
            style=CUSTOM_STYLE,
        ).ask()

        return selected

    def prompt_device(self) -> str:
        """Prompt for device."""
        if self.use_fallback:
            print("\nDevice options:")
            print("  1 - cuda (NVIDIA GPU)")
            print("  2 - cpu (CPU only)")

            device_map = {"1": "cuda", "2": "cpu"}
            choice = self._input_fallback("Select device (1/2)", "1")
            return device_map.get(choice, "cuda")

        choices = [
            Choice(title="1. CUDA (NVIDIA GPU)", value="cuda"),
            Choice(title="2. CPU only", value="cpu"),
        ]

        # Add MPS for Apple Silicon
        try:
            import torch

            if torch.backends.mps.is_available():
                choices.append(Choice(title="3. MPS (Apple Silicon)", value="mps"))
        except Exception:
            pass

        device = questionary.select(
            "Select device:",
            choices=choices,
            style=CUSTOM_STYLE,
        ).ask()

        return device

    def prompt_seed(self) -> int:
        """Prompt for random seed."""
        if self.use_fallback:
            seed = self._input_fallback("Enter random seed (default: 42)", "42")
            return int(seed) if seed.strip() else 42

        seed = questionary.text(
            "Enter random seed (default: 42):",
            default="42",
            style=CUSTOM_STYLE,
        ).ask()

        try:
            return int(seed) if seed.strip() else 42
        except ValueError:
            return 42

    def prompt_quantization(self) -> str:
        """Prompt for quantization method."""
        if self.use_fallback:
            print("\nQuantization options:")
            print("  1 - auto (4-bit when the model cannot fit)")
            print("  2 - none (No quantization)")
            print("  3 - bnb_4bit (4-bit quantization)")
            print("  4 - bnb_8bit (8-bit quantization, benchmarks only)")

            quant_map = {"1": "auto", "2": "none", "3": "bnb_4bit", "4": "bnb_8bit"}
            choice = self._input_fallback("Select quantization (1/2/3/4)", "1")
            return quant_map.get(choice, "auto")

        choices = [
            Choice(title="1. Auto (recommended)", value="auto"),
            Choice(title="2. None (full precision)", value="none"),
            Choice(title="3. 4-bit (requires bitsandbytes)", value="bnb_4bit"),
            Choice(title="4. 8-bit (benchmarks only)", value="bnb_8bit"),
        ]

        selected = questionary.select(
            "Select quantization:",
            choices=choices,
            style=CUSTOM_STYLE,
        ).ask()

        return selected

    def prompt_confirm(self, message: str, default: bool = True) -> bool:
        """Prompt for confirmation."""
        if self.use_fallback:
            response = self._input_fallback(f"{message} (y/n)", "y" if default else "n")
            return response.lower() in ("y", "yes")

        return questionary.confirm(
            message,
            default=default,
            style=CUSTOM_STYLE,
        ).ask()

    def _input_fallback(self, prompt: str, default: str = "") -> str:
        """Basic input fallback when questionary is not available."""
        if default:
            response = input(f"{prompt} [{default}]: ").strip()
            return response if response else default
        return input(f"{prompt}: ").strip()

    def run_full_prompts(self) -> Settings:
        """
        Run all prompts to collect settings.

        Returns:
            Configured Settings object
        """
        print("\n" + "=" * 60)
        print("   Anlord Abliterator Configuration")
        print("=" * 60 + "\n")

        # Model selection
        print("Step 1/8: Model Selection")
        print("-" * 40)
        model_id = self.prompt_model_id()

        # Output directory
        print("\nStep 2/8: Output Directory")
        print("-" * 40)
        output_dir = self.prompt_output_dir()

        # Benchmarks
        print("\nStep 3/8: Benchmark Selection")
        print("-" * 40)
        benchmarks = self.prompt_benchmarks()
        skip_benchmarks = not benchmarks

        # Mode
        print("\nStep 4/8: Evaluation Mode")
        print("-" * 40)
        mode = self.prompt_mode()

        # Limit
        print("\nStep 5/8: Example Limit")
        print("-" * 40)
        limit = self.prompt_limit(mode)

        # Trials
        print("\nStep 6/8: Heretic Configuration")
        print("-" * 40)
        trials = self.prompt_trials()

        # Dtype
        print("\nStep 7/8: Model Configuration")
        print("-" * 40)
        dtype = self.prompt_dtype()
        device = self.prompt_device()
        quantization = self.prompt_quantization()

        # Seed
        print("\nStep 8/8: Reproducibility")
        print("-" * 40)
        seed = self.prompt_seed()

        # Create settings
        settings = Settings(
            model=model_id,
            output_dir=output_dir,
            benchmarks=benchmarks,
            mode=mode,
            limit=limit,
            heretic_trials=trials,
            dtype=dtype,
            device=device,
            quantization=quantization,
            seed=seed,
            skip_benchmarks=skip_benchmarks,
        )

        # Summary
        print("\n" + "=" * 60)
        print("   Configuration Summary")
        print("=" * 60)
        print(f"  Model:         {model_id}")
        print(f"  Output:        {output_dir}")
        print(f"  Benchmarks:    {', '.join(benchmarks)}")
        print(f"  Mode:          {mode.value}")
        print(f"  Limit:         {limit if limit else 'Full'}")
        print(f"  Trials:        {trials}")
        print(f"  Dtype:         {dtype}")
        print(f"  Device:        {device}")
        print(f"  Quantization:  {quantization}")
        print(f"  Seed:          {seed}")
        print("=" * 60)

        return settings
