# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Main CLI entry point for Anlord Abliterator.
Provides command-line interface with interactive mode.
"""

import argparse
import logging
import sys
from pathlib import Path

from ..config import RunMode, Settings
from ..hardware import get_system_info, print_system_info
from ..pipeline import AbliterationPipeline
from ..utils.auth import prompt_for_hf_token
from ..utils.runtime import configure_huggingface_environment
from .prompts import InteractivePrompter

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True, log_file: Path | None = None):
    """Configure useful application logs without third-party HTTP debug noise."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    package_logger = __name__.split(".cli.", 1)[0]
    logging.getLogger(package_logger).setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("anlord").setLevel(logging.DEBUG if verbose else logging.INFO)
    for noisy_logger in ("httpcore", "httpx", "urllib3", "filelock"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Anlord Abliterator - LLM abliteration and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default on Windows)
  python -m anlord.cli.main
  
  # With command line arguments
  python -m anlord.cli.main --model unsloth/gpt-oss-20b-BF16 --output ./output
  
  # Quick evaluation
  python -m anlord.cli.main --model meta-llama/Llama-3.2-1B --mode quick --limit 50
  
  # Full evaluation with specific benchmarks
  python -m anlord.cli.main --model Qwen/Qwen2.5-7B --tasks mmlu,gsm8k --mode full
        """,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", "-m", type=str, default=None, help="HuggingFace model ID or local path"
    )
    model_group.add_argument(
        "--model-commit", type=str, default=None, help="Specific model commit hash"
    )
    model_group.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    model_group.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    model_group.add_argument(
        "--prefetch-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the model snapshot with visible progress before loading",
    )

    # Hugging Face authentication
    auth_group = parser.add_argument_group("Hugging Face Authentication")
    auth_group.add_argument(
        "--hf-token-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prompt for an optional in-memory HF token at startup",
    )

    # Heretic configuration
    heretic_group = parser.add_argument_group("Heretic Configuration")
    heretic_group.add_argument(
        "--trials", "-t", type=int, default=None, help="Number of Heretic optimization trials"
    )
    heretic_group.add_argument(
        "--timeout",
        type=int,
        default=43200,
        help="Timeout for Heretic in seconds (default: 12 hours)",
    )
    heretic_group.add_argument(
        "--eval-prompts", type=int, default=100, help="Number of evaluation prompts"
    )

    # Benchmark configuration
    benchmark_group = parser.add_argument_group("Benchmark Configuration")
    benchmark_group.add_argument(
        "--tasks",
        "--benchmarks",
        "-b",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks (or 'all')",
    )
    benchmark_group.add_argument(
        "--mode", type=str, choices=["quick", "full"], default=None, help="Evaluation mode"
    )
    benchmark_group.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit examples per benchmark (quick mode only)",
    )
    benchmark_group.add_argument(
        "--num-fewshot", type=int, default=0, help="Number of few-shot examples"
    )
    benchmark_group.add_argument(
        "--native-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use native original benchmarks instead of lm-eval (default: native)",
    )

    # Hardware configuration
    hardware_group = parser.add_argument_group("Hardware Configuration")
    hardware_group.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype",
    )
    hardware_group.add_argument(
        "--device", type=str, choices=["cuda", "cpu", "mps"], default="cuda", help="Device to use"
    )
    hardware_group.add_argument(
        "--quantization",
        type=str,
        choices=["auto", "none", "bnb_4bit", "bnb_8bit"],
        default="auto",
        help="Quantization method (auto enables 4-bit when the model cannot fit)",
    )
    hardware_group.add_argument("--batch-size", type=int, default=1, help="Batch size")

    # Pipeline options
    pipeline_group = parser.add_argument_group("Pipeline Options")
    pipeline_group.add_argument(
        "--skip-baseline", action="store_true", help="Skip baseline evaluation"
    )
    pipeline_group.add_argument(
        "--skip-abliteration", action="store_true", help="Skip abliteration"
    )
    pipeline_group.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmarks")
    pipeline_group.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume valid completed work (use --no-resume for a fresh run)",
    )
    pipeline_group.add_argument(
        "--baseline-evaluate",
        action="store_true",
        default=False,
        help="Force separate Heretic baseline evaluation (default: reuse abliteration initial metrics)",
    )
    pipeline_group.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Auto-confirm long ETA prompts (>2h) without interactive question",
    )

    # Reproducibility
    repro_group = parser.add_argument_group("Reproducibility")
    repro_group.add_argument("--seed", "-s", type=int, default=42, help="Random seed")

    # System info
    system_group = parser.add_argument_group("System Information")
    system_group.add_argument(
        "--info", action="store_true", help="Show system information and exit"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--verbose", "-v", action="store_true", help="Include Anlord Abliterator debug details"
    )
    output_group.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (less output)")
    output_group.add_argument("--log-file", type=str, default=None, help="Log file path")

    return parser


def run_from_args(args: argparse.Namespace) -> Settings:
    """
    Run pipeline from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Settings object
    """
    # Parse benchmarks (supports groups: standard, extended, full, academic, etc.)
    benchmarks = None
    if args.tasks:
        from ..benchmarks.tasks import expand_task_list
        raw = args.tasks.strip()
        # Single group alias like "all", "standard", "extended", "full"
        if raw.lower() in ["*", "full", "extended", "standard", "academic", "reasoning", "parsing", "vision", "gpqa"]:
            benchmarks = expand_task_list([raw])
        elif raw.lower() == "all":
            # "all" = extended set (6 standard + 8 new high-value benches = 14) for user convenience, not 29 full
            benchmarks = expand_task_list(["extended"])
        else:
            # Comma-separated list may contain groups and task ids
            parts = [b.strip() for b in raw.split(",") if b.strip()]
            benchmarks = expand_task_list(parts)

    # Parse mode
    mode = None
    if args.mode:
        mode = RunMode.QUICK if args.mode == "quick" else RunMode.FULL

    selected_mode = mode or RunMode.QUICK
    if selected_mode == RunMode.FULL:
        limit = None
    else:
        limit = args.limit if args.limit is not None else 100

    # Build settings
    settings = Settings(
        model=args.model or "unsloth/gpt-oss-20b-BF16",
        model_commit=args.model_commit,
        output_dir=Path(args.output) if args.output else Path("./output"),
        cache_dir=Path(args.cache_dir) if args.cache_dir else Path("./cache"),
        prefetch_model=args.prefetch_model,
        heretic_trials=args.trials if args.trials is not None else 100,
        heretic_timeout=args.timeout,
        heretic_evaluation_prompts=args.eval_prompts,
        benchmarks=benchmarks,
        mode=selected_mode,
        limit=limit,
        num_fewshot=args.num_fewshot,
        native_benchmarks=getattr(args, "native_benchmarks", True),
        dtype=args.dtype,
        device=args.device,
        quantization=args.quantization,
        batch_size=args.batch_size,
        skip_baseline=args.skip_baseline,
        skip_abliteration=args.skip_abliteration,
        skip_benchmarks=args.skip_benchmarks,
        resume=args.resume,
        baseline_evaluate=getattr(args, "baseline_evaluate", False),
        yes=getattr(args, "yes", False),
        seed=args.seed,
        verbose=args.verbose and not args.quiet,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    return settings


def run_interactive() -> Settings:
    """
    Run interactive configuration prompt.

    Returns:
        Settings object
    """
    prompter = InteractivePrompter()
    return prompter.run_full_prompts()


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle system info request
    if args.info:
        info = get_system_info()
        print_system_info(info)
        return 0

    try:
        prompt_for_hf_token(enabled=args.hf_token_prompt)
    except KeyboardInterrupt:
        print("\nHugging Face token prompt cancelled.")
        return 130

    # Check if we should run interactively
    # Interactive mode if no model specified and stdin is a TTY
    run_interactive_mode = args.model is None and sys.stdin.isatty()

    if run_interactive_mode:
        settings = run_interactive()
    else:
        settings = run_from_args(args)

    configure_huggingface_environment(settings.cache_dir)

    # Setup logging
    setup_logging(
        verbose=settings.verbose,
        log_file=settings.log_file,
    )

    logger.info("Starting Anlord Abliterator pipeline")
    logger.info(f"Model: {settings.model}")
    logger.info(f"Output: {settings.output_dir}")

    # Print system info
    info = get_system_info()
    print_system_info(info)

    # Create and run pipeline
    try:
        pipeline = AbliterationPipeline(settings)
        results = pipeline.run()

        if results.success:
            logger.info("Pipeline completed successfully!")
            return 0
        logger.error("Pipeline failed: %s", results.error or "unknown error")
        return 1

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
