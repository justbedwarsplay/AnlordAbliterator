# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Benchmark task definitions for Anlord Abliterator.
Defines available tasks and their configurations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskConfig:
    """Configuration for a benchmark task."""

    task_id: str  # lm-eval task ID
    name: str  # Display name
    description: str  # Task description
    num_fewshot: int = 0  # Default few-shot count
    primary_metric: str = "acc"  # Primary metric to use
    group: Optional[str] = None  # Task group (e.g., "academic")

    # Task-specific configuration
    gen_kwargs: dict = None  # Generation arguments
    filter_list: list = None  # Post-processing filters

    def __post_init__(self):
        if self.gen_kwargs is None:
            self.gen_kwargs = {}
        if self.filter_list is None:
            self.filter_list = []


# Standard academic benchmarks
AVAILABLE_TASKS: dict[str, TaskConfig] = {
    # MMLU variants
    "mmlu": TaskConfig(
        task_id="mmlu",
        name="MMLU",
        description="Massively Multilingual Language Understanding",
        num_fewshot=5,
        primary_metric="acc",
        group="academic",
    ),
    "mmlu_abstract_algebra": TaskConfig(
        task_id="mmlu_abstract_algebra",
        name="MMLU Abstract Algebra",
        description="Abstract algebra subset of MMLU",
        num_fewshot=5,
        primary_metric="acc",
    ),
    "mmlu_anatomy": TaskConfig(
        task_id="mmlu_anatomy",
        name="MMLU Anatomy",
        description="Anatomy subset of MMLU",
        num_fewshot=5,
        primary_metric="acc",
    ),
    "mmlu_astronomy": TaskConfig(
        task_id="mmlu_astronomy",
        name="MMLU Astronomy",
        description="Astronomy subset of MMLU",
        num_fewshot=5,
        primary_metric="acc",
    ),
    # Math benchmarks
    "gsm8k": TaskConfig(
        task_id="gsm8k",
        name="GSM8K",
        description="Grade School Math 8K",
        num_fewshot=5,
        primary_metric="exact_match,flexible-extract",
        group="math",
    ),
    "gsm8k_cot": TaskConfig(
        task_id="gsm8k",
        name="GSM8K (CoT)",
        description="Grade School Math 8K with Chain-of-Thought",
        num_fewshot=5,
        primary_metric="acc",
        group="math",
        gen_kwargs={"do_sample": False, "temperature": 0.0},
    ),
    "math": TaskConfig(
        task_id="math",
        name="MATH",
        description="Mathematical problem solving",
        num_fewshot=5,
        primary_metric="acc",
        group="math",
    ),
    # Commonsense reasoning
    "hellaswag": TaskConfig(
        task_id="hellaswag",
        name="HellaSwag",
        description="Commonsense inference challenge",
        num_fewshot=10,
        primary_metric="acc_norm",
        group="reasoning",
    ),
    "winogrande": TaskConfig(
        task_id="winogrande",
        name="Winogrande",
        description="WinoGrande commonsense reasoning",
        num_fewshot=0,
        primary_metric="acc",
        group="reasoning",
    ),
    "arc_easy": TaskConfig(
        task_id="arc_easy",
        name="ARC-Easy",
        description="AI2 Reasoning Challenge - Easy",
        num_fewshot=0,
        primary_metric="acc",
        group="reasoning",
    ),
    "arc_challenge": TaskConfig(
        task_id="arc_challenge",
        name="ARC-Challenge",
        description="AI2 Reasoning Challenge",
        num_fewshot=0,
        primary_metric="acc_norm",
        group="reasoning",
    ),
    # Truthfulness
    "truthfulqa": TaskConfig(
        task_id="truthfulqa_mc2",
        name="TruthfulQA",
        description="Truthful question answering (multiple-answer MC2)",
        num_fewshot=0,
        primary_metric="acc",
        group="truthfulness",
    ),
    "truthfulqa_gen": TaskConfig(
        task_id="truthfulqa_gen",
        name="TruthfulQA (Generation)",
        description="TruthfulQA with generation metrics",
        num_fewshot=0,
        primary_metric="truthful_acc",
        group="truthfulness",
    ),
    # Reading comprehension
    "squad_v2": TaskConfig(
        task_id="squad_v2",
        name="SQuAD v2",
        description="Stanford Question Answering Dataset v2",
        num_fewshot=0,
        primary_metric="f1",
        group="reading",
    ),
    "race": TaskConfig(
        task_id="race",
        name="RACE",
        description="Reading comprehension from exams",
        num_fewshot=0,
        primary_metric="acc",
        group="reading",
    ),
    # Code
    "humaneval": TaskConfig(
        task_id="humaneval",
        name="HumanEval",
        description="Code generation evaluation",
        num_fewshot=0,
        primary_metric="pass@1",
        group="code",
    ),
    "mbpp": TaskConfig(
        task_id="mbpp",
        name="MBPP",
        description="Mostly Basic Python Problems",
        num_fewshot=0,
        primary_metric="pass@1",
        group="code",
    ),
    # Instruction following
    "ifeval": TaskConfig(
        task_id="ifeval",
        name="IFEval",
        description="Instruction Following Evaluation",
        num_fewshot=0,
        primary_metric="acc",
        group="instruction",
    ),
    # GPQA (graduate-level science QA)
    "gpqa": TaskConfig(
        task_id="gpqa_diamond",
        name="GPQA Diamond",
        description="Graduate-level Google-proof QA (diamond, hardest)",
        num_fewshot=0,
        primary_metric="acc",
        group="academic",
    ),
    "gpqa_diamond": TaskConfig(
        task_id="gpqa_diamond",
        name="GPQA Diamond",
        description="Graduate-level Google-proof QA (diamond, hardest)",
        num_fewshot=0,
        primary_metric="acc",
        group="academic",
    ),
    "gpqa_main": TaskConfig(
        task_id="gpqa_main",
        name="GPQA Main",
        description="GPQA main split",
        num_fewshot=0,
        primary_metric="acc",
        group="academic",
    ),
    "gpqa_extended": TaskConfig(
        task_id="gpqa_extended",
        name="GPQA Extended",
        description="GPQA extended",
        num_fewshot=0,
        primary_metric="acc",
        group="academic",
    ),
    "mmlu_pro": TaskConfig(
        task_id="mmlu_pro",
        name="MMLU-Pro",
        description="MMLU-Pro 10-option challenging version",
        num_fewshot=5,
        primary_metric="acc",
        group="academic",
    ),
    "ifstruct": TaskConfig(
        task_id="ifstruct",
        name="Ifstruct V1",
        description="LiquidAI Ifstruct instruction structure following",
        num_fewshot=0,
        primary_metric="acc",
        group="instruction",
    ),
    "ifstruct_v1": TaskConfig(
        task_id="ifstruct",
        name="Ifstruct V1",
        description="LiquidAI Ifstruct instruction structure following",
        num_fewshot=0,
        primary_metric="acc",
        group="instruction",
    ),
    "parsebench": TaskConfig(
        task_id="parsebench",
        name="ParseBench",
        description="LlamaIndex ParseBench document parsing",
        num_fewshot=0,
        primary_metric="f1",
        group="parsing",
    ),
    "extractbench": TaskConfig(
        task_id="extractbench",
        name="ExtractBench",
        description="LlamaIndex ExtractBench structured extraction",
        num_fewshot=0,
        primary_metric="f1",
        group="parsing",
    ),
    "screenspot_pro": TaskConfig(
        task_id="screenspot_pro",
        name="ScreenSpot-Pro",
        description="ScreenSpot-Pro GUI grounding (overall)",
        num_fewshot=0,
        primary_metric="acc",
        group="vision",
    ),
    "mmmu_pro": TaskConfig(
        task_id="mmmu_pro",
        name="MMMU-Pro Vision",
        description="MMMU-Pro multimodal reasoning (vision)",
        num_fewshot=0,
        primary_metric="acc",
        group="vision",
    ),

}


# Task groups for batch running
TASK_GROUPS: dict[str, list[str]] = {
    "standard": ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "winogrande", "truthfulqa"],
    "academic": ["mmlu", "gsm8k", "arc_challenge", "winogrande", "mmlu_pro", "gpqa", "gpqa_diamond"],
    "reasoning": ["hellaswag", "arc_challenge", "winogrande", "arc_easy"],
    "math": ["gsm8k", "math"],
    "truthfulness": ["truthfulqa"],
    "code": ["humaneval", "mbpp"],
    "parsing": ["parsebench", "extractbench"],
    "vision": ["screenspot_pro", "mmmu_pro"],
    "instruction": ["ifeval", "ifstruct", "ifstruct_v1"],
    "gpqa": ["gpqa", "gpqa_diamond", "gpqa_main", "gpqa_extended"],
    "extended": ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "winogrande", "truthfulqa", "gpqa", "gpqa_diamond", "mmlu_pro", "ifstruct", "parsebench", "extractbench", "screenspot_pro", "mmmu_pro"],
    "full": list(AVAILABLE_TASKS.keys()),
}


def list_available_tasks() -> list[str]:
    """Get list of all available task IDs."""
    return list(AVAILABLE_TASKS.keys())


def get_task_config(task_id: str) -> TaskConfig | None:
    """Get configuration for a specific task."""
    return AVAILABLE_TASKS.get(task_id)


def expand_task_list(task_names: list[str]) -> list[str]:
    """
    Expand a list of task names, handling groups and aliases.

    Args:
        task_names: List of task names or group names

    Returns:
        Expanded list of task IDs
    """
    expanded = []

    for name in task_names:
        name_lower = name.lower().strip()

        # Check if it's a group
        if name_lower in TASK_GROUPS:
            expanded.extend(TASK_GROUPS[name_lower])
        # Check if it's "all"
        elif name_lower in ["all", "*"]:
            expanded.extend(list(AVAILABLE_TASKS.keys()))
        # Check if it's an individual task
        elif name_lower in AVAILABLE_TASKS:
            expanded.append(name_lower)
        # Check if the exact ID exists
        elif name in AVAILABLE_TASKS:
            expanded.append(name)
        else:
            # Treat as a custom task ID
            expanded.append(name)

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for task in expanded:
        if task not in seen:
            seen.add(task)
            result.append(task)

    return result
