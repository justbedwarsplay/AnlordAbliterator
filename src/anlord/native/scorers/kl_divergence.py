# SPDX-License-Identifier: AGPL-3.0-or-later
"""
KL-divergence scorer: measures how far the model's behavior has drifted from
the baseline model. Lower is better (less capability damage).

By default the KL divergence is computed on the first generated token (classic
first-token KL). With `token_count > 1` it is averaged over the first
`token_count` generated token positions, which is less sensitive to
single-token noise and evaluates a few tokens of actual generation.
"""

from __future__ import annotations

import torch.nn.functional as F
from pydantic import BaseModel, Field

from ..config import DatasetSpecification
from ..plugin import Context
from ..prompts import Prompt
from ..scorer import Score, Scorer
from ..utils import print


class Settings(BaseModel):
    prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Prompt dataset used to measure KL divergence from original model.",
    )

    token_count: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of generated token positions the KL divergence is "
        "averaged over. 1 = classic first-token KL; 3-5 averages a few tokens "
        "of actual generation and is less sensitive to single-token noise.",
    )


class KLDivergence(Scorer):
    """
    KL divergence between current model and baseline.

    Measures how much the model's behavior has drifted from baseline.
    Lower is better (less damage).
    """

    settings: Settings

    # Class-level default so the scorer is detectable before its first
    # get_score() call (used by the evaluator's quick-evaluation methods).
    last_kl_value: float = 0.0

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "KL divergence"

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading KL divergence evaluation prompts from "
            f"[bold]{self.settings.prompts.dataset}[/]..."
        )
        self.prompts: list[Prompt] = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        print(
            f"* Obtaining baseline probability distributions "
            f"(first {self.settings.token_count} token(s))..."
        )
        baseline_logits = ctx.get_logits_multi(self.prompts, self.settings.token_count)
        self._baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)

    def get_score(self, ctx: Context) -> Score:
        logprobs = F.log_softmax(
            ctx.get_logits_multi(self.prompts, self.settings.token_count), dim=-1
        )
        positions = logprobs.shape[0]
        kls = [
            F.kl_div(
                logprobs[t],
                self._baseline_logprobs[t],
                reduction="batchmean",
                log_target=True,
            ).item()
            for t in range(positions)
        ]
        kl_divergence = sum(kls) / positions

        # Legacy metric extraction: the pipeline reports the raw KL value
        # alongside the scores, so remember the most recent value.
        self.last_kl_value = kl_divergence

        return Score(
            value=kl_divergence,
            rich_display=f"[bold]{kl_divergence:.4f}[/]",
            md_display=f"{kl_divergence:.4f}",
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        return Score(
            value=0,
            rich_display="[bold]0[/] [italic](by definition)[/]",
            md_display="0 *(by definition)*",
        )
