# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Scorer base classes for the native abliteration pipeline.

A scorer evaluates model behavior and returns a Score. Scorers are plugins:
built-in scorers live in `anlord.native.scorers`, and external scorers can be
loaded from a filesystem file or import path via the `scorers` config list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from .config import NativeConfig
from .plugin import Context, Plugin


@dataclass
class Score:
    """
    Result of evaluating a scorer.

    - `value`: scalar value used for optimization (if enabled).
    - `rich_display`: formatted Rich markup shown to the user in logs/console.
    - `md_display`: formatted value in reports and reproduction metadata.
    """

    value: float
    rich_display: str
    md_display: str


class Scorer(Plugin, ABC):
    """
    Abstract base class for scorer plugins.

    Scorers evaluate model behavior and return a Score.

    Example: counting refusals, measuring KL divergence, etc.
    """

    @property
    def score_name(self) -> str:
        """
        The name of the `Score` object returned by `get_score()`.
        This is what shows up in the CLI and in reproduction metadata.
        """
        return self.__class__.__name__

    def __init__(
        self,
        anlord_settings: NativeConfig,
        settings: BaseModel | None = None,
    ):
        super().__init__(anlord_settings=anlord_settings, settings=settings)

    @abstractmethod
    def get_score(self, ctx: Context) -> Score:
        """
        Return a `Score` given the evaluation context.
        The `value` of the `Score` must be of the order of magnitude 1
        to ensure that all scores are comparable during co-optimization.
        """

    def get_baseline_score(self, ctx: Context) -> Score:
        """
        Calculates a baseline score.

        Defaults to the current `get_score(...)` implementation and can be
        overridden by scorers that need a distinct baseline.
        """
        return self.get_score(ctx)
