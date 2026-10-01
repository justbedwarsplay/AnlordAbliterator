# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Architecture profiles — per-architecture default overrides for the
abliteration search.

The base search defaults (search bounds, ceilings) were tuned on Qwen-family
models. Some architecture families have a measurably different ablation
structure and get their own defaults; every unregistered architecture keeps
the base defaults.

Resolution order for a profiled setting:
    explicit CLI / config value  >  architecture profile  >  base default

A profiled field is "on auto" when its Settings value is None. Detection uses
the Hugging Face config ``model_type`` (e.g. "qwen3", "gemma3_text"); when it
cannot be determined, base defaults apply — detection must never block a run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchitectureProfile:
    """Default overrides for one architecture family.

    `model_type_prefixes` are matched against the HF config `model_type`
    (prefix match, so "gemma" covers "gemma3_text" / "gemma3" / ...).
    """

    label: str
    model_type_prefixes: tuple[str, ...]
    overrides: dict[str, Any]


# Base values of the profiled Settings fields (the Qwen-tuned defaults).
BASE_DEFAULTS = {
    "abliteration_max_weight_limit": 1.5,
}

ARCHITECTURE_PROFILES: tuple[ArchitectureProfile, ...] = (
    # Qwen families: the base defaults were derived from Qwen experiments.
    # Registered as an explicit no-op so profile resolution is visible in logs.
    ArchitectureProfile(
        label="qwen",
        model_type_prefixes=("qwen",),
        overrides={},
    ),
    # Gemma families: the refusal mechanism is carried by the MLP down_proj,
    # and the optimal MLP ablation weight lies well above the Qwen-tuned
    # ceiling of 1.5. Measured on google/gemma-3-270m-it: a 500-trial search
    # with the 1.5 ceiling stalled at 36/100 refusals (the winning trial sat
    # at mlp max_weight 1.494), while the same parameters with the MLP weight
    # scaled to 2.0 / 2.5 reached 14 / 10 refusals. Attention-only ablation is
    # nearly useless on this family (81/100 refusals even at weight 2.5), so
    # the ceiling — not the direction — was the binding constraint.
    # See docs/optimization_ideas.md ("Gemma case study").
    ArchitectureProfile(
        label="gemma",
        model_type_prefixes=("gemma",),
        overrides={"abliteration_max_weight_limit": 2.0},
    ),
)


def detect_model_type(
    model: str, cache_dir=None, model_commit: str | None = None
) -> str | None:
    """
    The Hugging Face config `model_type` of the model, or None when it cannot
    be determined. Any failure (offline, unknown model, local path) returns
    None — detection must never block a run.
    """
    try:
        from transformers import AutoConfig

        kwargs: dict[str, Any] = {}
        if model_commit:
            kwargs["revision"] = model_commit
        config = AutoConfig.from_pretrained(
            model,
            cache_dir=str(cache_dir) if cache_dir else None,
            **kwargs,
        )
        model_type = str(getattr(config, "model_type", "") or "").lower()
        return model_type or None
    except Exception as error:
        logger.debug("Could not detect model_type for %s: %s", model, error)
        return None


def match_profile(model_type: str | None) -> ArchitectureProfile | None:
    """The first profile whose prefix list matches `model_type`, or None."""
    if not model_type:
        return None
    normalized = model_type.lower()
    for profile in ARCHITECTURE_PROFILES:
        if any(
            normalized.startswith(prefix) for prefix in profile.model_type_prefixes
        ):
            return profile
    return None


def apply_architecture_profile(
    settings,
    model_type: str | None,
    log: Callable[[str], None] | None = None,
) -> ArchitectureProfile | None:
    """
    Applies the matching profile's overrides to `settings` for fields the user
    left on auto (None). Explicit values are never touched. Reproduction runs
    are skipped entirely: the bundle carries the original run's values.

    Returns the applied profile, or None when no profile applies.
    """
    if getattr(settings, "reproduce", False):
        return None
    profile = match_profile(model_type)
    if profile is None:
        return None

    def _log(message: str) -> None:
        logger.info("%s", message)
        if log is not None:
            log(message)

    _log(f"Architecture profile '{profile.label}' (model_type {model_type})")
    for field_name, value in profile.overrides.items():
        current = getattr(settings, field_name, None)
        if current is None:
            setattr(settings, field_name, value)
            _log(f"  * {field_name}: auto -> {value}")
        else:
            _log(f"  * {field_name}: {current} (explicit value kept)")
    if not profile.overrides:
        _log("  * base defaults (no overrides for this family)")
    return profile
