# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ephemeral Hugging Face authentication for CLI runs."""

from __future__ import annotations

import getpass
import os
import sys
from typing import Callable, TextIO

HF_TOKEN_ENV = "HF_TOKEN"


def prompt_for_hf_token(
    *,
    enabled: bool = True,
    input_stream: TextIO | None = None,
    password_reader: Callable[[str], str] | None = None,
) -> bool:
    """Prompt for an optional token and keep it only in process memory.

    The token is exposed to child processes through ``HF_TOKEN`` but is never
    passed to ``huggingface_hub.login``, written to settings, logged, or saved to
    disk. Pressing Enter continues anonymously.

    Returns ``True`` when a token is available for this process.
    """
    if os.environ.get(HF_TOKEN_ENV, "").strip():
        return True
    if not enabled:
        return False

    stream = input_stream or sys.stdin
    if stream is None or not stream.isatty():
        return False

    print("\nHugging Face authentication (optional)")
    print("Create a read token at: https://huggingface.co/settings/tokens")
    print("The token is used only for this run and is not saved by Anlord Abliterator.")

    reader = password_reader or getpass.getpass
    try:
        token = reader("HF token (press Enter to skip): ").strip()
    except (EOFError, OSError):
        return False

    if not token:
        print("Continuing without a Hugging Face token.\n")
        return False

    os.environ[HF_TOKEN_ENV] = token
    print("Hugging Face token enabled for this run.\n")
    return True
