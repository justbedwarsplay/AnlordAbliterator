# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ephemeral Hugging Face token handling."""

from __future__ import annotations

import io
import os

from anlord.utils.auth import HF_TOKEN_ENV, prompt_for_hf_token


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _NonTTY(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_token_is_kept_only_in_process_environment(monkeypatch):
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)

    available = prompt_for_hf_token(
        input_stream=_TTY(),
        password_reader=lambda _prompt: "  hf_session_only  ",
    )

    assert available is True
    assert os.environ[HF_TOKEN_ENV] == "hf_session_only"


def test_token_prompt_can_be_skipped(monkeypatch):
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)

    available = prompt_for_hf_token(
        input_stream=_TTY(),
        password_reader=lambda _prompt: "",
    )

    assert available is False
    assert HF_TOKEN_ENV not in os.environ


def test_non_interactive_process_never_blocks_for_token(monkeypatch):
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)
    called = False

    def password_reader(_prompt):
        nonlocal called
        called = True
        return "hf_should_not_be_read"

    assert prompt_for_hf_token(input_stream=_NonTTY(), password_reader=password_reader) is False
    assert called is False
