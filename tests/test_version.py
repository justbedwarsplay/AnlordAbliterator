# SPDX-License-Identifier: AGPL-3.0-or-later
"""The reported version must match the release / installed distribution."""

from pathlib import Path


def _pyproject_version() -> str:
    text = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    return text.split('version = "')[1].split('"')[0]


def test_source_run_reports_release_version():
    """Running from a source checkout reports the release constant, not a
    stale hardcoded string."""
    from anlord import __version__

    assert __version__ == _pyproject_version()


def test_version_is_not_the_old_hardcoded_value():
    from anlord import __version__

    assert __version__ != "1.0.0"
