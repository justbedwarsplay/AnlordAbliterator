# SPDX-License-Identifier: AGPL-3.0-or-later
"""Allow ``python -m anlord`` to invoke the CLI."""

from .cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
