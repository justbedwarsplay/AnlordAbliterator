# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lightweight stubs so unit tests can run without the full ML stack."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    try:
        import torch  # noqa: F401

        return
    except ImportError:
        pass

    torch = ModuleType("torch")
    torch.__version__ = "0.0-test"
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        device_count=lambda: 0,
        memory_reserved=lambda: 0,
        empty_cache=lambda: None,
        synchronize=lambda: None,
        get_device_properties=lambda index: SimpleNamespace(name="cpu", total_memory=0),
    )
    torch.version = SimpleNamespace(cuda=None)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch


def _install_psutil_stub() -> None:
    if "psutil" in sys.modules:
        return
    try:
        import psutil  # noqa: F401

        return
    except ImportError:
        pass

    psutil = ModuleType("psutil")
    psutil.virtual_memory = lambda: SimpleNamespace(total=16 * 1024**3, used=8 * 1024**3, available=8 * 1024**3)
    psutil.swap_memory = lambda: SimpleNamespace(total=0)
    psutil.cpu_count = lambda logical=True: 8
    psutil.cpu_percent = lambda interval=None: 1.0
    sys.modules["psutil"] = psutil


_install_torch_stub()
_install_psutil_stub()
