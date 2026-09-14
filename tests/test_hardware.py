# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for Anlord Abliterator hardware monitoring module.
"""

import pytest
import time

from anlord.hardware import (
    HardwareMonitor,
    SystemInfo,
    get_system_info,
)


class TestSystemInfo:
    """Test SystemInfo class."""

    def test_get_system_info(self):
        """Test getting system information."""
        info = get_system_info()

        assert isinstance(info, SystemInfo)
        assert isinstance(info.has_cuda, bool)
        assert isinstance(info.pytorch_version, str)
        assert isinstance(info.python_version, str)

    def test_system_info_to_dict(self):
        """Test converting SystemInfo to dictionary."""
        info = get_system_info()
        data = info.to_dict()

        assert "has_cuda" in data
        assert "pytorch_version" in data
        assert "python_version" in data


class TestHardwareMonitor:
    """Test HardwareMonitor class."""

    def test_monitor_init(self):
        """Test monitor initialization."""
        monitor = HardwareMonitor(interval=0.5)

        assert monitor.interval == 0.5
        assert monitor._monitoring is False

    def test_monitor_start_stop(self):
        """Test starting and stopping the monitor."""
        monitor = HardwareMonitor(interval=0.1)

        # Start monitoring
        monitor.start()
        assert monitor._monitoring is True

        # Wait a bit for some samples
        time.sleep(0.5)

        # Stop monitoring
        metrics = monitor.stop()

        assert metrics is not None
        assert metrics.duration_seconds > 0
        assert len(metrics.samples) > 0

    def test_monitor_metrics(self):
        """Test that metrics are collected."""
        monitor = HardwareMonitor(interval=0.1)

        monitor.start()
        time.sleep(0.3)
        metrics = monitor.stop()

        # Check metrics are populated
        assert metrics.peak_vram_gb >= 0
        assert metrics.peak_ram_gb >= 0
        assert metrics.duration_seconds > 0

    def test_monitor_to_dict(self):
        """Test converting metrics to dictionary."""
        monitor = HardwareMonitor(interval=0.1)

        monitor.start()
        time.sleep(0.2)
        metrics = monitor.stop()

        data = metrics.to_dict()

        assert "peak_vram_gb" in data
        assert "peak_ram_gb" in data
        assert "duration_seconds" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
