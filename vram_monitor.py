"""VRAM monitoring via NVML for GPU memory management.

Provides threshold-based monitoring for proactive VRAM management
when running concurrent GPU workloads (Whisper + Ollama).
"""

import pynvml


class VRAMMonitor:
    """Monitor GPU VRAM usage and trigger actions at thresholds.

    Thresholds are tuned for RTX 3070 (8192 MB):
      - WARNING:   75%   (6144 MB) -- reduce Ollama context
      - CRITICAL:  87.5% (7168 MB) -- unload Ollama
      - EMERGENCY: 95%   (7782 MB) -- fall back to CPU Whisper
    """

    WARNING_MB = 6144      # 75% of 8192
    CRITICAL_MB = 7168     # 87.5% of 8192
    EMERGENCY_MB = 7782    # 95% of 8192

    def __init__(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self.total_mb = info.total // (1024 * 1024)

    @classmethod
    def create(cls) -> "VRAMMonitor | None":
        """Factory that returns None on init failure (no GPU, driver issues, etc.)."""
        try:
            return cls()
        except Exception:
            return None

    def get_usage_mb(self) -> int:
        """Return used VRAM in MB."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.used // (1024 * 1024)

    def get_free_mb(self) -> int:
        """Return free VRAM in MB."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.free // (1024 * 1024)

    def check(self) -> str:
        """Return current VRAM pressure level.

        Returns one of: 'ok', 'warning', 'critical', 'emergency'.
        """
        used = self.get_usage_mb()
        if used >= self.EMERGENCY_MB:
            return "emergency"
        elif used >= self.CRITICAL_MB:
            return "critical"
        elif used >= self.WARNING_MB:
            return "warning"
        return "ok"

    def get_stats(self) -> dict:
        """Return structured VRAM statistics.

        Returns dict with keys:
          used_mb, free_mb, total_mb, level, utilization_pct
        """
        used = self.get_usage_mb()
        free = self.get_free_mb()
        return {
            "used_mb": used,
            "free_mb": free,
            "total_mb": self.total_mb,
            "level": self.check(),
            "utilization_pct": round(used / self.total_mb * 100, 1) if self.total_mb > 0 else 0.0,
        }

    def shutdown(self):
        """Clean up NVML resources."""
        pynvml.nvmlShutdown()
