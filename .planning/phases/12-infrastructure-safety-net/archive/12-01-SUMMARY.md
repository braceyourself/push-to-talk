---
phase: 12-infrastructure-safety-net
plan: 01
subsystem: infra
tags: [nvidia, nvml, pynvml, vram, gpu, pipewire, echo-cancellation, webrtc-aec, whisper, ollama]

# Dependency graph
requires: []
provides:
  - VRAMMonitor module with NVML-based GPU memory watchdog
  - VRAM budget validation (go/no-go gate passed)
  - PipeWire echo cancellation configured with verified device name
affects: [12-02, 12-03, 13-decision-engine]

# Tech tracking
tech-stack:
  added: [nvidia-ml-py]
  patterns: [factory-method-with-graceful-degradation, threshold-based-monitoring]

key-files:
  created: [vram_monitor.py]
  modified: [test_live_session.py]

key-decisions:
  - "VRAM GO: Whisper+Ollama concurrent peak 5421MB (66%) leaves 2771MB headroom on RTX 3070"
  - "AEC device name is 'Echo Cancellation Source' (with spaces) for pasimple device_name param"
  - "VRAMMonitor uses factory method create() returning None on failure for graceful GPU-less operation"

patterns-established:
  - "Factory method pattern: create() classmethod returns None instead of raising on init failure"
  - "Threshold-based monitoring: WARNING(6144MB) / CRITICAL(7168MB) / EMERGENCY(7782MB)"

# Metrics
duration: 11min
completed: 2026-02-21
---

# Phase 12 Plan 01: Infrastructure Validation Summary

**VRAM go/no-go gate passed (5421MB peak, 66% of 8GB), VRAMMonitor module with 7 tests, PipeWire WebRTC echo cancellation configured and verified**

## Performance

- **Duration:** 11 min
- **Started:** 2026-02-21T22:40:28Z
- **Completed:** 2026-02-21T22:51:28Z
- **Tasks:** 3 (+ checkpoint pending)
- **Files modified:** 2

## Accomplishments
- VRAM budget validated empirically: Whisper distil-large-v3 + Ollama Llama 3.2 3B fit comfortably in 8GB with 2771MB headroom at concurrent peak
- VRAMMonitor module built with factory method, threshold levels, and structured stats output
- PipeWire WebRTC echo cancellation configured, verified recording from AEC source via pasimple

## VRAM Budget Measurements

| State | Used MB | Utilization | Delta |
|-------|---------|-------------|-------|
| Baseline (desktop idle) | 1464 | 17.9% | -- |
| + Whisper distil-large-v3 int8_float16 | 2580 | 31.5% | +1116 MB |
| + Ollama Llama 3.2 3B Q4_K_M | 5418 | 66.1% | +2838 MB |
| Concurrent inference peak | 5421 | 66.2% | -- |
| **Headroom at peak** | **2771** | **33.8% free** | -- |

**Go/no-go decision: GO** -- concurrent peak of 5421 MB is well below the WARNING threshold of 6144 MB. The fallback chain (reduce context, unload Ollama, CPU Whisper) exists but is unlikely to be needed under normal operation.

## PipeWire Echo Cancellation

- **Config file:** `~/.config/pipewire/pipewire.conf.d/echo-cancel.conf`
- **Module:** `libpipewire-module-echo-cancel` with `monitor.mode = true`
- **PulseAudio source name:** `Echo Cancellation Source` (with spaces)
- **pasimple device_name:** `"Echo Cancellation Source"` -- tested and confirmed working
- **Format:** float32le 2ch 48000Hz (PipeWire native), pasimple records at S16LE 1ch 24000Hz (resampled internally)

## Task Commits

1. **Task 1: VRAMMonitor module with tests** - `1f7c396` (feat)
2. **Task 2a: VRAM validation spike** - no repo commit (measurements only, documented above)
3. **Task 2b: PipeWire AEC configuration** - no repo commit (system config file at ~/.config/pipewire/)

## Files Created/Modified
- `vram_monitor.py` - NVML-based GPU memory watchdog with threshold levels and factory method
- `test_live_session.py` - Added 7 VRAMMonitor tests (mock-based, no GPU required)

## Decisions Made
- **VRAM budget: GO** -- 5421 MB peak concurrent usage leaves 2771 MB headroom (33.8%). WARNING threshold at 6144 MB won't trigger under normal operation.
- **AEC device name**: `Echo Cancellation Source` with spaces works directly in pasimple's device_name parameter. No need for underscore substitution.
- **Factory method pattern**: VRAMMonitor.create() returns None instead of raising, allowing graceful degradation on systems without NVIDIA GPU.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
- `ollama` shell alias interfered with direct invocation -- used full path `/home/ethan/.local/bin/ollama` instead
- llama3.2:3b was not pre-downloaded -- pulled during execution (2.0 GB download)
- nvidia-ml-py only available in service venv, not system Python -- verification commands used service venv path

## User Setup Required
None - PipeWire AEC config was applied and PipeWire restarted during execution.

## Next Phase Readiness
- VRAMMonitor ready for import by continuous STT pipeline (Plan 02)
- AEC device name `"Echo Cancellation Source"` ready for capture stage (Plan 03)
- VRAM budget confirms architecture is viable -- proceed with confidence
- Checkpoint pending user approval of VRAM measurements

---
*Phase: 12-infrastructure-safety-net*
*Completed: 2026-02-21*
