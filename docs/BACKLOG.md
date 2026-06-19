# Backlog

## Verify Orbbec hardware multi-camera sync

Status: backlog

Context observed on 2026-06-19:

```text
[2026-06-19 17:29:15,208] [INFO] recon3d.sensors.femto_bolt: [cam2] Femto Bolt started (sync=secondary_synced, align=hw)
[2026-06-19 17:29:16,590] [WARNING] recon3d.sensors.femto_bolt: [cam0] hardware D2C unavailable; falling back to software D2C: [cam0] no hardware-D2C-compatible color/depth profile pair found
[2026-06-19 17:29:17,131] [INFO] recon3d.sensors.femto_bolt: [cam0] Femto Bolt started (sync=primary, align=hw)
```

Notes:

- The warning above is specifically about hardware depth-to-color alignment (D2C), not definitive proof that hardware multi-device sync is disabled.
- Current logs show configured roles (`primary`, `secondary_synced`) but do not explicitly confirm that Orbbec multi-device hardware sync is active and producing shared-clock frames.
- Recording/replay can proceed with SDK timestamps and offline synchronization, but benchmark-grade capture should verify hardware sync separately.

Acceptance criteria:

- Log the applied Orbbec multi-device sync config per camera after `set_multi_device_sync_config`.
- Log success/failure of context-level `enable_multi_device_sync`.
- Add a short hardware diagnostic command or mode that reports sync role, timestamp source family, per-camera timestamp drift, and whether hardware sync appears active.
- Document the expected cabling/startup order and the difference between hardware D2C alignment and hardware multi-camera sync.
