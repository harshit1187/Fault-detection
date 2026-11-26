# Edge Fault Detection

Real-time fault detection for a small motor/fan rig, using a tiny INT8-quantized 1D-CNN that runs on an STM32 microcontroller.

> **Status: work in progress.** The ML pipeline, INT8 export and on-board inference are working and verified. Real fault data has not been collected yet (motor hardware pending), so there are **no real-data accuracy results yet**. See [Results](#results).

## Idea

Detect faults (imbalance, loose mount, blocked airflow) from accelerometer and motor-current signals, on the microcontroller itself, with no cloud or laptop in the loop.

Design principles:

1. **Honest evaluation.** Splits are by recording session, never by random window. Neighbouring windows are near-duplicates, so a random split gives inflated scores.
2. **Deployability first.** The model must fit the MCU's memory and latency budget, so every model is judged on F1, size, RAM and latency together.
3. **Train/firmware parity.** Preprocessing in Python and in C must produce identical numbers. This is checked with a golden test vector.
4. **Measure, then optimize.** Optimizations are driven by on-board profiling.
5. **Report baselines honestly.** An FFT-feature baseline is always reported next to the neural network.

## System

```
Fan + MPU-6500 + current sense -> STM32F401RE (1 kHz sampling)
  -> 1024-sample window -> preprocess (int8) -> TinyCNN (INT8) -> class
```

| | Owner |
|---|---|
| Hardware, sensor drivers, firmware, X-CUBE-AI runs, on-board profiling | Sehaj |
| Data pipeline, models, quantization and export, evaluation, golden checks | Harshit |

## Data spec (v1)

- 1 kHz, timer-triggered. Raw integers only; conversion happens in the ML code.
- One CSV per recording: `sample_idx, t_us, ax, ay, az, gx, gy, gz, i_adc, duty_pct, flags`, with `# key=value` metadata lines at the top.
- Accelerometer: MPU-6500, +-2 g (16384 LSB/g). Current: 12-bit ADC, 3.3 V reference, 1 ohm shunt.
- Window 1024 samples (~1 s), hop 512. Model input is `(4, 1024)`: ax, ay, az, current.
- Labels: normal, imbalance, loose, blocked. Other run types: idle, normal_ramp, unlabeled.
- Metadata per recording: session, motor, mount, supply voltage, fault, severity, duty, `who_am_i`, accelerometer filter setting (`a_dlpf_cfg`).

## ML pipeline (my part)

| File (`ml/`) | Purpose |
|---|---|
| `configs/config.yaml` | Single source of settings (rate, window, hop, constants, classes) |
| `windowing.py` | Overlapping window slicing via NumPy broadcasting |
| `loader.py` | Reads recordings, drops spin-up, converts counts to g and amps, rejects windows with dropped samples, tags sessions |
| `features.py` | 16 features per channel: RMS, peak-to-peak, crest, kurtosis, skew, band energies, centroid, dominant peak, 1x/2x/3x harmonics, 2x/1x ratio |
| `baseline_v1.py` | FFT features + logistic regression / random forest, session split |
| `cnn.py` | TinyCNN (3 conv layers, global average pooling, linear head, 4,492 parameters) and `prep()` normalization |
| `train_cnn.py` | Training with a session split; saves weights and `norm.json` |
| `export_int8.py` | PyTorch to ONNX to static INT8 (QDQ), FP32 vs INT8 comparison |
| `export_qop.py` | QOperator export (conv-only variant uses standard ONNX ops only) |
| `make_int8_input.py` | Makes the model input INT8 (4 KB buffer instead of 16 KB), with an equivalence check |
| `gen_fd_norm.py` | Generates `fd_norm.h` (normalization and input scale) for the firmware |
| `verify_golden.py`, `make_golden_qop.py` | Golden-window checks: Python preprocessing vs firmware vs model |
| `check_idle.py`, `peaks.py` | Sanity checks on real recordings (timing, gravity, noise, spectral peaks) |
| `synth_data.py` | Synthetic generator
