import numpy as np, pandas as pd, yaml
from pathlib import Path

cfg = yaml.safe_load(open("configs/config.yaml"))
FS = cfg["fs"]
OUT = Path("../data/synthetic")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(cfg["seed"])

FAULTS = ["normal", "imbalance", "loose", "blocked"]
DUTY_HZ = {30: 15, 40: 20, 50: 25, 60: 30}   # fake shaft frequency per duty

def make_recording(fault, duty, seconds=20):
    n = seconds * FS
    t = np.arange(n) / FS
    f0 = DUTY_HZ[duty] * (1 + 0.01 * rng.standard_normal())
    amp = {"normal": 0.02, "imbalance": 0.08, "loose": 0.03, "blocked": 0.02}[fault]
    noise = 0.05 if fault == "blocked" else 0.02

    acc = []
    for ph in (0.0, 1.0, 2.0):
        s = amp * np.sin(2 * np.pi * f0 * t + ph)
        if fault == "loose":
            s += 0.03 * np.sin(2 * np.pi * 2 * f0 * t) + 0.02 * np.sin(2 * np.pi * 3 * f0 * t)
        acc.append(s + noise * rng.standard_normal(n))
    acc[2] += 1.0                                   # gravity on z

    amps = 0.3 + duty / 200 + (0.03 if fault == "blocked" else 0.0)
    amps = amps + 0.01 * rng.standard_normal(n)

    df = pd.DataFrame({
        "sample_idx": np.arange(n, dtype=np.uint32),
        "t_us": (np.arange(n) * 1000 + rng.integers(0, 5, n)).astype(np.uint32),
        "ax": np.clip(acc[0] * 16384, -32768, 32767).astype(np.int16),
        "ay": np.clip(acc[1] * 16384, -32768, 32767).astype(np.int16),
        "az": np.clip(acc[2] * 16384, -32768, 32767).astype(np.int16),
        "gx": rng.integers(-20, 20, n).astype(np.int16),
        "gy": rng.integers(-20, 20, n).astype(np.int16),
        "gz": rng.integers(-20, 20, n).astype(np.int16),
        "i_adc": np.clip(amps * 4095 / 3.3, 0, 4095).astype(np.uint16),
        "duty_pct": np.uint8(duty),
        "flags": np.uint8(0),
    })
    return df

rows = []
for s in range(1, 4):                               # 3 sessions
    for fault in FAULTS:
        for duty in DUTY_HZ:
            name = f"s{s}_{fault}_d{duty}.csv"
            make_recording(fault, duty).to_csv(OUT / name, index=False)
            rows.append(dict(file=name, session_id=f"s{s}", date="2026-11-11",
                             motor_id="synthetic", mount_id="AB"[s % 2],
                             sensor_orient="z-up", supply_v=5.0, fault=fault,
                             severity="", duty_pct=duty, duration_s=20,
                             fs_hz=FS, accel_range_g=2, fw_version="synth",
                             notes=""))
pd.DataFrame(rows).to_csv(OUT / "metadata.csv", index=False)
print("wrote", len(rows), "recordings to", OUT)