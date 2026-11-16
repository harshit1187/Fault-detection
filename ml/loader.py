import numpy as np, pandas as pd, yaml
from pathlib import Path
from windowing import make_windows

cfg = yaml.safe_load(open("configs/config.yaml"))

def load_recording(path):
    df = pd.read_csv(path, comment="#")
    df = df[df["sample_idx"] >= cfg["drop_first_s"] * cfg["fs"]]    # drop spin-up
    g = df[["ax", "ay", "az"]].to_numpy(float) / cfg["accel_lsb_per_g"]
    full = 2 ** cfg["adc_bits"] - 1
    amps = df["i_adc"].to_numpy(float) * cfg["vref"] / full / cfg["shunt_ohm"]
    x = np.column_stack([g, amps])                                  # (T, 4)
    bad = ((df["flags"].to_numpy() & 1) > 0).astype(float)          # dropped samples
    return x, bad

def window_recording(x, bad):
    w, h = cfg["window"], cfg["hop"]
    W = np.stack([make_windows(x[:, c], w, h) for c in range(x.shape[1])], axis=-1)
    ok = make_windows(bad, w, h).sum(axis=1) == 0                   # skip windows with drops
    return W[ok]                                                    # (N, 1024, 4)

def build_dataset(data_dir):
    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / "metadata.csv")
    meta = meta[meta["fault"].isin(cfg["classes"])]
    X, y, groups, duty = [], [], [], []
    for _, r in meta.iterrows():
        W = window_recording(*load_recording(data_dir / r["file"]))
        X.append(W)
        y += [cfg["classes"].index(r["fault"])] * len(W)
        groups += [r["session_id"]] * len(W)
        duty += [r["duty_pct"]] * len(W)
    return np.concatenate(X), np.array(y), np.array(groups), np.array(duty)

if __name__ == "__main__":
    X, y, groups, duty = build_dataset("../data/synthetic")
    print("X", X.shape, "classes", np.bincount(y))
    for s in np.unique(groups):
        print(s, (groups == s).sum(), "windows")