import yaml, numpy as np
from windowing import make_windows
from features import window_features

cfg = yaml.safe_load(open("configs/config.yaml"))
fs, rpm = cfg["fs"], cfg["rpm_hz"]
t = np.arange(0, 10, 1 / fs)

healthy = np.sin(2 * np.pi * rpm * t) + 0.1 * np.random.randn(len(t))
imbal = 3 * np.sin(2 * np.pi * rpm * t) + 0.1 * np.random.randn(len(t))

for name, sig in [("healthy", healthy), ("imbalance", imbal)]:
    w = make_windows(sig, cfg["window"], cfg["hop"])
    f = window_features(w[0], fs, rpm, cfg["bands"])
    print(name, w.shape, "features:", len(f), "1x amp:", round(float(f[-4]), 1))