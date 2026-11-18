import sys, numpy as np, pandas as pd
from scipy.signal import welch, find_peaks

for p in sys.argv[1:]:
    df = pd.read_csv(p, comment="#")
    g = df[["ax", "ay", "az"]].to_numpy(float) / 16384
    print(p, len(df), "samples")
    for c, name in enumerate(["ax", "ay", "az"]):
        x = g[:, c] - g[:, c].mean()
        f, psd = welch(x, fs=1000, nperseg=8192)          # 0.12 Hz resolution, averaged
        floor = np.median(psd[(f > 5)])
        idx, _ = find_peaks(psd, height=floor * 10)         # peaks >= 10 dB above floor
        idx = idx[f[idx] > 5]
        print(f"  {name}: noise floor {10*np.log10(floor):.1f} dB (g^2/Hz)")
        for i in idx[np.argsort(psd[idx])[::-1][:8]]:
            print(f"    {f[i]:7.2f} Hz   {10*np.log10(psd[i]/floor):5.1f} dB above floor")