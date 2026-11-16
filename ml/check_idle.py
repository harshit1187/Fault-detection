import sys, numpy as np, pandas as pd
for p in sys.argv[1:]:
    df = pd.read_csv(p, comment="#")
    dt = np.diff(df["t_us"].to_numpy(np.int64))
    g = df[["ax", "ay", "az"]].to_numpy(float) / 16384
    print(p, len(df), "samples | dt min/max us:", dt.min(), dt.max(),
          "| flags set:", int((df["flags"] > 0).sum()))
    print("  mean g:", g.mean(0).round(4), "| std g:", g.std(0).round(5))
    a = g[:, 0] - g[:, 0].mean()
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))); f = np.fft.rfftfreq(len(a), 1e-3)
    top = np.argsort(sp[1:])[::-1][:5] + 1
    print("  top ax peaks (Hz):", sorted(f[top].round(1)))
    print("  i_adc mean/std:", round(df["i_adc"].mean(), 1), round(df["i_adc"].std(), 2))