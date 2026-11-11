import numpy as np
from scipy.stats import kurtosis, skew

def window_features(w, fs, rpm_hz, bands):
    """Features for one 1D window (one axis)."""
    w = w - w.mean()
    rms = np.sqrt(np.mean(w ** 2))
    p2p = w.max() - w.min()
    crest = np.max(np.abs(w)) / (rms + 1e-12)
    feats = [rms, p2p, crest, kurtosis(w), skew(w)]

    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    freqs = np.fft.rfftfreq(len(w), d=1 / fs)
    total = spec.sum() + 1e-12

    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        feats.append((spec[m] ** 2).sum())

    feats.append((freqs * spec).sum() / total)
    k = spec[1:].argmax() + 1
    feats += [freqs[k], spec[k]]

    amps = []
    for h in (1, 2, 3):
        m = np.abs(freqs - h * rpm_hz) <= 2 * (freqs[1] - freqs[0])
        amps.append(spec[m].max() if m.any() else 0.0)
    feats += amps
    feats.append(amps[1] / (amps[0] + 1e-12))
    return np.array(feats, dtype=np.float32)

def multi_axis_features(win, fs, rpm_hz, bands):
    """win: (window, n_axes) -> concatenated features."""
    return np.concatenate([window_features(win[:, a], fs, rpm_hz, bands)
                           for a in range(win.shape[1])])