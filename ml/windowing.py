import numpy as np

def make_windows(x, window, hop):
    """x: 1D array -> (n_windows, window) array."""
    n = (len(x) - window) // hop + 1
    if n <= 0:
        return np.empty((0, window))
    idx = np.arange(window)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]