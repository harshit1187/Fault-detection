import yaml, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from windowing import make_windows
from features import window_features

cfg = yaml.safe_load(open("configs/config.yaml"))
FS_RAW, FS = 12000, cfg["fs"]
DATA = Path("../data/cwru")

FILES = {
    "normal": {0: 97, 1: 98, 2: 99, 3: 100},
    "inner":  {0: 105, 1: 106, 2: 107, 3: 108},
    "outer":  {0: 130, 1: 131, 2: 132, 3: 133},
    "ball":   {0: 118, 1: 119, 2: 120, 3: 121},
}
CLASSES = list(FILES)

def load_signal(num):
    d = loadmat(DATA / f"{num}.mat")
    key = [k for k in d if k.endswith("DE_time")][0]
    x = d[key].flatten()
    return resample_poly(x, FS, FS_RAW)

def build(loads):
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for ld in loads:
            sig = load_signal(FILES[c][ld])
            for w in make_windows(sig, cfg["window"], cfg["hop"]):
                X.append(window_features(w, FS, 29.5, cfg["bands"]))
                y.append(ci)
    return np.array(X), np.array(y)

Xtr, ytr = build([0, 1, 2])
Xte, yte = build([3])
sc = StandardScaler().fit(Xtr)

clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                             random_state=cfg["seed"])
clf.fit(sc.transform(Xtr), ytr)
pred = clf.predict(sc.transform(Xte))

print("Macro F1 (train loads 0-2, test load 3):",
      round(f1_score(yte, pred, average="macro"), 3))
print(confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=CLASSES))
names = ["rms","p2p","crest","kurt","skew","b0","b1","b2","b3","cent","peakf","peaka","1x","2x","3x","ratio"]
for i in np.argsort(clf.feature_importances_)[::-1][:5]:
    print(names[i], round(float(clf.feature_importances_[i]), 3))