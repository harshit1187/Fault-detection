import numpy as np, yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from loader import build_dataset
from features import multi_axis_features

cfg = yaml.safe_load(open("configs/config.yaml"))
FS = cfg["fs"]

X, y, groups, duty = build_dataset("../data/synthetic")

def estimate_f0(w):
    """Shaft frequency estimate: strongest peak of the ax spectrum (2-100 Hz)."""
    a = w[:, 0] - w[:, 0].mean()
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    freqs = np.fft.rfftfreq(len(a), 1 / FS)
    m = (freqs >= 2) & (freqs <= 100)
    return freqs[m][spec[m].argmax()]

def featurize(X):
    out = []
    for w in X:
        f0 = estimate_f0(w)
        out.append(multi_axis_features(w, FS, f0, cfg["bands"]))
    return np.array(out)

F = featurize(X)
print("features", F.shape)

tr, te = np.isin(groups, ["s1", "s2"]), groups == "s3"
sc = StandardScaler().fit(F[tr])

models = {
    "logreg": LogisticRegression(max_iter=2000),
    "rf": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=cfg["seed"]),
}
for name, clf in models.items():
    clf.fit(sc.transform(F[tr]), y[tr])
    pred = clf.predict(sc.transform(F[te]))
    print(f"\n{name}: macro F1 (train s1+s2, test s3) =",
          round(f1_score(y[te], pred, average="macro"), 3))
    print(confusion_matrix(y[te], pred))