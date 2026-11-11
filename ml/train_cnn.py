import json, numpy as np, torch, torch.nn as nn, yaml
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
from loader import build_dataset
from cnn import TinyCNN, prep

cfg = yaml.safe_load(open("configs/config.yaml"))
torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
MODELS = Path("../models"); MODELS.mkdir(exist_ok=True)

X, y, groups, duty = build_dataset("../data/synthetic")
tr, te = np.isin(groups, ["s1", "s2"]), groups == "s3"

Xtr, stats = prep(X[tr])            # statistics from TRAIN only
Xte, _ = prep(X[te], stats)
ytr, yte = y[tr], y[te]

model = TinyCNN()
n_params = sum(p.numel() for p in model.parameters())
print("parameters:", n_params, "-> about", round(n_params / 1024, 1), "KB as INT8")

opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()
Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)

EPOCHS = 20                          # fixed, NOT chosen by looking at the test session
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Xt))
    total = 0.0
    for i in range(0, len(Xt), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(model(Xt[idx]), yt[idx])
        loss.backward(); opt.step()
        total += loss.item() * len(idx)
    print(f"epoch {ep + 1:2d}  train loss {total / len(Xt):.4f}")

model.eval()
with torch.no_grad():
    pred = model(torch.from_numpy(Xte)).argmax(1).numpy()
print("\nCNN FP32: macro F1 (train s1+s2, test s3) =",
      round(f1_score(yte, pred, average="macro"), 3))
print(confusion_matrix(yte, pred))

torch.save(model.state_dict(), MODELS / "tinycnn.pt")
json.dump({"mean": stats[0].flatten().tolist(), "std": stats[1].flatten().tolist(),
           "window": cfg["window"], "fs": cfg["fs"], "channels": cfg["channels"]},
          open(MODELS / "norm.json", "w"), indent=2)
print("saved model and norm.json")