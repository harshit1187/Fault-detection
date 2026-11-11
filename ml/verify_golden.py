import json, numpy as np, pandas as pd, onnxruntime as ort
from cnn import prep

M = "../models/"
df = pd.read_csv("golden/golden_window.csv")          # note: no 3 s spin-up drop here
g = df[["ax", "ay", "az"]].to_numpy(np.float64) / 16384.0
amps = df["i_adc"].to_numpy(np.float64) * 3.3 / 4095.0
X = np.column_stack([g, amps])[None]                  # (1, 1024, 4)

n = json.load(open(M + "norm.json"))
stats = (np.array(n["mean"], np.float32).reshape(1, 4, 1),
         np.array(n["std"], np.float32).reshape(1, 4, 1))
x, _ = prep(X, stats)

sess = ort.InferenceSession(M + "tinycnn_int8.onnx", providers=["CPUExecutionProvider"])
logits = sess.run(None, {"x": x})[0][0]

exp = json.load(open("golden/golden_expected.json"))
ref = np.array(exp["expected_logits_float"])
print("my logits:      ", logits.round(4))
print("her expected:   ", ref.round(4))
print("max abs diff:   ", float(np.abs(logits - ref).max()), "(1 int8 LSB = 0.0414)")
print("argmax match:   ", int(logits.argmax()) == exp["expected_argmax"])