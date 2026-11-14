import json, numpy as np, pandas as pd, onnxruntime as ort
from cnn import prep

M = "../models/"
SCALE = np.float32(0.03954191)
df = pd.read_csv("golden/golden_window.csv")
g = df[["ax", "ay", "az"]].to_numpy(np.float64) / 16384.0
amps = df["i_adc"].to_numpy(np.float64) * 3.3 / 4095.0
X = np.column_stack([g, amps])[None]

n = json.load(open(M + "norm.json"))
stats = (np.array(n["mean"], np.float32).reshape(1, 4, 1),
         np.array(n["std"], np.float32).reshape(1, 4, 1))
x, _ = prep(X, stats)
xq = np.clip(np.rint(x / SCALE), -128, 127).astype(np.int8)

s_in8 = ort.InferenceSession(M + "tinycnn_qop_conv_in8.onnx", providers=["CPUExecutionProvider"])
s_f = ort.InferenceSession(M + "tinycnn_qop_conv.onnx", providers=["CPUExecutionProvider"])
logits = s_in8.run(None, {"x": xq})[0][0]
ref = s_f.run(None, {"x": x})[0][0]
print("int8-input vs float-input max diff:", float(np.abs(logits - ref).max()))

out = {
    "model": "tinycnn_qop_conv_in8.onnx",
    "note": "Synthetic window and weights. Logits mean nothing; use only to check board == ONNX.",
    "input_layout": "channel-first [1,4,1024]: ax,ay,az,current, int8, scale 0.03954191, zp 0",
    "input_int8_first8_ch0": xq[0, 0, :8].tolist(),
    "input_int8_sum_per_channel": [int(xq[0, c].astype(np.int32).sum()) for c in range(4)],
    "expected_logits_float": [round(float(v), 6) for v in logits],
    "expected_argmax": int(logits.argmax()),
    "tolerance": "input int8 sums exact; logits within 1e-3; argmax must match",
}
json.dump(out, open("golden/golden_expected_qop_conv.json", "w"), indent=2)
print(json.dumps(out, indent=2))