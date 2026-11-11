import numpy as np, torch, onnxruntime as ort, yaml
from pathlib import Path
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat, QuantType)
from sklearn.metrics import f1_score
from loader import build_dataset
from cnn import TinyCNN, prep

cfg = yaml.safe_load(open("configs/config.yaml"))
M = Path("../models")
FP32, INT8 = M / "tinycnn_fp32.onnx", M / "tinycnn_int8.onnx"

# 1. export FP32 ONNX (batch size 1, fixed shape, as on the MCU)
model = TinyCNN()
model.load_state_dict(torch.load(M / "tinycnn.pt"))
model.eval()
torch.onnx.export(model, torch.zeros(1, 4, cfg["window"]), str(FP32),
                  input_names=["x"], output_names=["logits"],
                  opset_version=13, dynamo=False)

# 2. data (same split and stats as training)
X, y, groups, duty = build_dataset("../data/synthetic")
tr, te = np.isin(groups, ["s1", "s2"]), groups == "s3"
Xtr, stats = prep(X[tr])
Xte, _ = prep(X[te], stats)
yte = y[te]

# 3. static INT8 quantization, calibrated on 200 TRAIN windows
class Reader(CalibrationDataReader):
    def __init__(self, data):
        self.it = iter([{"x": w[None]} for w in data])
    def get_next(self):
        return next(self.it, None)

rng = np.random.default_rng(cfg["seed"])
calib = Xtr[rng.choice(len(Xtr), 200, replace=False)]
quantize_static(str(FP32), str(INT8), Reader(calib),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True)

# 4. compare FP32 vs INT8 on the held-out session
def evaluate(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pred = [sess.run(None, {"x": w[None]})[0].argmax() for w in Xte]
    return f1_score(yte, pred, average="macro")

f_fp32, f_int8 = evaluate(FP32), evaluate(INT8)
print(f"\nFP32 ONNX: F1 {f_fp32:.3f}, size {FP32.stat().st_size / 1024:.1f} KB")
print(f"INT8 ONNX: F1 {f_int8:.3f}, size {INT8.stat().st_size / 1024:.1f} KB")
print(f"F1 drop from quantization: {f_fp32 - f_int8:.3f}")