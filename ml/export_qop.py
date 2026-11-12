import sys, onnx, numpy as np, onnxruntime as ort, yaml
from pathlib import Path
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat, QuantType)
from sklearn.metrics import f1_score
from loader import build_dataset
from cnn import prep

cfg = yaml.safe_load(open("configs/config.yaml"))
M = Path("../models")
FP32 = M / "tinycnn_fp32.onnx"                 # made earlier by export_int8.py
variant = sys.argv[1] if len(sys.argv) > 1 else "all"   # "all" or "conv"
OUT = M / f"tinycnn_qop_{variant}.onnx"
ops = ["Conv"] if variant == "conv" else None  # "conv": pool/Gemm stay float

X, y, groups, duty = build_dataset("../data/synthetic")
tr, te = np.isin(groups, ["s1", "s2"]), groups == "s3"
Xtr, stats = prep(X[tr])
Xte, _ = prep(X[te], stats)
yte = y[te]

class Reader(CalibrationDataReader):
    def __init__(self, data):
        self.it = iter([{"x": w[None]} for w in data])
    def get_next(self):
        return next(self.it, None)

rng = np.random.default_rng(cfg["seed"])
calib = Xtr[rng.choice(len(Xtr), 200, replace=False)]
quantize_static(str(FP32), str(OUT), Reader(calib),
                quant_format=QuantFormat.QOperator,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True, op_types_to_quantize=ops)

used = sorted({(n.domain or "ai.onnx") + ":" + n.op_type for n in onnx.load(OUT).graph.node})
print("ops in graph:", used)
sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
pred = [sess.run(None, {"x": w[None]})[0].argmax() for w in Xte]
print("F1", round(f1_score(yte, pred, average="macro"), 3),
      "| size KB", round(OUT.stat().st_size / 1024, 1))