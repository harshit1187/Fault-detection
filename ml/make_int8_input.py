import onnx, numpy as np
from onnx import TensorProto, numpy_helper

SRC, DST = "../models/tinycnn_int8.onnx", "../models/tinycnn_int8_in8.onnx"
m = onnx.load(SRC)
g = m.graph
inp = g.input[0]
q = next(n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] == inp.name)

inits = {i.name: i for i in g.initializer}
scale = float(numpy_helper.to_array(inits[q.input[1]]))
print("input scale:", scale)

for n in g.node:                                  # feed the INT8 input straight into the DQ
    for i, name in enumerate(n.input):
        if name == q.output[0]:
            n.input[i] = inp.name
g.node.remove(q)
inp.type.tensor_type.elem_type = TensorProto.INT8
onnx.checker.check_model(m)
onnx.save(m, DST)

# check: INT8 input model must reproduce the float-input model
import onnxruntime as ort
rng = np.random.default_rng(0)
xf = rng.normal(0, 1, (1, 4, 1024)).astype(np.float32)
xq = np.clip(np.rint(xf / scale), -128, 127).astype(np.int8)
a = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"]).run(None, {"x": xf})[0]
b = ort.InferenceSession(DST, providers=["CPUExecutionProvider"]).run(None, {"x": xq})[0]
print("max abs diff:", float(np.abs(a - b).max()), "| argmax match:", int(a.argmax()) == int(b.argmax()))