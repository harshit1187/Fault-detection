"""Reference preprocessing + golden vector for tinycnn_int8.onnx (spec v1).
raw int16 accel/uint16 current -> float32 [1,4,1024] (channel-first, ax,ay,az,current)."""
import json, numpy as np

NORM = json.load(open("norm.json"))
MEAN = np.array(NORM["mean"], dtype=np.float32)
STD = np.array(NORM["std"], dtype=np.float32)
X_SCALE, X_ZP = 0.03954191, 0
W = 1024

def preprocess(ax, ay, az, i_adc):
    g = np.stack([ax, ay, az]).astype(np.float32) / 16384.0
    g -= g.mean(axis=1, keepdims=True)                       # per-window mean removal (accel only)
    amps = i_adc.astype(np.float32) * 3.3 / 4095.0 / 1.0      # 1 ohm shunt
    x = np.concatenate([g, amps[None]], axis=0)               # (4, 1024)
    x = (x - MEAN[:, None]) / STD[:, None]
    return x.astype(np.float32)

def quantize_input(x):
    return np.clip(np.rint(x / X_SCALE) + X_ZP, -128, 127).astype(np.int8)

if __name__ == "__main__":
    import onnxruntime as ort
    rng = np.random.default_rng(42)
    n = np.arange(W)
    ax = (700*np.sin(2*np.pi*40*n/1000) + rng.normal(0, 120, W) + 310).round().astype(np.int16)
    ay = (400*np.sin(2*np.pi*40*n/1000 + 1.0) + rng.normal(0, 120, W) - 150).round().astype(np.int16)
    az = (250*np.sin(2*np.pi*80*n/1000) + rng.normal(0, 120, W) + 16384).round().astype(np.int16)
    cur = (658 + 40*np.sin(2*np.pi*40*n/1000) + rng.normal(0, 25, W)).round().clip(0, 4095).astype(np.uint16)
    gx = rng.normal(0, 30, W).round().astype(np.int16); gy = gx[::-1].copy(); gz = np.zeros(W, np.int16)

    with open("golden_window.csv", "w") as f:
        f.write("sample_idx,t_us,ax,ay,az,gx,gy,gz,i_adc,duty_pct,flags\n")
        for k in range(W):
            f.write(f"{k},{k*1000},{ax[k]},{ay[k]},{az[k]},{gx[k]},{gy[k]},{gz[k]},{cur[k]},40,0\n")

    x = preprocess(ax, ay, az, cur)
    xq = quantize_input(x)
    s = ort.InferenceSession("tinycnn_int8.onnx", providers=["CPUExecutionProvider"])
    logits = s.run(None, {"x": x[None]})[0][0]
    L_SCALE, L_ZP = 0.04139081, -27
    lq = np.clip(np.rint(logits / L_SCALE) + L_ZP, -128, 127).astype(int)
    out = {
        "note": "Synthetic raw window; model weights are synthetic too, so logits mean nothing. Use only to check the board matches ONNX.",
        "input_layout": "channel-first [1,4,1024]: ax,ay,az,current; each channel contiguous",
        "input_float_first8": x[0,:8].round(6).tolist(),
        "input_int8_first8_ch0": xq[0,:8].tolist(),
        "input_int8_sum_per_channel": [int(xq[c].astype(np.int32).sum()) for c in range(4)],
        "expected_logits_float": logits.round(5).tolist(),
        "expected_logits_int8": lq.tolist(),
        "expected_argmax": int(np.argmax(logits)),
        "tolerance": "int8 logits within +/-1 LSB; argmax must match; per-channel input int8 sums should match exactly if rounding is half-to-even"
    }
    json.dump(out, open("golden_expected.json", "w"), indent=2)
    np.save("xq.npy", xq)
    print(json.dumps(out, indent=2))
