#!/usr/bin/env python3
"""Train the SILEX-1D reference model and freeze it to integer form.

Model: 196 (14x14 avg-pooled MNIST) -> 48 ReLU -> 10 logits.
Trained in float32, then post-training quantized to the exact integer
arithmetic the silicon implements:

  layer1: acc32 = sum(int8_w * uint8_x) + b32
          relu:  acc32 = max(acc32, 0)
          requant: a8 = clamp(round_shift(acc32 * M1, SH1), 0, 255)   (uint8)
  layer2: logit32 = sum(int8_w * uint8_a) + b32
          class = argmax(logit32)

round_shift(v, s) = (v + (1 << (s-1))) >> s   (arithmetic, round-half-up)

Outputs (build/):
  model.npz          frozen integer model + metadata
  w1.hex b1.hex w2.hex b2.hex   ROM images for RTL ($readmemh)
  golden.hex         BIST vector: one image (196 bytes) + expected class
  test_vectors.bin   1000 test images (uint8) for the testbench
  test_expected.txt  expected class per test image (integer reference)
"""
import numpy as np
import os
import struct

RNG = np.random.default_rng(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BUILD = os.path.join(ROOT, "build")
os.makedirs(BUILD, exist_ok=True)

H = 48          # hidden width
NIN = 196       # 14x14
NOUT = 10


def load_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def load_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)


def pool14(imgs):
    """28x28 -> 14x14 by 2x2 mean, kept as uint8 (this is what the chip sees)."""
    x = imgs.reshape(-1, 14, 2, 14, 2).astype(np.uint16)
    return (x.sum(axis=(2, 4)) // 4).astype(np.uint8).reshape(-1, NIN)


def train_float(xtr, ytr, xte, yte):
    """Two-layer MLP, SGD with momentum. Inputs already scaled to [0,1]."""
    w1 = RNG.normal(0, np.sqrt(2.0 / NIN), (NIN, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    w2 = RNG.normal(0, np.sqrt(2.0 / H), (H, NOUT)).astype(np.float32)
    b2 = np.zeros(NOUT, np.float32)
    vw1 = np.zeros_like(w1); vb1 = np.zeros_like(b1)
    vw2 = np.zeros_like(w2); vb2 = np.zeros_like(b2)
    lr, mom, bs = 0.08, 0.9, 128
    onehot = np.eye(NOUT, dtype=np.float32)[ytr]
    for epoch in range(15):
        idx = RNG.permutation(len(xtr))
        for k in range(0, len(xtr), bs):
            j = idx[k:k + bs]
            x, t = xtr[j], onehot[j]
            h_pre = x @ w1 + b1
            h = np.maximum(h_pre, 0)
            z = h @ w2 + b2
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            dz = (p - t) / len(j)
            dw2 = h.T @ dz; db2 = dz.sum(0)
            dh = dz @ w2.T; dh[h_pre <= 0] = 0
            dw1 = x.T @ dh; db1 = dh.sum(0)
            for v, g, p_ in ((vw1, dw1, w1), (vb1, db1, b1),
                             (vw2, dw2, w2), (vb2, db2, b2)):
                v *= mom; v -= lr * g; p_ += v
        acc = float((np.argmax(np.maximum(xte @ w1 + b1, 0) @ w2 + b2, 1) == yte).mean())
        print(f"epoch {epoch:2d}  float test acc {acc:.4f}")
    return w1, b1, w2, b2, acc


def quantize(w1f, b1f, w2f, b2f, xcal_u8):
    """Post-training quantization to the exact silicon arithmetic."""
    s_x = 1.0 / 255.0                                   # uint8 pixel scale
    s_w1 = float(np.max(np.abs(w1f))) / 127.0
    w1 = np.clip(np.round(w1f / s_w1), -127, 127).astype(np.int8)
    b1 = np.round(b1f / (s_x * s_w1)).astype(np.int32)

    # calibrate hidden activation scale on training data (float relu output)
    hf = np.maximum(xcal_u8.astype(np.float32) * s_x @ w1f + b1f, 0)
    s_a = float(np.percentile(hf, 99.99)) / 255.0        # uint8 activation scale

    # requant factor: acc32 * (s_x*s_w1) -> a8 * s_a   =>  M/2^SH = s_x*s_w1/s_a
    SH1 = 24
    M1 = int(round(s_x * s_w1 / s_a * (1 << SH1)))
    assert 0 < M1 < (1 << 31)

    s_w2 = float(np.max(np.abs(w2f))) / 127.0
    w2 = np.clip(np.round(w2f / s_w2), -127, 127).astype(np.int8)
    b2 = np.round(b2f / (s_a * s_w2)).astype(np.int32)
    return dict(w1=w1, b1=b1, w2=w2, b2=b2, M1=M1, SH1=SH1,
                s_x=s_x, s_w1=s_w1, s_a=s_a, s_w2=s_w2)


def int_forward(q, x_u8):
    """Bit-exact integer reference = RTL golden model."""
    acc = x_u8.astype(np.int64) @ q["w1"].astype(np.int64) + q["b1"]
    acc = np.maximum(acc, 0)
    a = (acc * q["M1"] + (1 << (q["SH1"] - 1))) >> q["SH1"]
    a = np.clip(a, 0, 255)
    logits = a @ q["w2"].astype(np.int64) + q["b2"]
    return np.argmax(logits, axis=1), logits


def write_hex8(path, arr_i8):
    with open(path, "w") as f:
        for v in arr_i8.astype(np.int64).ravel():
            f.write(f"{int(v) & 0xFF:02x}\n")


def write_hex32(path, arr_i32):
    with open(path, "w") as f:
        for v in arr_i32.astype(np.int64).ravel():
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")


def main():
    xtr_img = load_idx_images(os.path.join(DATA, "train-images-idx3-ubyte"))
    ytr = load_idx_labels(os.path.join(DATA, "train-labels-idx1-ubyte"))
    xte_img = load_idx_images(os.path.join(DATA, "t10k-images-idx3-ubyte"))
    yte = load_idx_labels(os.path.join(DATA, "t10k-labels-idx1-ubyte"))
    xtr_u8, xte_u8 = pool14(xtr_img), pool14(xte_img)

    w1f, b1f, w2f, b2f, facc = train_float(
        xtr_u8.astype(np.float32) / 255.0, ytr,
        xte_u8.astype(np.float32) / 255.0, yte)

    q = quantize(w1f, b1f, w2f, b2f, xtr_u8[:10000])
    pred, _ = int_forward(q, xte_u8)
    qacc = float((pred == yte).mean())
    print(f"float acc {facc:.4f} -> int8 acc {qacc:.4f}  (M1={q['M1']}, SH1={q['SH1']})")

    np.savez(os.path.join(BUILD, "model.npz"), **q)
    # ROMs: w1 stored neuron-major (all 196 weights of neuron 0, then neuron 1 ...)
    write_hex8(os.path.join(BUILD, "w1.hex"), q["w1"].T)
    write_hex32(os.path.join(BUILD, "b1.hex"), q["b1"])
    write_hex8(os.path.join(BUILD, "w2.hex"), q["w2"].T)
    write_hex32(os.path.join(BUILD, "b2.hex"), q["b2"])

    # BIST golden vector: first test image + its integer-reference class
    write_hex8(os.path.join(BUILD, "golden.hex"), xte_u8[0].view(np.int8))
    with open(os.path.join(BUILD, "golden_class.txt"), "w") as f:
        f.write(str(int(pred[0])))

    n = 1000
    xte_u8[:n].tofile(os.path.join(BUILD, "test_vectors.bin"))
    with open(os.path.join(BUILD, "test_expected.txt"), "w") as f:
        for c in pred[:n]:
            f.write(f"{int(c)}\n")
    print(f"frozen model + ROMs + {n} test vectors written to build/")


if __name__ == "__main__":
    main()
